"""
Abstract detector interface and concrete implementations for GPTZero and ZeroGPT.

This module defines the :class:`DetectorClient` abstract base class that every AI
text detector must implement, plus two production implementations:

- :class:`GPTZeroClient` -- GPTZero API v2 (rich per-sentence data).
- :class:`ZeroGPTClient` -- ZeroGPT API   (lower cost, coarser sentence data).

The abstract interface makes it trivial to swap detectors or add new ones (e.g.
Originality.ai, Sapling) without changing the experiment runner.

Data Classes
------------
- :class:`SentenceScore` -- per-sentence AI probability + perplexity.
- :class:`DetectorResult` -- standardised output from any detector.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceScore:
    """
    AI detection score for a single sentence.

    @param sentence:       The original sentence text.
    @param generated_prob: Probability (0-1) that this sentence was AI-generated.
    @param perplexity:     Token-level perplexity (0.0 when the detector does not
                           provide it, e.g. ZeroGPT).
    """

    sentence: str
    generated_prob: float
    perplexity: float


@dataclass(frozen=True)
class DetectorResult:
    """
    Standardised output from any AI text detector.

    Every concrete :class:`DetectorClient` maps its raw API response into this
    common shape so the experiment runner and analysis code never need to know
    which detector was used.

    @param overall_ai_prob:      Overall probability (0-1) that the text is
                                 AI-generated.  This is the primary dependent
                                 variable in the experiment.
    @param burstiness:           Sentence-complexity variance (``None`` when the
                                 detector does not report it).
    @param per_sentence_scores:  One :class:`SentenceScore` per sentence.
    @param flagged_sentence_pct: Percentage of sentences individually flagged as
                                 AI-generated.
    @param raw_response:         The full API JSON response, preserved for
                                 debugging and logging.
    """

    overall_ai_prob: float
    burstiness: float | None
    per_sentence_scores: list[SentenceScore]
    flagged_sentence_pct: float
    raw_response: dict


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class DetectorClient(ABC):
    """
    Abstract base for AI text detectors.

    Subclass this for each detector API and implement :meth:`name` and
    :meth:`check`.  The experiment runner programmes against this interface,
    so detectors are interchangeable.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short identifier for this detector (e.g. ``'gptzero'``, ``'zerogpt'``).

        Used in log messages and as the ``detector`` field in result records.
        """

    @abstractmethod
    def check(self, text: str) -> DetectorResult:
        """
        Analyse *text* and return a :class:`DetectorResult`.

        @param text: The essay to analyse.
        @returns:    A :class:`DetectorResult` with scores and raw response.
        """


# ---------------------------------------------------------------------------
# ZeroGPT implementation
# ---------------------------------------------------------------------------


class ZeroGPTClient(DetectorClient):
    """
    ZeroGPT API implementation.

    ZeroGPT returns a ``fakePercentage`` (0-100) and a list ``h`` of sentences it
    considers AI-generated.  This client normalises those values into the common
    :class:`DetectorResult` format.

    Note: ZeroGPT does not provide per-sentence perplexity or burstiness, so
    those fields are set to ``0.0`` and ``None`` respectively.
    """

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "",
    ):
        """
        Initialise the ZeroGPT client.

        @param api_key: ZeroGPT API key.  Falls back to ``config.ZEROGPT_API_KEY``
                        (loaded from the ``ZEROGPT_API_KEY`` env var) when empty.
        @param api_url: Override the default endpoint URL (useful for testing).
        @raises ValueError: If no API key is available.
        """

        key = api_key or config.ZEROGPT_API_KEY

        if not key:
            raise ValueError(
                "ZEROGPT_API_KEY is not set. Add it to .env or pass it explicitly."
            )

        self._api_url = api_url or config.ZEROGPT_API_URL

        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "ApiKey": key,
        }

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Return the short detector identifier ``'zerogpt'``."""
        return "zerogpt"

    def check(self, text: str) -> DetectorResult:
        """
        Send *text* to ZeroGPT and return a normalised :class:`DetectorResult`.

        Retries on transient errors (HTTP 429, 5xx, connection failures) with
        exponential backoff.

        @param text: The essay to analyse.
        @returns:    A :class:`DetectorResult`.
        """

        raw = self._call_with_retry(text)
        return self._parse_response(raw, text)

    # ── HTTP layer with retry ──────────────────────────────────────────────

    def _call_with_retry(self, text: str) -> dict:
        """
        POST to ZeroGPT with exponential backoff on transient failures.

        @param text: The essay text to send in the request body.
        @returns:    The parsed JSON response dict.
        @raises RuntimeError: After ``MAX_RETRIES`` consecutive failures.
        """

        last_exc: Exception | None = None
        payload = {"input_text": text}

        for retry in range(config.MAX_RETRIES):

            try:
                # -- Fire the HTTP request ----------------------------------
                with httpx.Client(timeout=30) as client:
                    resp = client.post(
                        self._api_url,
                        json=payload,
                        headers=self._headers,
                    )

                # -- Handle rate-limiting (429) -- retry after backoff ------
                if resp.status_code == 429:
                    delay = config.RETRY_BASE_DELAY_S * (2**retry)

                    logger.warning(
                        "ZeroGPT rate limited (429), backing off %.1fs (retry %d/%d)",
                        delay,
                        retry + 1,
                        config.MAX_RETRIES,
                    )

                    time.sleep(delay)
                    continue

                # -- Raise on any other non-2xx status ----------------------
                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as exc:
                last_exc = exc

                # Server errors (5xx) are transient -- retry.
                if exc.response.status_code >= 500:
                    delay = config.RETRY_BASE_DELAY_S * (2**retry)

                    logger.warning(
                        "ZeroGPT server error %d, backing off %.1fs (retry %d/%d)",
                        exc.response.status_code,
                        delay,
                        retry + 1,
                        config.MAX_RETRIES,
                    )

                    time.sleep(delay)
                    continue

                # Client errors (4xx other than 429) are not retryable.
                raise

            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_exc = exc
                delay = config.RETRY_BASE_DELAY_S * (2**retry)

                logger.warning(
                    "ZeroGPT connection error, backing off %.1fs (retry %d/%d): %s",
                    delay,
                    retry + 1,
                    config.MAX_RETRIES,
                    exc,
                )

                time.sleep(delay)

        raise RuntimeError(
            f"ZeroGPT API failed after {config.MAX_RETRIES} retries"
        ) from last_exc

    # ── Response parsing ───────────────────────────────────────────────────

    def _parse_response(self, raw: dict, original_text: str) -> DetectorResult:
        """
        Extract a :class:`DetectorResult` from ZeroGPT's raw JSON response.

        ZeroGPT returns:

        - ``fakePercentage``  (0-100 float) -- overall AI probability.
        - ``h``               (list[str])   -- sentences the model considers
          AI-generated (exact sentence text).

        We split *original_text* into sentences and check membership in ``h``
        to build per-sentence binary scores (1.0 = flagged, 0.0 = not flagged).

        @param raw:           The parsed JSON response from ZeroGPT.
        @param original_text: The essay that was submitted (needed to reconstruct
                              per-sentence scores).
        @returns: A :class:`DetectorResult`.
        """

        # -- Overall AI probability (0-100 -> 0-1) -------------------------
        fake_pct = raw.get("fakePercentage", 0.0)
        overall_ai_prob = fake_pct / 100.0

        # -- Flagged sentences from ZeroGPT's "h" array --------------------
        ai_sentences: list[str] = raw.get("h", [])
        ai_sentences_set = set(ai_sentences)

        # -- Split the original essay into sentences for scoring ------------
        all_sentences = self._split_sentences(original_text)

        # -- Build per-sentence scores (binary: in "h" or not) --------------
        sentences: list[SentenceScore] = []

        for sent in all_sentences:
            is_flagged = sent in ai_sentences_set

            sentences.append(
                SentenceScore(
                    sentence=sent,
                    generated_prob=1.0 if is_flagged else 0.0,
                    perplexity=0.0,  # ZeroGPT does not provide perplexity
                )
            )

        # -- Compute flagged-sentence percentage ----------------------------
        flagged = len(ai_sentences)
        flagged_pct = (flagged / len(all_sentences) * 100) if all_sentences else 0.0

        return DetectorResult(
            overall_ai_prob=overall_ai_prob,
            burstiness=None,  # ZeroGPT does not report burstiness
            per_sentence_scores=sentences,
            flagged_sentence_pct=flagged_pct,
            raw_response=raw,
        )

    # ── Sentence splitting helper ──────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Naive sentence splitting on common terminators (``.``, ``!``, ``?``).

        This is intentionally simple -- it only needs to match the exact strings
        ZeroGPT returns in its ``h`` array.

        @param text: The full essay text.
        @returns:    A list of non-empty sentence strings.
        """

        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in raw if s]


# ---------------------------------------------------------------------------
# GPTZero implementation
# ---------------------------------------------------------------------------


class GPTZeroClient(DetectorClient):
    """
    GPTZero API v2 implementation.

    GPTZero provides richer per-sentence data than ZeroGPT: each sentence gets
    a continuous ``generated_prob`` (0-1) and a ``perplexity`` score.  The
    document-level response also includes ``overall_burstiness``.

    Sentences with ``generated_prob`` above :attr:`SENTENCE_FLAG_THRESHOLD` are
    counted as individually flagged for the ``flagged_sentence_pct`` metric.
    """

    # Sentences with generated_prob above this are counted as "flagged".
    SENTENCE_FLAG_THRESHOLD = 0.5

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "",
    ):
        """
        Initialise the GPTZero client.

        @param api_key: GPTZero API key.  Falls back to ``config.GPTZERO_API_KEY``
                        (loaded from the ``GPTZERO_API_KEY`` env var) when empty.
        @param api_url: Override the default endpoint URL (useful for testing).
        @raises ValueError: If no API key is available.
        """

        key = api_key or config.GPTZERO_API_KEY

        if not key:
            raise ValueError(
                "GPTZERO_API_KEY is not set. Add it to .env or pass it explicitly."
            )

        self._api_url = api_url or config.GPTZERO_API_URL

        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": key,
        }

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Return the short detector identifier ``'gptzero'``."""
        return "gptzero"

    def check(self, text: str) -> DetectorResult:
        """
        Send *text* to GPTZero and return a normalised :class:`DetectorResult`.

        Retries on transient errors (HTTP 429, 5xx, connection failures) with
        exponential backoff.

        @param text: The essay to analyse.
        @returns:    A :class:`DetectorResult`.
        """

        raw = self._call_with_retry(text)
        return self._parse_response(raw)

    # ── HTTP layer with retry ──────────────────────────────────────────────

    def _call_with_retry(self, text: str) -> dict:
        """
        POST to GPTZero with exponential backoff on transient failures.

        @param text: The essay text to send in the request body.
        @returns:    The parsed JSON response dict.
        @raises RuntimeError: After ``MAX_RETRIES`` consecutive failures.
        """

        last_exc: Exception | None = None
        payload = {"document": text}

        for retry in range(config.MAX_RETRIES):

            try:
                # -- Fire the HTTP request ----------------------------------
                with httpx.Client(timeout=30) as client:
                    resp = client.post(
                        self._api_url,
                        json=payload,
                        headers=self._headers,
                    )

                # -- Handle rate-limiting (429) -- retry after backoff ------
                if resp.status_code == 429:
                    delay = config.RETRY_BASE_DELAY_S * (2**retry)

                    logger.warning(
                        "GPTZero rate limited (429), backing off %.1fs (retry %d/%d)",
                        delay,
                        retry + 1,
                        config.MAX_RETRIES,
                    )

                    time.sleep(delay)
                    continue

                # -- Raise on any other non-2xx status ----------------------
                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as exc:
                last_exc = exc

                # Server errors (5xx) are transient -- retry.
                if exc.response.status_code >= 500:
                    delay = config.RETRY_BASE_DELAY_S * (2**retry)

                    logger.warning(
                        "GPTZero server error %d, backing off %.1fs (retry %d/%d)",
                        exc.response.status_code,
                        delay,
                        retry + 1,
                        config.MAX_RETRIES,
                    )

                    time.sleep(delay)
                    continue

                # Client errors (4xx other than 429) are not retryable.
                raise

            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_exc = exc
                delay = config.RETRY_BASE_DELAY_S * (2**retry)

                logger.warning(
                    "GPTZero connection error, backing off %.1fs (retry %d/%d): %s",
                    delay,
                    retry + 1,
                    config.MAX_RETRIES,
                    exc,
                )

                time.sleep(delay)

        raise RuntimeError(
            f"GPTZero API failed after {config.MAX_RETRIES} retries"
        ) from last_exc

    # ── Response parsing ───────────────────────────────────────────────────

    def _parse_response(self, raw: dict) -> DetectorResult:
        """
        Extract a :class:`DetectorResult` from GPTZero's raw JSON response.

        GPTZero nests everything under ``documents[0]``.  Each sentence in the
        ``sentences`` array carries a continuous ``generated_prob`` and a
        ``perplexity`` value.

        @param raw: The parsed JSON response from GPTZero.
        @returns:   A :class:`DetectorResult`.
        """

        doc = raw["documents"][0]

        # -- Build per-sentence scores from GPTZero's detailed array --------
        sentences: list[SentenceScore] = []

        for s in doc.get("sentences", []):
            sentences.append(
                SentenceScore(
                    sentence=s.get("sentence", ""),
                    generated_prob=s.get("generated_prob", 0.0),
                    perplexity=s.get("perplexity", 0.0),
                )
            )

        # -- Count flagged sentences (prob > threshold) ---------------------
        flagged = sum(
            1 for s in sentences
            if s.generated_prob > self.SENTENCE_FLAG_THRESHOLD
        )

        flagged_pct = (flagged / len(sentences) * 100) if sentences else 0.0

        return DetectorResult(
            overall_ai_prob=doc.get("completely_generated_prob", 0.0),
            burstiness=doc.get("overall_burstiness", None),
            per_sentence_scores=sentences,
            flagged_sentence_pct=flagged_pct,
            raw_response=raw,
        )
