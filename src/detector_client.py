"""Abstract detector interface + GPTZero implementation."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx

from src import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentenceScore:
    """AI detection score for a single sentence."""

    sentence: str
    generated_prob: float
    perplexity: float


@dataclass(frozen=True)
class DetectorResult:
    """Standardized output from any AI text detector."""

    overall_ai_prob: float  # 0-1, primary metric (completely_generated_prob for GPTZero)
    burstiness: float | None  # Sentence complexity variance
    per_sentence_scores: list[SentenceScore]
    flagged_sentence_pct: float  # % of sentences individually flagged as AI
    raw_response: dict  # Full API response for logging/debugging


class DetectorClient(ABC):
    """Abstract base for AI text detectors. Subclass for each detector API."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this detector (e.g. 'gptzero', 'zerogpt')."""

    @abstractmethod
    def check(self, text: str) -> DetectorResult:
        """Analyze text and return a DetectorResult."""


class ZeroGPTClient(DetectorClient):
    """ZeroGPT API implementation."""

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "",
    ):
        key = api_key or config.ZEROGPT_API_KEY
        if not key:
            raise ValueError("ZEROGPT_API_KEY is not set. Add it to .env or pass it explicitly.")
        self._api_url = api_url or config.ZEROGPT_API_URL
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "ApiKey": key,
        }

    @property
    def name(self) -> str:
        return "zerogpt"

    def check(self, text: str) -> DetectorResult:
        """Send text to ZeroGPT and parse the response into a DetectorResult.

        Retries on transient errors (429, 5xx) with exponential backoff.
        """
        raw = self._call_with_retry(text)
        return self._parse_response(raw, text)

    def _call_with_retry(self, text: str) -> dict:
        """POST to ZeroGPT with exponential backoff on transient failures."""
        last_exc: Exception | None = None
        payload = {"input_text": text}

        for retry in range(config.MAX_RETRIES):
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.post(
                        self._api_url,
                        json=payload,
                        headers=self._headers,
                    )

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

                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as exc:
                last_exc = exc
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
                raise  # 4xx (other than 429) are not retryable

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

    def _parse_response(self, raw: dict, original_text: str) -> DetectorResult:
        """Extract a DetectorResult from ZeroGPT's raw JSON response.

        ZeroGPT returns:
        - fakePercentage: 0-100 float (AI probability)
        - h: list of sentence strings detected as AI-generated
        """
        fake_pct = raw.get("fakePercentage", 0.0)
        overall_ai_prob = fake_pct / 100.0

        ai_sentences: list[str] = raw.get("h", [])
        ai_sentences_set = set(ai_sentences)

        # Split original text into sentences for per-sentence scoring
        all_sentences = self._split_sentences(original_text)

        sentences: list[SentenceScore] = []
        for sent in all_sentences:
            is_flagged = sent in ai_sentences_set
            sentences.append(
                SentenceScore(
                    sentence=sent,
                    generated_prob=1.0 if is_flagged else 0.0,
                    perplexity=0.0,  # ZeroGPT doesn't provide perplexity
                )
            )

        flagged = len(ai_sentences)
        flagged_pct = (flagged / len(all_sentences) * 100) if all_sentences else 0.0

        return DetectorResult(
            overall_ai_prob=overall_ai_prob,
            burstiness=None,
            per_sentence_scores=sentences,
            flagged_sentence_pct=flagged_pct,
            raw_response=raw,
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Naive sentence splitting on common terminators."""
        import re
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in raw if s]


class GPTZeroClient(DetectorClient):
    """GPTZero API v2 implementation."""

    SENTENCE_FLAG_THRESHOLD = 0.5  # Sentences with generated_prob above this count as flagged

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "",
    ):
        key = api_key or config.GPTZERO_API_KEY
        if not key:
            raise ValueError("GPTZERO_API_KEY is not set. Add it to .env or pass it explicitly.")
        self._api_url = api_url or config.GPTZERO_API_URL
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": key,
        }

    @property
    def name(self) -> str:
        return "gptzero"

    def check(self, text: str) -> DetectorResult:
        """Send text to GPTZero and parse the response into a DetectorResult.

        Retries on transient errors (429, 5xx) with exponential backoff.
        """
        raw = self._call_with_retry(text)
        return self._parse_response(raw)

    def _call_with_retry(self, text: str) -> dict:
        """POST to GPTZero with exponential backoff on transient failures."""
        last_exc: Exception | None = None
        payload = {"document": text}

        for retry in range(config.MAX_RETRIES):
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.post(
                        self._api_url,
                        json=payload,
                        headers=self._headers,
                    )

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

                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as exc:
                last_exc = exc
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
                raise  # 4xx (other than 429) are not retryable

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

    def _parse_response(self, raw: dict) -> DetectorResult:
        """Extract a DetectorResult from GPTZero's raw JSON response."""
        doc = raw["documents"][0]

        sentences: list[SentenceScore] = []
        for s in doc.get("sentences", []):
            sentences.append(
                SentenceScore(
                    sentence=s.get("sentence", ""),
                    generated_prob=s.get("generated_prob", 0.0),
                    perplexity=s.get("perplexity", 0.0),
                )
            )

        flagged = sum(1 for s in sentences if s.generated_prob > self.SENTENCE_FLAG_THRESHOLD)
        flagged_pct = (flagged / len(sentences) * 100) if sentences else 0.0

        return DetectorResult(
            overall_ai_prob=doc.get("completely_generated_prob", 0.0),
            burstiness=doc.get("overall_burstiness", None),
            per_sentence_scores=sentences,
            flagged_sentence_pct=flagged_pct,
            raw_response=raw,
        )
