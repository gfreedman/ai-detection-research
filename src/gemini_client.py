"""
Thin Gemini wrapper -- generation with retry logic, word-count validation,
and metadata capture.

This module provides :class:`GeminiClient`, which wraps the Google GenAI SDK
to generate essays for the experiment.  It enforces the target word-count range
(re-generating when necessary) and retries transient API failures with
exponential backoff.

Data Classes
------------
- :class:`GenerationParams` -- immutable snapshot of temperature / top_p / top_k.
- :class:`GenerationResult` -- the generated text plus all metadata needed for
  reproducible logging (model, params, latency, token counts, attempt number).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationParams:
    """
    Immutable snapshot of the parameters used for a single generation call.

    @param temperature: Sampling temperature -- higher values increase randomness.
    @param top_p:       Nucleus-sampling threshold.
    @param top_k:       Top-k sampling cutoff.
    """

    temperature: float
    top_p: float
    top_k: int


@dataclass(frozen=True)
class GenerationResult:
    """
    Output of a single essay generation, bundled with metadata for logging.

    @param text:          The generated essay text.
    @param word_count:    Number of whitespace-delimited words in *text*.
    @param model:         Model identifier used for the generation.
    @param params:        The :class:`GenerationParams` that produced this result.
    @param latency_ms:    Wall-clock latency of the successful API call (ms).
    @param prompt_tokens: Prompt token count reported by the API (may be None).
    @param output_tokens: Output token count reported by the API (may be None).
    @param attempts:      How many generation attempts were needed before the
                          word count was acceptable (1 = first try was good).
    """

    text: str
    word_count: int
    model: str
    params: GenerationParams
    latency_ms: int
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    attempts: int = 1


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GeminiClient:
    """
    Generates essays via the Google GenAI SDK with retry and word-count
    enforcement.

    Usage::

        client = GeminiClient()                      # reads key from .env
        result = client.generate("Write an essay.")   # -> GenerationResult
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
    ):
        """
        Initialise the Gemini client.

        @param api_key: Gemini API key.  Falls back to ``config.GEMINI_API_KEY``
                        (loaded from the ``GEMINI_API_KEY`` env var) when empty.
        @param model:   Model identifier.  Falls back to ``config.GEMINI_MODEL``.
        @raises ValueError: If no API key is available.
        """

        key = api_key or config.GEMINI_API_KEY

        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to .env or pass it explicitly."
            )

        self._client = genai.Client(api_key=key)
        self._model = model or config.GEMINI_MODEL

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = config.DEFAULT_TEMPERATURE,
        top_p: float = config.DEFAULT_TOP_P,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> GenerationResult:
        """
        Generate an essay and validate its word count.

        The method retries the underlying API call on transient errors (with
        exponential backoff).  If the returned text falls outside the target
        word-count range ``[WORD_COUNT_MIN, WORD_COUNT_MAX]``, it regenerates
        up to ``MAX_REGENERATION_ATTEMPTS`` times, ultimately returning the
        attempt whose word count is closest to ``TARGET_WORD_COUNT``.

        @param user_prompt:   The essay prompt sent as user content.
        @param system_prompt: Optional system instruction (persona, constraints).
        @param temperature:   Sampling temperature for this call.
        @param top_p:         Nucleus-sampling threshold for this call.
        @param top_k:         Top-k cutoff for this call.
        @returns: A :class:`GenerationResult` with text, word count, and metadata.
        """

        params = GenerationParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # -- Build the SDK generation config --------------------------------
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
        )

        if system_prompt:
            gen_config.system_instruction = system_prompt

        # -- Attempt loop: regenerate until word count is in range ----------
        best_result: GenerationResult | None = None

        for attempt in range(1, config.MAX_REGENERATION_ATTEMPTS + 1):

            text, latency_ms, prompt_tokens, output_tokens = self._call_with_retry(
                user_prompt, gen_config
            )

            wc = len(text.split())

            result = GenerationResult(
                text=text,
                word_count=wc,
                model=self._model,
                params=params,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                attempts=attempt,
            )

            # Accept immediately if the word count is within target range.
            if config.WORD_COUNT_MIN <= wc <= config.WORD_COUNT_MAX:
                return result

            # Otherwise track the closest-to-target result as a fallback.
            if (
                best_result is None
                or abs(wc - config.TARGET_WORD_COUNT)
                < abs(best_result.word_count - config.TARGET_WORD_COUNT)
            ):
                best_result = result

            logger.warning(
                "Word count %d outside target range [%d, %d] (attempt %d/%d)",
                wc,
                config.WORD_COUNT_MIN,
                config.WORD_COUNT_MAX,
                attempt,
                config.MAX_REGENERATION_ATTEMPTS,
            )

        # All attempts missed -- return the best we got.
        logger.warning(
            "All %d attempts missed word count target. Using best result (%d words).",
            config.MAX_REGENERATION_ATTEMPTS,
            best_result.word_count,
        )
        return best_result  # type: ignore[return-value]

    # ── Internals ──────────────────────────────────────────────────────────

    def _call_with_retry(
        self,
        user_prompt: str,
        gen_config: types.GenerateContentConfig,
    ) -> tuple[str, int, int | None, int | None]:
        """
        Call the Gemini API with exponential backoff on transient failures.

        @param user_prompt: The user content to send to the model.
        @param gen_config:  Fully configured ``GenerateContentConfig``.
        @returns: A tuple of ``(text, latency_ms, prompt_tokens, output_tokens)``.
        @raises RuntimeError: After ``MAX_RETRIES`` consecutive failures.
        """

        last_exc: Exception | None = None

        for retry in range(config.MAX_RETRIES):

            try:
                # -- Time the actual API round-trip --------------------------
                t0 = time.perf_counter()

                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user_prompt,
                    config=gen_config,
                )

                latency_ms = int((time.perf_counter() - t0) * 1000)

                text = response.text or ""

                # -- Extract optional token-usage metadata -------------------
                prompt_tokens = None
                output_tokens = None

                if response.usage_metadata:
                    prompt_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count

                return text, latency_ms, prompt_tokens, output_tokens

            except Exception as exc:
                last_exc = exc

                # Use a longer base delay for rate-limit (429) errors since
                # the free-tier daily quota recovers slowly.
                is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                base = 30.0 if is_rate_limit else config.RETRY_BASE_DELAY_S
                delay = base * (2**retry)

                # Respect Gemini's suggested retry delay if present.
                match = re.search(r"retry in ([\d.]+)s", str(exc))
                if match:
                    delay = max(delay, float(match.group(1)) + 1)

                logger.warning(
                    "Gemini API error (retry %d/%d, backoff %.1fs): %s",
                    retry + 1,
                    config.MAX_RETRIES,
                    delay,
                    exc,
                )

                time.sleep(delay)

        raise RuntimeError(
            f"Gemini API failed after {config.MAX_RETRIES} retries"
        ) from last_exc
