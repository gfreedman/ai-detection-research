"""Thin Gemini wrapper — generation with retry logic, word count validation, and metadata capture."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from src import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationParams:
    """Snapshot of the parameters used for a single generation call."""

    temperature: float
    top_p: float
    top_k: int


@dataclass(frozen=True)
class GenerationResult:
    """Output of a single essay generation, including metadata for logging."""

    text: str
    word_count: int
    model: str
    params: GenerationParams
    latency_ms: int
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    attempts: int = 1  # How many generation attempts before word count was acceptable


class GeminiClient:
    """Generates essays via the Google GenAI SDK with retry and word count enforcement."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
    ):
        key = api_key or config.GEMINI_API_KEY
        if not key:
            raise ValueError("GEMINI_API_KEY is not set. Add it to .env or pass it explicitly.")
        self._client = genai.Client(api_key=key)
        self._model = model or config.GEMINI_MODEL

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = config.DEFAULT_TEMPERATURE,
        top_p: float = config.DEFAULT_TOP_P,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> GenerationResult:
        """Generate an essay and validate its word count.

        Retries the API call on transient errors (with exponential backoff).
        If the returned text falls outside the target word count range, regenerates
        up to MAX_REGENERATION_ATTEMPTS times.

        Returns a GenerationResult with the text, word count, and generation metadata.
        """
        params = GenerationParams(temperature=temperature, top_p=top_p, top_k=top_k)

        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=1024,
        )
        if system_prompt:
            gen_config.system_instruction = system_prompt

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

            if config.WORD_COUNT_MIN <= wc <= config.WORD_COUNT_MAX:
                return result

            # Keep the closest-to-target result as fallback
            if best_result is None or abs(wc - config.TARGET_WORD_COUNT) < abs(
                best_result.word_count - config.TARGET_WORD_COUNT
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

        logger.warning(
            "All %d attempts missed word count target. Using best result (%d words).",
            config.MAX_REGENERATION_ATTEMPTS,
            best_result.word_count,
        )
        return best_result  # type: ignore[return-value]

    def _call_with_retry(
        self,
        user_prompt: str,
        gen_config: types.GenerateContentConfig,
    ) -> tuple[str, int, int | None, int | None]:
        """Call the Gemini API with exponential backoff on transient failures.

        Returns (text, latency_ms, prompt_tokens, output_tokens).
        """
        last_exc: Exception | None = None

        for retry in range(config.MAX_RETRIES):
            try:
                t0 = time.perf_counter()
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user_prompt,
                    config=gen_config,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)

                text = response.text or ""

                prompt_tokens = None
                output_tokens = None
                if response.usage_metadata:
                    prompt_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count

                return text, latency_ms, prompt_tokens, output_tokens

            except Exception as exc:
                last_exc = exc
                delay = config.RETRY_BASE_DELAY_S * (2**retry)
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
