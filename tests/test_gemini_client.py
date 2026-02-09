"""
Tests for GeminiClient -- uses mocks so no API key is needed.

Covers:

- Word-count acceptance on the first attempt.
- Re-generation when word count is outside the target range.
- Fallback to the closest-to-target result when all attempts miss.
- System prompt and custom parameter pass-through.
- Exponential-backoff retry on transient API errors.
- Failure after ``MAX_RETRIES`` consecutive errors.
- Missing API key validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src import config
from src.gemini_client import GeminiClient, GenerationParams, GenerationResult


# ===========================================================================
# Helpers
# ===========================================================================


def _fake_response(text: str, prompt_tokens: int = 50, output_tokens: int = 400):
    """
    Build a mock Gemini API response object.

    @param text:          The generated text body.
    @param prompt_tokens: Simulated prompt token count.
    @param output_tokens: Simulated output token count.
    @returns:             A ``MagicMock`` shaped like a GenAI response.
    """

    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = output_tokens

    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = usage

    return resp


def _words(n: int) -> str:
    """
    Return a string with exactly *n* whitespace-delimited words.

    Useful for testing word-count validation without needing real prose.

    @param n: Desired word count.
    @returns: A string like ``"word0 word1 word2 ..."``.
    """

    return " ".join(f"word{i}" for i in range(n))


# ===========================================================================
# Word-count validation tests
# ===========================================================================


@patch("src.gemini_client.genai.Client")
def test_generate_returns_result_in_word_range(mock_client_cls):
    """A response within [WORD_COUNT_MIN, WORD_COUNT_MAX] should be accepted on the first attempt."""

    text = _words(500)
    mock_client_cls.return_value.models.generate_content.return_value = _fake_response(text)

    client = GeminiClient(api_key="test-key")
    result = client.generate(user_prompt="Write an essay.")

    assert isinstance(result, GenerationResult)
    assert result.word_count == 500
    assert result.attempts == 1
    assert result.model == config.GEMINI_MODEL
    assert result.params == GenerationParams(
        temperature=config.DEFAULT_TEMPERATURE,
        top_p=config.DEFAULT_TOP_P,
        top_k=config.DEFAULT_TOP_K,
    )
    assert result.prompt_tokens == 50
    assert result.output_tokens == 400
    assert result.latency_ms >= 0


@patch("src.gemini_client.genai.Client")
def test_generate_retries_on_bad_word_count(mock_client_cls):
    """If the first response is too short, it should regenerate and accept a valid one."""

    short_text = _words(100)
    good_text = _words(480)

    mock_gen = mock_client_cls.return_value.models.generate_content
    mock_gen.side_effect = [_fake_response(short_text), _fake_response(good_text)]

    client = GeminiClient(api_key="test-key")
    result = client.generate(user_prompt="Write an essay.")

    assert result.word_count == 480
    assert result.attempts == 2
    assert mock_gen.call_count == 2


@patch("src.gemini_client.genai.Client")
def test_generate_returns_best_result_after_all_attempts_miss(mock_client_cls):
    """When all attempts miss the word-count range, return the closest to target (500)."""

    texts = [_words(300), _words(420), _words(600)]
    mock_gen = mock_client_cls.return_value.models.generate_content
    mock_gen.side_effect = [_fake_response(t) for t in texts]

    client = GeminiClient(api_key="test-key")
    result = client.generate(user_prompt="Write an essay.")

    # 420 words is closest to the 500-word target.
    assert result.word_count == 420
    assert result.attempts == 2


# ===========================================================================
# Parameter pass-through tests
# ===========================================================================


@patch("src.gemini_client.genai.Client")
def test_generate_passes_system_prompt(mock_client_cls):
    """System prompt should be set in the generation config sent to the API."""

    mock_gen = mock_client_cls.return_value.models.generate_content
    mock_gen.return_value = _fake_response(_words(500))

    client = GeminiClient(api_key="test-key")
    client.generate(user_prompt="Write an essay.", system_prompt="You are a student.")

    call_kwargs = mock_gen.call_args
    gen_config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")

    assert gen_config.system_instruction == "You are a student."


@patch("src.gemini_client.genai.Client")
def test_generate_passes_custom_params(mock_client_cls):
    """Custom temperature/top_p/top_k should be captured in the result's params."""

    mock_gen = mock_client_cls.return_value.models.generate_content
    mock_gen.return_value = _fake_response(_words(500))

    client = GeminiClient(api_key="test-key")
    result = client.generate(
        user_prompt="Write an essay.",
        temperature=1.3,
        top_p=0.98,
        top_k=50,
    )

    assert result.params == GenerationParams(temperature=1.3, top_p=0.98, top_k=50)


# ===========================================================================
# Retry / error handling tests
# ===========================================================================


@patch("src.gemini_client.time.sleep")  # Don't actually sleep in tests
@patch("src.gemini_client.genai.Client")
def test_api_retry_on_transient_error(mock_client_cls, mock_sleep):
    """Transient API errors should trigger exponential-backoff retries."""

    mock_gen = mock_client_cls.return_value.models.generate_content
    mock_gen.side_effect = [
        ConnectionError("network blip"),
        _fake_response(_words(500)),
    ]

    client = GeminiClient(api_key="test-key")
    result = client.generate(user_prompt="Write an essay.")

    assert result.word_count == 500
    mock_sleep.assert_called_once()  # One retry delay


@patch("src.gemini_client.time.sleep")
@patch("src.gemini_client.genai.Client")
def test_api_raises_after_max_retries(mock_client_cls, mock_sleep):
    """After MAX_RETRIES consecutive failures, should raise RuntimeError."""

    mock_gen = mock_client_cls.return_value.models.generate_content
    mock_gen.side_effect = ConnectionError("persistent failure")

    client = GeminiClient(api_key="test-key")

    with pytest.raises(RuntimeError, match="failed after"):
        client.generate(user_prompt="Write an essay.")

    assert mock_gen.call_count == config.MAX_RETRIES


# ===========================================================================
# Initialization tests
# ===========================================================================


def test_missing_api_key_raises():
    """Should raise ValueError at init time if no API key is provided."""

    with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False), \
         patch("src.config.GEMINI_API_KEY", ""):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiClient()
