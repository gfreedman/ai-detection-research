"""
Tests for DetectorClient -- uses mocked HTTP responses, no API key needed.

Covers both :class:`GPTZeroClient` and :class:`ZeroGPTClient`:

- Response parsing (AI-heavy text, human text, per-sentence scores).
- Retry behaviour on transient errors (429, 5xx, connection failures).
- Immediate failure on non-retryable errors (401).
- Missing API key validation.
- The ``name`` property contract.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import httpx
import pytest

from src import config
from src.detector_client import (
    DetectorClient,
    DetectorResult,
    GPTZeroClient,
    SentenceScore,
    ZeroGPTClient,
)


# ===========================================================================
# GPTZero fixtures
# ===========================================================================


# A typical GPTZero response for a mostly-AI essay:
# - completely_generated_prob = 0.87
# - 4 sentences, 2 above the 0.5 flag threshold
SAMPLE_GPTZERO_RESPONSE = {
    "documents": [
        {
            "completely_generated_prob": 0.87,
            "average_generated_prob": 0.82,
            "overall_burstiness": 34.5,
            "document_classification": "AI_ONLY",
            "class_probabilities": {"ai": 0.87, "mixed": 0.10, "human": 0.03},
            "confidence_category": "high",
            "sentences": [
                {"sentence": "AI is transforming education.", "generated_prob": 0.95, "perplexity": 12.3},
                {"sentence": "I think it's kinda weird though.", "generated_prob": 0.15, "perplexity": 85.7},
                {"sentence": "Studies show measurable impacts.", "generated_prob": 0.88, "perplexity": 18.1},
                {"sentence": "My teacher mentioned this once.", "generated_prob": 0.30, "perplexity": 62.4},
            ],
        }
    ]
}

# A typical GPTZero response for a genuinely human-written essay:
# - completely_generated_prob = 0.05
# - 2 sentences, both well below the flag threshold
SAMPLE_HUMAN_RESPONSE = {
    "documents": [
        {
            "completely_generated_prob": 0.05,
            "average_generated_prob": 0.08,
            "overall_burstiness": 78.2,
            "document_classification": "HUMAN_ONLY",
            "class_probabilities": {"ai": 0.05, "mixed": 0.10, "human": 0.85},
            "confidence_category": "high",
            "sentences": [
                {"sentence": "So like yesterday I was thinking.", "generated_prob": 0.05, "perplexity": 120.0},
                {"sentence": "It hit me pretty hard tbh.", "generated_prob": 0.08, "perplexity": 95.3},
            ],
        }
    ]
}


def _mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """
    Build a fake ``httpx.Response`` for GPTZero tests.

    @param json_data:   JSON body to return.
    @param status_code: HTTP status code.
    @returns:           A fully formed ``httpx.Response``.
    """

    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "https://api.gptzero.me/v2/predict/text"),
    )


# ===========================================================================
# Abstract base class tests
# ===========================================================================


def test_detector_client_is_abstract():
    """DetectorClient cannot be instantiated directly -- it is abstract."""
    with pytest.raises(TypeError):
        DetectorClient()


# ===========================================================================
# GPTZeroClient -- response parsing
# ===========================================================================


@patch("src.detector_client.httpx.Client")
def test_check_parses_ai_response(mock_httpx_cls):
    """
    Given a high-AI-probability GPTZero response, the client should:
    - Set overall_ai_prob to 0.87.
    - Report burstiness as 34.5.
    - Parse all 4 sentences.
    - Flag 2 of 4 sentences (generated_prob > 0.5) = 50%.
    """

    mock_httpx_cls.return_value.__enter__ = lambda self: self
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_httpx_cls.return_value.post.return_value = _mock_response(SAMPLE_GPTZERO_RESPONSE)

    client = GPTZeroClient(api_key="test-key")
    result = client.check("Some essay text.")

    assert isinstance(result, DetectorResult)
    assert result.overall_ai_prob == 0.87
    assert result.burstiness == 34.5
    assert len(result.per_sentence_scores) == 4

    # 2 of 4 sentences have generated_prob > 0.5
    assert result.flagged_sentence_pct == 50.0
    assert result.raw_response == SAMPLE_GPTZERO_RESPONSE


@patch("src.detector_client.httpx.Client")
def test_check_parses_human_response(mock_httpx_cls):
    """A human-written text should have low AI prob and 0% flagged sentences."""

    mock_httpx_cls.return_value.__enter__ = lambda self: self
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_httpx_cls.return_value.post.return_value = _mock_response(SAMPLE_HUMAN_RESPONSE)

    client = GPTZeroClient(api_key="test-key")
    result = client.check("A real student essay.")

    assert result.overall_ai_prob == 0.05
    assert result.burstiness == 78.2
    assert result.flagged_sentence_pct == 0.0
    assert len(result.per_sentence_scores) == 2


@patch("src.detector_client.httpx.Client")
def test_sentence_scores_populated(mock_httpx_cls):
    """Per-sentence scores should carry through sentence text, prob, and perplexity."""

    mock_httpx_cls.return_value.__enter__ = lambda self: self
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_httpx_cls.return_value.post.return_value = _mock_response(SAMPLE_GPTZERO_RESPONSE)

    client = GPTZeroClient(api_key="test-key")
    result = client.check("Text.")

    s = result.per_sentence_scores[0]

    assert isinstance(s, SentenceScore)
    assert s.sentence == "AI is transforming education."
    assert s.generated_prob == 0.95
    assert s.perplexity == 12.3


# ===========================================================================
# GPTZeroClient -- retry / error handling
# ===========================================================================


@patch("src.detector_client.time.sleep")
@patch("src.detector_client.httpx.Client")
def test_retries_on_429(mock_httpx_cls, mock_sleep):
    """Should retry with exponential backoff on rate-limit (429), then succeed."""

    rate_limited = _mock_response({}, status_code=429)
    ok = _mock_response(SAMPLE_HUMAN_RESPONSE)

    mock_client = MagicMock()
    mock_httpx_cls.return_value.__enter__ = lambda self: mock_client
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = [rate_limited, ok]

    client = GPTZeroClient(api_key="test-key")
    result = client.check("Text.")

    assert result.overall_ai_prob == 0.05
    mock_sleep.assert_called_once()


@patch("src.detector_client.time.sleep")
@patch("src.detector_client.httpx.Client")
def test_retries_on_500(mock_httpx_cls, mock_sleep):
    """Should retry with exponential backoff on server errors (5xx)."""

    error_resp = httpx.Response(
        status_code=500,
        text="Internal Server Error",
        request=httpx.Request("POST", "https://api.gptzero.me/v2/predict/text"),
    )
    ok = _mock_response(SAMPLE_HUMAN_RESPONSE)

    mock_client = MagicMock()
    mock_httpx_cls.return_value.__enter__ = lambda self: mock_client
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = [
        httpx.HTTPStatusError("500", request=error_resp.request, response=error_resp),
        ok,
    ]

    client = GPTZeroClient(api_key="test-key")
    result = client.check("Text.")

    assert result.overall_ai_prob == 0.05
    mock_sleep.assert_called_once()


@patch("src.detector_client.time.sleep")
@patch("src.detector_client.httpx.Client")
def test_raises_after_max_retries(mock_httpx_cls, mock_sleep):
    """Should raise RuntimeError after exhausting all retry attempts."""

    mock_client = MagicMock()
    mock_httpx_cls.return_value.__enter__ = lambda self: mock_client
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = httpx.ConnectError("connection refused")

    client = GPTZeroClient(api_key="test-key")

    with pytest.raises(RuntimeError, match="failed after"):
        client.check("Text.")

    assert mock_client.post.call_count == config.MAX_RETRIES


@patch("src.detector_client.httpx.Client")
def test_raises_on_401(mock_httpx_cls):
    """Non-retryable client errors (401 Unauthorized) should raise immediately."""

    error_resp = httpx.Response(
        status_code=401,
        text="Unauthorized",
        request=httpx.Request("POST", "https://api.gptzero.me/v2/predict/text"),
    )

    mock_client = MagicMock()
    mock_httpx_cls.return_value.__enter__ = lambda self: mock_client
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = httpx.HTTPStatusError(
        "401", request=error_resp.request, response=error_resp
    )

    client = GPTZeroClient(api_key="test-key")

    with pytest.raises(httpx.HTTPStatusError):
        client.check("Text.")

    # Should NOT retry on 401 -- only one attempt.
    assert mock_client.post.call_count == 1


def test_missing_api_key_raises():
    """Should raise ValueError at init time if no API key is provided."""

    with patch.dict("os.environ", {"GPTZERO_API_KEY": ""}, clear=False):
        with pytest.raises(ValueError, match="GPTZERO_API_KEY"):
            GPTZeroClient()


def test_gptzero_name():
    """The ``name`` property should return ``'gptzero'``."""

    client = GPTZeroClient(api_key="test-key")
    assert client.name == "gptzero"


# ===========================================================================
# ZeroGPT fixtures
# ===========================================================================


# A typical ZeroGPT response for a mostly-AI essay:
# - fakePercentage = 85.64  (normalised to 0.8564)
# - "h" lists the 2 sentences ZeroGPT considers AI-generated
SAMPLE_ZEROGPT_AI_RESPONSE = {
    "success": True,
    "data": {
        "is_human_written": False,
        "is_gpt_generated": True,
    },
    "fakePercentage": 85.64,
    "textWords": 78,
    "aiWords": 67,
    "h": [
        "AI is transforming education.",
        "Studies show measurable impacts.",
    ],
}

# A typical ZeroGPT response for a genuinely human-written essay:
# - fakePercentage = 4.12  (normalised to 0.0412)
# - "h" is empty (no sentences flagged)
SAMPLE_ZEROGPT_HUMAN_RESPONSE = {
    "success": True,
    "data": {
        "is_human_written": True,
        "is_gpt_generated": False,
    },
    "fakePercentage": 4.12,
    "textWords": 45,
    "aiWords": 2,
    "h": [],
}

# The input text used for ZeroGPT AI-response tests.
# Four sentences -- two of which appear in SAMPLE_ZEROGPT_AI_RESPONSE["h"].
SAMPLE_ZEROGPT_INPUT = (
    "AI is transforming education. I think it's kinda weird though. "
    "Studies show measurable impacts. My teacher mentioned this once."
)


def _mock_zerogpt_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """
    Build a fake ``httpx.Response`` for ZeroGPT tests.

    @param json_data:   JSON body to return.
    @param status_code: HTTP status code.
    @returns:           A fully formed ``httpx.Response``.
    """

    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "https://api.zerogpt.com/api/detect/detectText"),
    )


# ===========================================================================
# ZeroGPTClient -- response parsing
# ===========================================================================


@patch("src.detector_client.httpx.Client")
def test_zerogpt_check_parses_ai_response(mock_httpx_cls):
    """
    Given a high-AI-probability ZeroGPT response, the client should:
    - Normalise fakePercentage (85.64) to overall_ai_prob (0.8564).
    - Set burstiness to None (not provided by ZeroGPT).
    - Split the input into 4 sentences.
    - Flag 2 of 4 sentences (those in the "h" array) = 50%.
    """

    mock_httpx_cls.return_value.__enter__ = lambda self: self
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_httpx_cls.return_value.post.return_value = _mock_zerogpt_response(SAMPLE_ZEROGPT_AI_RESPONSE)

    client = ZeroGPTClient(api_key="test-key")
    result = client.check(SAMPLE_ZEROGPT_INPUT)

    assert isinstance(result, DetectorResult)
    assert result.overall_ai_prob == pytest.approx(0.8564)
    assert result.burstiness is None
    assert len(result.per_sentence_scores) == 4

    # 2 of 4 sentences flagged
    assert result.flagged_sentence_pct == 50.0
    assert result.raw_response == SAMPLE_ZEROGPT_AI_RESPONSE


@patch("src.detector_client.httpx.Client")
def test_zerogpt_check_parses_human_response(mock_httpx_cls):
    """A human-written text should have low AI prob and 0% flagged sentences."""

    mock_httpx_cls.return_value.__enter__ = lambda self: self
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_httpx_cls.return_value.post.return_value = _mock_zerogpt_response(SAMPLE_ZEROGPT_HUMAN_RESPONSE)

    client = ZeroGPTClient(api_key="test-key")
    result = client.check("So like yesterday I was thinking. It hit me pretty hard tbh.")

    assert result.overall_ai_prob == pytest.approx(0.0412)
    assert result.burstiness is None
    assert result.flagged_sentence_pct == 0.0
    assert len(result.per_sentence_scores) == 2


@patch("src.detector_client.httpx.Client")
def test_zerogpt_sentence_scores_populated(mock_httpx_cls):
    """
    Per-sentence scores should use binary flagging:
    - Sentences in the "h" array get generated_prob = 1.0.
    - Sentences not in "h" get generated_prob = 0.0.
    """

    mock_httpx_cls.return_value.__enter__ = lambda self: self
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_httpx_cls.return_value.post.return_value = _mock_zerogpt_response(SAMPLE_ZEROGPT_AI_RESPONSE)

    client = ZeroGPTClient(api_key="test-key")
    result = client.check(SAMPLE_ZEROGPT_INPUT)

    # First sentence IS in "h" -- should be flagged.
    assert result.per_sentence_scores[0].sentence == "AI is transforming education."
    assert result.per_sentence_scores[0].generated_prob == 1.0

    # Second sentence is NOT in "h" -- should not be flagged.
    assert result.per_sentence_scores[1].sentence == "I think it's kinda weird though."
    assert result.per_sentence_scores[1].generated_prob == 0.0


# ===========================================================================
# ZeroGPTClient -- retry / error handling
# ===========================================================================


@patch("src.detector_client.time.sleep")
@patch("src.detector_client.httpx.Client")
def test_zerogpt_retries_on_429(mock_httpx_cls, mock_sleep):
    """Should retry with exponential backoff on rate-limit (429), then succeed."""

    rate_limited = _mock_zerogpt_response({}, status_code=429)
    ok = _mock_zerogpt_response(SAMPLE_ZEROGPT_HUMAN_RESPONSE)

    mock_client = MagicMock()
    mock_httpx_cls.return_value.__enter__ = lambda self: mock_client
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = [rate_limited, ok]

    client = ZeroGPTClient(api_key="test-key")
    result = client.check("Text.")

    assert result.overall_ai_prob == pytest.approx(0.0412)
    mock_sleep.assert_called_once()


@patch("src.detector_client.time.sleep")
@patch("src.detector_client.httpx.Client")
def test_zerogpt_retries_on_500(mock_httpx_cls, mock_sleep):
    """Should retry with exponential backoff on server errors (5xx)."""

    error_resp = httpx.Response(
        status_code=500,
        text="Internal Server Error",
        request=httpx.Request("POST", "https://api.zerogpt.com/api/detect/detectText"),
    )
    ok = _mock_zerogpt_response(SAMPLE_ZEROGPT_HUMAN_RESPONSE)

    mock_client = MagicMock()
    mock_httpx_cls.return_value.__enter__ = lambda self: mock_client
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = [
        httpx.HTTPStatusError("500", request=error_resp.request, response=error_resp),
        ok,
    ]

    client = ZeroGPTClient(api_key="test-key")
    result = client.check("Text.")

    assert result.overall_ai_prob == pytest.approx(0.0412)
    mock_sleep.assert_called_once()


@patch("src.detector_client.time.sleep")
@patch("src.detector_client.httpx.Client")
def test_zerogpt_raises_after_max_retries(mock_httpx_cls, mock_sleep):
    """Should raise RuntimeError after exhausting all retry attempts."""

    mock_client = MagicMock()
    mock_httpx_cls.return_value.__enter__ = lambda self: mock_client
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = httpx.ConnectError("connection refused")

    client = ZeroGPTClient(api_key="test-key")

    with pytest.raises(RuntimeError, match="failed after"):
        client.check("Text.")

    assert mock_client.post.call_count == config.MAX_RETRIES


@patch("src.detector_client.httpx.Client")
def test_zerogpt_raises_on_401(mock_httpx_cls):
    """Non-retryable client errors (401 Unauthorized) should raise immediately."""

    error_resp = httpx.Response(
        status_code=401,
        text="Unauthorized",
        request=httpx.Request("POST", "https://api.zerogpt.com/api/detect/detectText"),
    )

    mock_client = MagicMock()
    mock_httpx_cls.return_value.__enter__ = lambda self: mock_client
    mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = httpx.HTTPStatusError(
        "401", request=error_resp.request, response=error_resp
    )

    client = ZeroGPTClient(api_key="test-key")

    with pytest.raises(httpx.HTTPStatusError):
        client.check("Text.")

    # Should NOT retry on 401 -- only one attempt.
    assert mock_client.post.call_count == 1


def test_zerogpt_missing_api_key_raises():
    """Should raise ValueError at init time if no API key is provided."""

    with patch.dict("os.environ", {"ZEROGPT_API_KEY": ""}, clear=False):
        with pytest.raises(ValueError, match="ZEROGPT_API_KEY"):
            ZeroGPTClient()


def test_zerogpt_name():
    """The ``name`` property should return ``'zerogpt'``."""

    client = ZeroGPTClient(api_key="test-key")
    assert client.name == "zerogpt"
