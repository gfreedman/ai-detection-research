"""Tests for DetectorClient — uses mocked HTTP responses, no API key needed."""

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
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

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
    """Build a fake httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "https://api.gptzero.me/v2/predict/text"),
    )
    return resp


# ── Abstract base class tests ─────────────────────────────────────────────────

def test_detector_client_is_abstract():
    """DetectorClient can't be instantiated directly."""
    with pytest.raises(TypeError):
        DetectorClient()


# ── GPTZeroClient parsing tests ────────────────────────────────────────────────

@patch("src.detector_client.httpx.Client")
def test_check_parses_ai_response(mock_httpx_cls):
    """Should correctly parse a high-AI-probability response."""
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


# ── Retry / error handling tests ───────────────────────────────────────────────

@patch("src.detector_client.time.sleep")
@patch("src.detector_client.httpx.Client")
def test_retries_on_429(mock_httpx_cls, mock_sleep):
    """Should retry with backoff on rate limit (429) then succeed."""
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
    """Should retry on server errors (5xx)."""
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
    """Should raise RuntimeError after exhausting retries."""
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
    """Non-retryable client errors (401) should raise immediately."""
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

    # Should NOT retry on 401
    assert mock_client.post.call_count == 1


def test_missing_api_key_raises():
    """Should raise ValueError if no API key is provided."""
    with patch.dict("os.environ", {"GPTZERO_API_KEY": ""}, clear=False):
        with pytest.raises(ValueError, match="GPTZERO_API_KEY"):
            GPTZeroClient()


def test_gptzero_name():
    """The name property should return 'gptzero'."""
    client = GPTZeroClient(api_key="test-key")
    assert client.name == "gptzero"
