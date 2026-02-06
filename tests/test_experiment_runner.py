"""Tests for ExperimentRunner — fully mocked, no API calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import config
from src.detector_client import DetectorResult, SentenceScore
from src.gemini_client import GenerationParams, GenerationResult
from src.experiment_runner import ExperimentRunner, RunRecord


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _fake_gen_result(text: str = "word " * 500) -> GenerationResult:
    return GenerationResult(
        text=text.strip(),
        word_count=len(text.strip().split()),
        model="gemini-2.0-flash",
        params=GenerationParams(temperature=0.7, top_p=0.9, top_k=40),
        latency_ms=150,
        prompt_tokens=50,
        output_tokens=400,
        attempts=1,
    )


def _fake_det_result(ai_prob: float = 0.85) -> DetectorResult:
    return DetectorResult(
        overall_ai_prob=ai_prob,
        burstiness=40.0,
        per_sentence_scores=[
            SentenceScore(sentence="Test.", generated_prob=ai_prob, perplexity=20.0),
        ],
        flagged_sentence_pct=100.0 if ai_prob > 0.5 else 0.0,
        raw_response={"documents": [{"completely_generated_prob": ai_prob}]},
    )


@pytest.fixture
def mock_gemini():
    g = MagicMock()
    g.generate.return_value = _fake_gen_result()
    return g


@pytest.fixture
def mock_detector():
    d = MagicMock()
    d.name = "mock_detector"
    d.check.return_value = _fake_det_result(0.85)
    return d


@pytest.fixture
def registry():
    from src.prompt_registry import PromptRegistry
    return PromptRegistry()


@pytest.fixture
def runner(mock_gemini, mock_detector, registry, tmp_path):
    return ExperimentRunner(
        gemini=mock_gemini,
        detector=mock_detector,
        registry=registry,
        output_path=tmp_path / "results.jsonl",
        rate_limit_delay=0,  # No sleeping in tests
    )


# ── JSONL logging ──────────────────────────────────────────────────────────────

def test_records_written_to_jsonl(runner, tmp_path):
    """Each run should append one JSON line to the output file."""
    runner.run_gen_params_sweep(topics=["Write a test essay."], n_runs=1)

    output = tmp_path / "results.jsonl"
    assert output.exists()

    lines = output.read_text().strip().split("\n")
    # 4 gen_params variants × 1 topic × 1 run = 4 lines
    assert len(lines) == 4

    record = json.loads(lines[0])
    assert record["phase"] == "gen_params_sweep"
    assert record["dimension"] == "gen_params"
    assert "timestamp" in record
    assert "essay_text" in record


def test_jsonl_is_valid_json_per_line(runner, tmp_path):
    """Every line in the JSONL must be independently parseable."""
    runner.run_ablation("persona", topics=["Write a test essay."], n_runs=1)

    output = tmp_path / "results.jsonl"
    for line in output.read_text().strip().split("\n"):
        parsed = json.loads(line)
        assert isinstance(parsed, dict)


# ── Gen-params sweep ──────────────────────────────────────────────────────────

def test_gen_params_sweep_runs_all_variants(runner, mock_gemini, mock_detector):
    """Should run every gen-params variant × topics × n_runs."""
    records = runner.run_gen_params_sweep(topics=["Topic A", "Topic B"], n_runs=2)
    # 4 variants × 2 topics × 2 runs = 16
    assert len(records) == 16
    assert mock_gemini.generate.call_count == 16
    assert mock_detector.check.call_count == 16


def test_gen_params_sweep_passes_temperature(runner, mock_gemini):
    """The temperature/top_p from each variant should be passed to generate()."""
    runner.run_gen_params_sweep(topics=["Topic."], n_runs=1)

    temps = [call.kwargs["temperature"] for call in mock_gemini.generate.call_args_list]
    assert 0.7 in temps  # P6a baseline
    assert 1.0 in temps  # P6b
    assert 1.3 in temps  # P6c
    assert 1.5 in temps  # P6d


# ── Ablation ──────────────────────────────────────────────────────────────────

def test_ablation_runs_all_variants_in_dimension(runner, mock_gemini):
    """Should test every variant in the given dimension."""
    records = runner.run_ablation("texture", topics=["Topic."], n_runs=1)
    # texture has 5 variants (P3a-P3e)
    assert len(records) == 5

    ids = [r.variant_id for r in records]
    assert ids == ["P3a", "P3b", "P3c", "P3d", "P3e"]


def test_ablation_uses_provided_temperature(runner, mock_gemini):
    """Ablation should use the best temperature/top_p from the sweep."""
    runner.run_ablation("persona", topics=["T."], n_runs=1, temperature=1.3, top_p=0.98)

    for call in mock_gemini.generate.call_args_list:
        assert call.kwargs["temperature"] == 1.3
        assert call.kwargs["top_p"] == 0.98


def test_run_all_ablations(runner):
    """Should run ablations for all 5 prompt dimensions."""
    records = runner.run_all_ablations(topics=["Topic."], n_runs=1)
    dims = set(r.dimension for r in records)
    assert dims == {"persona", "structure", "texture", "content", "meta"}


# ── Composite prompts ─────────────────────────────────────────────────────────

def test_composites_skipped_when_empty(runner):
    """With no composites defined, should return empty and not crash."""
    records = runner.run_composites()
    assert records == []


# ── Human baselines ───────────────────────────────────────────────────────────

def test_human_baselines_reads_txt_files(runner, mock_detector, tmp_path):
    """Should read .txt files from the baselines dir and detect each one."""
    baselines = tmp_path / "baselines"
    baselines.mkdir()
    (baselines / "social_media.txt").write_text("This is a real student essay about social media.")
    (baselines / "uniforms.txt").write_text("School uniforms are a complex topic.")

    mock_detector.check.return_value = _fake_det_result(0.08)

    records = runner.run_human_baselines(baselines_dir=baselines)
    assert len(records) == 2
    assert all(r.phase == "human_baseline" for r in records)
    assert all(r.model is None for r in records)
    assert mock_detector.check.call_count == 2


def test_human_baselines_empty_dir(runner, tmp_path):
    """If no .txt files exist, should return empty list gracefully."""
    empty = tmp_path / "empty"
    empty.mkdir()
    records = runner.run_human_baselines(baselines_dir=empty)
    assert records == []


# ── RunRecord fields ──────────────────────────────────────────────────────────

def test_record_passes_threshold_field(runner, mock_detector):
    """passes_threshold should be True when AI prob < DETECTION_PASS_THRESHOLD."""
    mock_detector.check.return_value = _fake_det_result(0.10)
    records = runner.run_gen_params_sweep(topics=["T."], n_runs=1)
    assert all(r.passes_threshold for r in records)

    mock_detector.check.return_value = _fake_det_result(0.90)
    records = runner.run_ablation("persona", topics=["T."], n_runs=1)
    assert not any(r.passes_threshold for r in records)


def test_record_contains_prompts(runner):
    """Each record should log the exact system/user prompts used."""
    records = runner.run_ablation("persona", topics=["Write about testing."], n_runs=1)

    # P1a baseline — no system prompt
    baseline = [r for r in records if r.variant_id == "P1a"][0]
    assert baseline.system_prompt == ""
    assert "testing" in baseline.user_prompt

    # P1b — teen student persona
    teen = [r for r in records if r.variant_id == "P1b"][0]
    assert "Alex" in teen.system_prompt
