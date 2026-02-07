"""
Tests for ExperimentRunner -- fully mocked, no API calls.

Covers:

- JSONL logging (one JSON line per run, valid JSON per line).
- Gen-params sweep (correct variant count, temperature pass-through).
- Per-dimension ablation (correct variant count, temperature override).
- All-dimensions ablation convenience method.
- Composite prompt handling when none are defined.
- Human baseline reading from ``.txt`` files.
- RunRecord field correctness (passes_threshold, prompt logging).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import config
from src.detector_client import DetectorResult, SentenceScore
from src.gemini_client import GenerationParams, GenerationResult
from src.experiment_runner import ExperimentRunner, RunRecord


# ===========================================================================
# Fixtures
# ===========================================================================


def _fake_gen_result(text: str = "word " * 500) -> GenerationResult:
    """
    Build a mock :class:`GenerationResult` for testing.

    @param text: The essay text (defaults to 500 words).
    @returns:    A :class:`GenerationResult` with plausible metadata.
    """

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
    """
    Build a mock :class:`DetectorResult` for testing.

    @param ai_prob: The overall AI probability (0-1).
    @returns:       A :class:`DetectorResult` with one sentence.
    """

    return DetectorResult(
        overall_ai_prob=ai_prob,
        burstiness=40.0,
        per_sentence_scores=[
            SentenceScore(sentence="Test.", generated_prob=ai_prob, perplexity=20.0),
        ],
        flagged_sentence_pct=100.0 if ai_prob > 0.5 else 0.0,
        raw_response={"documents": [{"completely_generated_prob": ai_prob}]},
    )


def _make_run_record(
    variant_id="P6a",
    temperature=0.7,
    top_p=0.9,
    overall_ai_prob=0.85,
) -> RunRecord:
    """Build a minimal RunRecord for testing best_params_from_sweep."""

    return RunRecord(
        timestamp="2025-01-01T00:00:00+00:00",
        phase="gen_params_sweep",
        dimension="gen_params",
        variant_id=variant_id,
        variant_label="test",
        topic="test topic",
        run_index=0,
        model="gemini-2.0-flash",
        temperature=temperature,
        top_p=top_p,
        top_k=40,
        word_count=500,
        generation_latency_ms=150,
        generation_attempts=1,
        essay_text="test",
        detector="mock",
        overall_ai_prob=overall_ai_prob,
        burstiness=40.0,
        flagged_sentence_pct=50.0,
        passes_threshold=overall_ai_prob < config.DETECTION_PASS_THRESHOLD,
        system_prompt="",
        user_prompt="test",
    )


@pytest.fixture
def mock_gemini():
    """A mock GeminiClient that always returns a 500-word essay."""

    g = MagicMock()
    g.generate.return_value = _fake_gen_result()

    return g


@pytest.fixture
def mock_detector():
    """A mock DetectorClient that always returns 0.85 AI probability."""

    d = MagicMock()
    d.name = "mock_detector"
    d.check.return_value = _fake_det_result(0.85)

    return d


@pytest.fixture
def registry():
    """Load the real PromptRegistry for test use."""

    from src.prompt_registry import PromptRegistry
    return PromptRegistry()


@pytest.fixture
def runner(mock_gemini, mock_detector, registry, tmp_path):
    """
    Build an ExperimentRunner wired to mocks, writing to a temp JSONL file.

    Rate-limit delay is set to 0 so tests run instantly.
    """

    return ExperimentRunner(
        gemini=mock_gemini,
        detector=mock_detector,
        registry=registry,
        output_path=tmp_path / "results.jsonl",
        rate_limit_delay=0,  # No sleeping in tests
    )


# ===========================================================================
# JSONL logging
# ===========================================================================


def test_records_written_to_jsonl(runner, tmp_path):
    """Each run should append one JSON line to the output file."""

    runner.run_gen_params_sweep(topics=["Write a test essay."], n_runs=1)

    output = tmp_path / "results.jsonl"
    assert output.exists()

    lines = output.read_text().strip().split("\n")

    # 4 gen_params variants x 1 topic x 1 run = 4 lines
    assert len(lines) == 4

    record = json.loads(lines[0])
    assert record["phase"] == "gen_params_sweep"
    assert record["dimension"] == "gen_params"
    assert "timestamp" in record
    assert "essay_text" in record


def test_jsonl_is_valid_json_per_line(runner, tmp_path):
    """Every line in the JSONL must be independently parseable as JSON."""

    runner.run_ablation("persona", topics=["Write a test essay."], n_runs=1)

    output = tmp_path / "results.jsonl"

    for line in output.read_text().strip().split("\n"):
        parsed = json.loads(line)
        assert isinstance(parsed, dict)


# ===========================================================================
# Gen-params sweep (Phase 2)
# ===========================================================================


def test_gen_params_sweep_runs_all_variants(runner, mock_gemini, mock_detector):
    """Should run every gen-params variant x topics x n_runs."""

    records = runner.run_gen_params_sweep(topics=["Topic A", "Topic B"], n_runs=2)

    # 4 variants x 2 topics x 2 runs = 16
    assert len(records) == 16
    assert mock_gemini.generate.call_count == 16
    assert mock_detector.check.call_count == 16


def test_gen_params_sweep_passes_temperature(runner, mock_gemini):
    """The temperature/top_p from each Tier 6 variant should be passed to generate()."""

    runner.run_gen_params_sweep(topics=["Topic."], n_runs=1)

    temps = [call.kwargs["temperature"] for call in mock_gemini.generate.call_args_list]

    # All four Tier 6 temperatures should appear.
    assert 0.7 in temps   # P6a baseline
    assert 1.0 in temps   # P6b
    assert 1.3 in temps   # P6c
    assert 1.5 in temps   # P6d


# ===========================================================================
# Ablation (Phase 3)
# ===========================================================================


def test_ablation_runs_all_variants_in_dimension(runner, mock_gemini):
    """Should test every variant in the given dimension."""

    records = runner.run_ablation("texture", topics=["Topic."], n_runs=1)

    # Texture has 5 variants (P3a-P3e).
    assert len(records) == 5

    ids = [r.variant_id for r in records]
    assert ids == ["P3a", "P3b", "P3c", "P3d", "P3e"]


def test_ablation_uses_provided_temperature(runner, mock_gemini):
    """Ablation should use the specified temperature/top_p from the sweep results."""

    runner.run_ablation("persona", topics=["T."], n_runs=1, temperature=1.3, top_p=0.98)

    for call in mock_gemini.generate.call_args_list:
        assert call.kwargs["temperature"] == 1.3
        assert call.kwargs["top_p"] == 0.98


def test_run_all_ablations(runner):
    """Should run ablations for all 5 prompt dimensions (Tiers 1-5)."""

    records = runner.run_all_ablations(topics=["Topic."], n_runs=1)
    dims = set(r.dimension for r in records)

    assert dims == {"persona", "structure", "texture", "content", "meta"}


# ===========================================================================
# best_params_from_sweep
# ===========================================================================


def test_best_params_from_sweep_picks_lowest_mean():
    """Should return the temperature/top_p of the variant with lowest mean AI prob."""

    records = [
        # P6a: mean AI prob = 0.85
        _make_run_record(variant_id="P6a", temperature=0.7, top_p=0.9, overall_ai_prob=0.80),
        _make_run_record(variant_id="P6a", temperature=0.7, top_p=0.9, overall_ai_prob=0.90),
        # P6b: mean AI prob = 0.55 -- winner
        _make_run_record(variant_id="P6b", temperature=1.0, top_p=0.95, overall_ai_prob=0.50),
        _make_run_record(variant_id="P6b", temperature=1.0, top_p=0.95, overall_ai_prob=0.60),
    ]

    temp, top_p = ExperimentRunner.best_params_from_sweep(records)
    assert temp == 1.0
    assert top_p == 0.95


def test_best_params_from_sweep_empty_returns_defaults():
    """Empty sweep results should fall back to config defaults."""

    temp, top_p = ExperimentRunner.best_params_from_sweep([])
    assert temp == config.DEFAULT_TEMPERATURE
    assert top_p == config.DEFAULT_TOP_P


# ===========================================================================
# Composite prompts (Phase 4)
# ===========================================================================


def test_composites_skipped_when_empty(runner):
    """With no composites defined, should return empty list without crashing."""

    records = runner.run_composites()
    assert records == []


# ===========================================================================
# Human baselines (Phase 1)
# ===========================================================================


def test_human_baselines_reads_txt_files(runner, mock_detector, tmp_path):
    """Should read .txt files from the baselines directory and detect each one."""

    # -- Create two fake human essays in a temp baselines directory ----------
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
    """If no .txt files exist in the directory, should return an empty list."""

    empty = tmp_path / "empty"
    empty.mkdir()

    records = runner.run_human_baselines(baselines_dir=empty)
    assert records == []


# ===========================================================================
# RunRecord field validation
# ===========================================================================


def test_record_passes_threshold_field(runner, mock_detector):
    """
    ``passes_threshold`` should be True when AI prob < DETECTION_PASS_THRESHOLD
    and False otherwise.
    """

    # AI prob 0.10 < threshold 0.15 --> passes
    mock_detector.check.return_value = _fake_det_result(0.10)
    records = runner.run_gen_params_sweep(topics=["T."], n_runs=1)
    assert all(r.passes_threshold for r in records)

    # AI prob 0.90 >= threshold 0.15 --> fails
    mock_detector.check.return_value = _fake_det_result(0.90)
    records = runner.run_ablation("persona", topics=["T."], n_runs=1)
    assert not any(r.passes_threshold for r in records)


def test_record_contains_prompts(runner):
    """Each record should log the exact system and user prompts used."""

    records = runner.run_ablation("persona", topics=["Write about testing."], n_runs=1)

    # P1a baseline -- no system prompt.
    baseline = [r for r in records if r.variant_id == "P1a"][0]
    assert baseline.system_prompt == ""
    assert "testing" in baseline.user_prompt

    # P1b -- teen student persona.
    teen = [r for r in records if r.variant_id == "P1b"][0]
    assert "Alex" in teen.system_prompt
