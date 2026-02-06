"""Tests for src/analysis.py — aggregation, stats, and export helpers."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import (
    cohens_d,
    compare_all_dimensions,
    compare_to_baseline,
    export_summary_csv,
    human_baseline_summary,
    load_results,
    rank_variants,
    summarize_by_topic,
    summarize_variants,
    temperature_summary,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_record(
    phase="ablation",
    dimension="persona",
    variant_id="P1b",
    variant_label="Teen student",
    topic="social media",
    overall_ai_prob=0.85,
    burstiness=0.5,
    flagged_sentence_pct=60.0,
    passes_threshold=False,
    temperature=0.7,
    **overrides,
):
    rec = {
        "timestamp": "2025-01-01T00:00:00+00:00",
        "phase": phase,
        "dimension": dimension,
        "variant_id": variant_id,
        "variant_label": variant_label,
        "topic": topic,
        "run_index": 0,
        "model": "gemini-2.0-flash",
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "word_count": 500,
        "generation_latency_ms": 1200,
        "generation_attempts": 1,
        "essay_text": "Some essay text.",
        "detector": "gptzero",
        "overall_ai_prob": overall_ai_prob,
        "burstiness": burstiness,
        "flagged_sentence_pct": flagged_sentence_pct,
        "passes_threshold": passes_threshold,
        "system_prompt": "",
        "user_prompt": topic,
    }
    rec.update(overrides)
    return rec


def _make_df(records):
    return pd.DataFrame(records)


def _ablation_dataset():
    """Build a small ablation dataset with a baseline and two variants."""
    records = []
    # Baseline P1a — high detection
    for i in range(5):
        records.append(
            _make_record(
                variant_id="P1a",
                variant_label="No persona",
                overall_ai_prob=0.80 + i * 0.02,
                passes_threshold=False,
            )
        )
    # Variant P1b — moderate detection
    for i in range(5):
        records.append(
            _make_record(
                variant_id="P1b",
                variant_label="Teen student",
                overall_ai_prob=0.50 + i * 0.02,
                passes_threshold=False,
            )
        )
    # Variant P1c — low detection
    for i in range(5):
        records.append(
            _make_record(
                variant_id="P1c",
                variant_label="B+ student",
                overall_ai_prob=0.10 + i * 0.02,
                passes_threshold=True,
            )
        )
    return _make_df(records)


# ── load_results ─────────────────────────────────────────────────────────────


def test_load_results(tmp_path):
    jsonl = tmp_path / "results.jsonl"
    records = [_make_record(run_index=i) for i in range(3)]
    jsonl.write_text("\n".join(json.dumps(r) for r in records))

    df = load_results(jsonl)
    assert len(df) == 3
    assert "overall_ai_prob" in df.columns


def test_load_results_skips_blank_lines(tmp_path):
    jsonl = tmp_path / "results.jsonl"
    rec = _make_record()
    jsonl.write_text(json.dumps(rec) + "\n\n" + json.dumps(rec) + "\n")

    df = load_results(jsonl)
    assert len(df) == 2


# ── summarize_variants ───────────────────────────────────────────────────────


def test_summarize_variants_groups_correctly():
    df = _ablation_dataset()
    summary = summarize_variants(df)
    assert len(summary) == 3  # P1a, P1b, P1c
    assert set(summary["variant_id"]) == {"P1a", "P1b", "P1c"}
    assert all(summary["n"] == 5)


def test_summarize_variants_computes_pass_rate():
    df = _ablation_dataset()
    summary = summarize_variants(df)
    p1c_row = summary[summary["variant_id"] == "P1c"].iloc[0]
    assert p1c_row["pass_rate"] == 1.0

    p1a_row = summary[summary["variant_id"] == "P1a"].iloc[0]
    assert p1a_row["pass_rate"] == 0.0


# ── summarize_by_topic ───────────────────────────────────────────────────────


def test_summarize_by_topic():
    records = [
        _make_record(topic="topic_a", variant_id="P1a", overall_ai_prob=0.8),
        _make_record(topic="topic_a", variant_id="P1a", overall_ai_prob=0.9),
        _make_record(topic="topic_b", variant_id="P1a", overall_ai_prob=0.7),
    ]
    df = _make_df(records)
    summary = summarize_by_topic(df)
    assert len(summary) == 2
    topic_a = summary[summary["topic"] == "topic_a"].iloc[0]
    assert topic_a["n"] == 2
    assert topic_a["mean_ai_prob"] == pytest.approx(0.85)


# ── cohens_d ─────────────────────────────────────────────────────────────────


def test_cohens_d_identical_groups():
    a = pd.Series([0.5, 0.5, 0.5])
    b = pd.Series([0.5, 0.5, 0.5])
    assert cohens_d(a, b) == 0.0


def test_cohens_d_distinct_groups():
    a = pd.Series([10.0, 10.0, 10.0, 10.0])
    b = pd.Series([0.0, 0.0, 0.0, 0.0])
    # All same within groups, so pooled_sd = 0 → returns 0
    assert cohens_d(a, b) == 0.0


def test_cohens_d_positive_effect():
    a = pd.Series([0.80, 0.82, 0.84, 0.86, 0.88])  # baseline (higher)
    b = pd.Series([0.50, 0.52, 0.54, 0.56, 0.58])  # variant (lower)
    d = cohens_d(a, b)
    assert d > 0  # baseline > variant → positive d


def test_cohens_d_too_few_samples():
    a = pd.Series([0.5])
    b = pd.Series([0.3, 0.4])
    assert math.isnan(cohens_d(a, b))


# ── compare_to_baseline ─────────────────────────────────────────────────────


def test_compare_to_baseline_structure():
    df = _ablation_dataset()
    result = compare_to_baseline(df, "persona")
    assert len(result) == 2  # P1b and P1c (not baseline itself)
    assert "cohens_d" in result.columns
    assert "p_value" in result.columns
    assert all(result["baseline_id"] == "P1a")


def test_compare_to_baseline_p_values():
    df = _ablation_dataset()
    result = compare_to_baseline(df, "persona")
    # P1b and P1c differ from baseline → p should be small
    for _, row in result.iterrows():
        assert row["p_value"] < 0.05


def test_compare_to_baseline_no_baseline_raises():
    records = [_make_record(variant_id="P1b") for _ in range(5)]
    df = _make_df(records)
    with pytest.raises(ValueError, match="No baseline"):
        compare_to_baseline(df, "persona")


# ── compare_all_dimensions ───────────────────────────────────────────────────


def test_compare_all_dimensions_skips_human_and_composite():
    records = []
    # Add persona ablation data
    for i in range(5):
        records.append(_make_record(variant_id="P1a", overall_ai_prob=0.8))
        records.append(_make_record(variant_id="P1b", overall_ai_prob=0.5))
    # Add human baselines (should be skipped)
    records.append(
        _make_record(phase="human_baseline", dimension="human", variant_id="human_baseline")
    )
    df = _make_df(records)
    result = compare_all_dimensions(df)
    assert all(result["dimension"] == "persona")


# ── rank_variants ────────────────────────────────────────────────────────────


def test_rank_variants_sorted_ascending():
    df = _ablation_dataset()
    ranked = rank_variants(df)
    # P1c (low prob) should be first, P1b (medium) second
    assert ranked.iloc[0]["variant_id"] == "P1c"
    assert ranked.iloc[1]["variant_id"] == "P1b"


def test_rank_variants_excludes_baselines():
    df = _ablation_dataset()
    ranked = rank_variants(df)
    assert "P1a" not in ranked["variant_id"].values


# ── temperature_summary ──────────────────────────────────────────────────────


def test_temperature_summary():
    records = []
    for temp, vid, label in [(0.7, "P6a", "Baseline"), (1.0, "P6b", "Moderate")]:
        for _ in range(3):
            records.append(
                _make_record(
                    phase="gen_params_sweep",
                    dimension="gen_params",
                    variant_id=vid,
                    variant_label=label,
                    temperature=temp,
                    overall_ai_prob=0.9 - temp * 0.2,
                )
            )
    df = _make_df(records)
    result = temperature_summary(df)
    assert len(result) == 2
    assert result.iloc[0]["temperature"] < result.iloc[1]["temperature"]


def test_temperature_summary_empty():
    df = _make_df([_make_record(phase="ablation")])
    result = temperature_summary(df)
    assert result.empty


# ── human_baseline_summary ───────────────────────────────────────────────────


def test_human_baseline_summary():
    records = [
        _make_record(
            phase="human_baseline",
            dimension="human",
            variant_id="human_baseline",
            overall_ai_prob=prob,
        )
        for prob in [0.05, 0.10, 0.20, 0.30, 0.08]
    ]
    df = _make_df(records)
    result = human_baseline_summary(df)
    assert len(result) == 1
    assert result.iloc[0]["n_essays"] == 5
    # 0.20 and 0.30 are >= 0.15 threshold → 2/5 = 0.4
    assert result.iloc[0]["false_positive_rate"] == pytest.approx(0.4)


def test_human_baseline_summary_empty():
    df = _make_df([_make_record(phase="ablation")])
    result = human_baseline_summary(df)
    assert result.empty


# ── export_summary_csv ───────────────────────────────────────────────────────


def test_export_summary_csv(tmp_path):
    df = _ablation_dataset()
    out = tmp_path / "summary.csv"
    result_path = export_summary_csv(df, out)
    assert result_path.exists()

    loaded = pd.read_csv(result_path)
    assert len(loaded) == 3
    assert "mean_ai_prob" in loaded.columns
