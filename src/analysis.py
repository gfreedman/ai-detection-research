"""Score aggregation, statistical analysis, and export helpers for experiment results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats as sp_stats

from src import config


# ── Loading ──────────────────────────────────────────────────────────────────


def load_results(path: Path | None = None) -> pd.DataFrame:
    """Load raw_results.jsonl into a DataFrame.

    Each line is a JSON object written by the experiment runner.
    """
    path = path or config.RAW_RESULTS_PATH
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── Aggregation ──────────────────────────────────────────────────────────────


def summarize_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detection scores per (phase, dimension, variant_id).

    Returns one row per variant with: mean, std, N, pass_rate, median,
    mean_burstiness, and mean_flagged_sentence_pct.
    """
    group_cols = ["phase", "dimension", "variant_id", "variant_label"]
    agg = (
        df.groupby(group_cols, sort=False)
        .agg(
            n=("overall_ai_prob", "count"),
            mean_ai_prob=("overall_ai_prob", "mean"),
            std_ai_prob=("overall_ai_prob", "std"),
            median_ai_prob=("overall_ai_prob", "median"),
            pass_rate=("passes_threshold", "mean"),
            mean_burstiness=("burstiness", "mean"),
            mean_flagged_pct=("flagged_sentence_pct", "mean"),
        )
        .reset_index()
    )
    agg["std_ai_prob"] = agg["std_ai_prob"].fillna(0.0)
    return agg


def summarize_by_topic(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detection scores per (variant_id, topic).

    Useful for checking consistency across topics.
    """
    group_cols = ["variant_id", "variant_label", "topic"]
    agg = (
        df.groupby(group_cols, sort=False)
        .agg(
            n=("overall_ai_prob", "count"),
            mean_ai_prob=("overall_ai_prob", "mean"),
            std_ai_prob=("overall_ai_prob", "std"),
            pass_rate=("passes_threshold", "mean"),
        )
        .reset_index()
    )
    agg["std_ai_prob"] = agg["std_ai_prob"].fillna(0.0)
    return agg


# ── Statistical Tests ────────────────────────────────────────────────────────


def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses the pooled standard deviation. Returns 0.0 if both groups have
    zero variance (e.g., all identical scores).
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")

    mean_a, mean_b = group_a.mean(), group_b.mean()
    var_a, var_b = group_a.var(ddof=1), group_b.var(ddof=1)

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_sd = math.sqrt(pooled_var)

    if pooled_sd == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_sd


def compare_to_baseline(
    df: pd.DataFrame,
    dimension: str,
    metric: str = "overall_ai_prob",
) -> pd.DataFrame:
    """Compare each variant in a dimension to the baseline (variant ending in 'a').

    Returns a DataFrame with: variant_id, variant_label, baseline_mean,
    variant_mean, cohens_d, mann_whitney_U, p_value.
    """
    dim_df = df[df["dimension"] == dimension].copy()
    baseline_id = None
    for vid in dim_df["variant_id"].unique():
        if vid.endswith("a"):
            baseline_id = vid
            break

    if baseline_id is None:
        raise ValueError(f"No baseline variant found for dimension '{dimension}'")

    baseline_scores = dim_df.loc[dim_df["variant_id"] == baseline_id, metric]
    results = []

    for vid in dim_df["variant_id"].unique():
        if vid == baseline_id:
            continue
        variant_scores = dim_df.loc[dim_df["variant_id"] == vid, metric]
        label = dim_df.loc[dim_df["variant_id"] == vid, "variant_label"].iloc[0]

        d = cohens_d(baseline_scores, variant_scores)

        if len(baseline_scores) >= 2 and len(variant_scores) >= 2:
            u_stat, p_val = sp_stats.mannwhitneyu(
                baseline_scores, variant_scores, alternative="two-sided"
            )
        else:
            u_stat, p_val = float("nan"), float("nan")

        results.append(
            {
                "dimension": dimension,
                "variant_id": vid,
                "variant_label": label,
                "baseline_id": baseline_id,
                "baseline_mean": baseline_scores.mean(),
                "baseline_std": baseline_scores.std(),
                "variant_mean": variant_scores.mean(),
                "variant_std": variant_scores.std(),
                "cohens_d": d,
                "mann_whitney_U": u_stat,
                "p_value": p_val,
                "n_baseline": len(baseline_scores),
                "n_variant": len(variant_scores),
            }
        )

    return pd.DataFrame(results)


def compare_all_dimensions(
    df: pd.DataFrame,
    metric: str = "overall_ai_prob",
) -> pd.DataFrame:
    """Run compare_to_baseline for every prompt dimension in the data."""
    frames = []
    for dim in df["dimension"].unique():
        if dim in ("human", "composite", "gen_params"):
            continue
        try:
            frames.append(compare_to_baseline(df, dim, metric))
        except ValueError:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def rank_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Rank all non-baseline variants by mean detection probability (ascending).

    Lower AI prob = harder to detect = more effective.
    """
    summary = summarize_variants(df)
    non_baseline = summary[
        ~summary["variant_id"].str.endswith("a")
        & ~summary["phase"].isin(["human_baseline"])
    ].copy()
    return non_baseline.sort_values("mean_ai_prob").reset_index(drop=True)


# ── Temperature Analysis ─────────────────────────────────────────────────────


def temperature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize detection scores by temperature from the gen_params sweep."""
    sweep = df[df["phase"] == "gen_params_sweep"].copy()
    if sweep.empty:
        return pd.DataFrame()
    return (
        sweep.groupby(["variant_id", "variant_label", "temperature"], sort=False)
        .agg(
            n=("overall_ai_prob", "count"),
            mean_ai_prob=("overall_ai_prob", "mean"),
            std_ai_prob=("overall_ai_prob", "std"),
            pass_rate=("passes_threshold", "mean"),
            mean_burstiness=("burstiness", "mean"),
        )
        .reset_index()
        .sort_values("temperature")
    )


# ── Human Baseline Analysis ─────────────────────────────────────────────────


def human_baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize GPTZero scores on human-written essays."""
    human = df[df["phase"] == "human_baseline"].copy()
    if human.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "n_essays": [len(human)],
            "mean_ai_prob": [human["overall_ai_prob"].mean()],
            "std_ai_prob": [human["overall_ai_prob"].std()],
            "median_ai_prob": [human["overall_ai_prob"].median()],
            "false_positive_rate": [(human["overall_ai_prob"] >= config.DETECTION_PASS_THRESHOLD).mean()],
            "mean_burstiness": [human["burstiness"].mean()],
            "mean_flagged_pct": [human["flagged_sentence_pct"].mean()],
        }
    )


# ── Export ───────────────────────────────────────────────────────────────────


def export_summary_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Export the variant summary to a CSV file."""
    path = path or config.SUMMARY_CSV_PATH
    summary = summarize_variants(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path
