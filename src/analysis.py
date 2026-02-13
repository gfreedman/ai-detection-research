"""
Score aggregation, statistical analysis, and export helpers for experiment results.

This module provides the analytical backbone of the project.  It loads JSONL
results into a pandas DataFrame, aggregates scores by variant/topic, runs
statistical comparisons (Mann-Whitney U, Cohen's d), and exports summaries.

Sections
--------
- **Loading**            -- :func:`load_results` reads the JSONL into a DataFrame.
- **Aggregation**        -- :func:`summarize_variants` and :func:`summarize_by_topic`
  group scores and compute mean, std, pass rate, etc.
- **Statistical Tests**  -- :func:`cohens_d` and :func:`compare_to_baseline` quantify
  effect sizes and significance per dimension.
- **Temperature**        -- :func:`temperature_summary` summarises the gen-params sweep.
- **Human Baselines**    -- :func:`human_baseline_summary` reports false-positive rates.
- **Ranking**            -- :func:`rank_variants` orders all non-baseline variants by
  effectiveness (lower AI prob = harder to detect).
- **Export**             -- :func:`export_summary_csv` writes the variant summary to CSV.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats as sp_stats

from src import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(path: Path | None = None) -> pd.DataFrame:
    """
    Load ``raw_results.jsonl`` into a pandas DataFrame.

    Each non-blank line in the file is expected to be a JSON object written by
    the experiment runner.  Blank lines are silently skipped.

    @param path: Path to the JSONL file.  Defaults to ``config.RAW_RESULTS_PATH``.
    @returns:    A DataFrame with one row per experiment run.
    """

    path = path or config.RAW_RESULTS_PATH
    records = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Convert timestamps to proper datetime objects for downstream analysis.
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate detection scores per (phase, dimension, variant_id).

    Groups by phase/dimension/variant and computes descriptive statistics for
    the overall AI probability, pass rate, burstiness, and flagged sentence
    percentage.

    @param df: A DataFrame of experiment results (from :func:`load_results`).
    @returns:  One row per variant with columns: ``n``, ``mean_ai_prob``,
               ``std_ai_prob``, ``median_ai_prob``, ``pass_rate``,
               ``mean_burstiness``, ``mean_flagged_pct``.
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

    # Fill NaN std (happens when N=1) with 0.0 for clean downstream use.
    agg["std_ai_prob"] = agg["std_ai_prob"].fillna(0.0)

    return agg


def summarize_by_topic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate detection scores per (variant_id, topic).

    Useful for checking whether a variant's performance is consistent across
    topics or whether it only works for specific subject matter.

    @param df: A DataFrame of experiment results.
    @returns:  One row per (variant, topic) pair with ``n``, ``mean_ai_prob``,
               ``std_ai_prob``, and ``pass_rate``.
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


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------


def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """
    Compute Cohen's d effect size between two groups using the pooled
    standard deviation.

    Cohen's d quantifies the *magnitude* of the difference between two groups,
    independent of sample size.  Conventional thresholds: 0.2 = small,
    0.5 = medium, 0.8 = large.

    @param group_a: Scores from the first group (typically the baseline).
    @param group_b: Scores from the second group (typically the variant).
    @returns:       Cohen's d (positive means group_a > group_b).
                    Returns ``float('nan')`` if either group has fewer than 2
                    observations.  Returns ``0.0`` if both groups have zero
                    variance (e.g. all identical scores).
    """

    n_a, n_b = len(group_a), len(group_b)

    # Need at least 2 observations per group to compute variance.
    if n_a < 2 or n_b < 2:
        return float("nan")

    mean_a, mean_b = group_a.mean(), group_b.mean()
    var_a, var_b = group_a.var(ddof=1), group_b.var(ddof=1)

    # Pooled variance: weighted average of within-group variances.
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_sd = math.sqrt(pooled_var)

    # Guard against division by zero (both groups have zero variance).
    if pooled_sd == 0:
        return 0.0

    return (mean_a - mean_b) / pooled_sd


def compare_to_baseline(
    df: pd.DataFrame,
    dimension: str,
    metric: str = "overall_ai_prob",
) -> pd.DataFrame:
    """
    Compare each variant in a dimension to the baseline (variant ending in ``'a'``).

    For each non-baseline variant, computes:
    - Cohen's d effect size relative to the baseline.
    - Mann-Whitney U statistic and p-value (non-parametric significance test).

    @param df:        A DataFrame of experiment results.
    @param dimension: The prompt dimension to analyse (e.g. ``"persona"``).
    @param metric:    The column to compare (default ``"overall_ai_prob"``).
    @returns:         A DataFrame with one row per non-baseline variant,
                      including ``cohens_d``, ``mann_whitney_U``, ``p_value``,
                      and summary statistics.
    @raises ValueError: If no baseline variant is found for *dimension*.
    """

    # -- Filter to just this dimension's data -------------------------------
    dim_df = df[df["dimension"] == dimension].copy()

    # -- Identify the baseline variant (ID ending in 'a') -------------------
    baseline_id = None

    for vid in dim_df["variant_id"].unique():
        if vid.endswith("a"):
            baseline_id = vid
            break

    if baseline_id is None:
        raise ValueError(f"No baseline variant found for dimension '{dimension}'")

    baseline_scores = dim_df.loc[dim_df["variant_id"] == baseline_id, metric]

    # -- Compare each non-baseline variant to the baseline ------------------
    results = []

    for vid in dim_df["variant_id"].unique():
        if vid == baseline_id:
            continue

        variant_scores = dim_df.loc[dim_df["variant_id"] == vid, metric]
        label = dim_df.loc[dim_df["variant_id"] == vid, "variant_label"].iloc[0]

        # Effect size
        d = cohens_d(baseline_scores, variant_scores)

        # Non-parametric significance test (does not assume normality)
        if len(baseline_scores) >= 2 and len(variant_scores) >= 2:
            u_stat, p_val = sp_stats.mannwhitneyu(
                baseline_scores,
                variant_scores,
                alternative="two-sided",
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
    """
    Run :func:`compare_to_baseline` for every prompt dimension in the data.

    Skips the ``"human"``, ``"composite"``, and ``"gen_params"`` dimensions,
    which are not standard ablation dimensions.

    @param df:     A DataFrame of experiment results.
    @param metric: The column to compare (default ``"overall_ai_prob"``).
    @returns:      A concatenated DataFrame of all per-dimension comparisons.
    """

    frames = []

    for dim in df["dimension"].unique():

        # Skip non-ablation dimensions.
        if dim in ("human", "composite", "gen_params"):
            continue

        try:
            frames.append(compare_to_baseline(df, dim, metric))
        except ValueError:
            # No baseline found for this dimension -- skip.
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def rank_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank all non-baseline variants by mean detection probability (ascending).

    Lower AI probability = harder to detect = more effective prompt variant.
    Excludes baseline variants (IDs ending in ``'a'``) and human baselines.

    @param df: A DataFrame of experiment results.
    @returns:  A ranked DataFrame with the most effective variants first.
    """

    summary = summarize_variants(df)

    # Filter out baselines and human-baseline rows.
    non_baseline = summary[
        ~summary["variant_id"].str.endswith("a")
        & ~summary["phase"].isin(["human_baseline"])
    ].copy()

    return non_baseline.sort_values("mean_ai_prob").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Temperature Analysis
# ---------------------------------------------------------------------------


def temperature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise detection scores by temperature from the gen-params sweep.

    Useful for identifying the optimal temperature range before running
    full ablations.

    @param df: A DataFrame of experiment results.
    @returns:  A DataFrame with one row per temperature variant, sorted by
               temperature ascending.  Empty DataFrame if no sweep data exists.
    """

    sweep = df[df["phase"] == "gen_params_sweep"].copy()

    if sweep.empty:
        return pd.DataFrame()

    return (
        sweep.groupby(
            ["variant_id", "variant_label", "temperature"], sort=False
        )
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


# ---------------------------------------------------------------------------
# Human Baseline Analysis
# ---------------------------------------------------------------------------


def human_baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise detector scores on human-written essays.

    Reports the false-positive rate (percentage of human essays flagged as AI)
    which contextualises all other detection results.  If the detector flags 30%
    of human essays, "passing" is less impressive.

    @param df: A DataFrame of experiment results.
    @returns:  A single-row DataFrame with ``n_essays``, ``mean_ai_prob``,
               ``std_ai_prob``, ``median_ai_prob``, ``false_positive_rate``,
               ``mean_burstiness``, and ``mean_flagged_pct``.
               Empty DataFrame if no human baselines exist.
    """

    human = df[df["phase"] == "human_baseline"].copy()

    if human.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "n_essays": [len(human)],
            "mean_ai_prob": [human["overall_ai_prob"].mean()],
            "std_ai_prob": [human["overall_ai_prob"].std()],
            "median_ai_prob": [human["overall_ai_prob"].median()],
            "false_positive_rate": [
                (human["overall_ai_prob"] >= config.DETECTION_PASS_THRESHOLD).mean()
            ],
            "mean_burstiness": [human["burstiness"].mean()],
            "mean_flagged_pct": [human["flagged_sentence_pct"].mean()],
        }
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_summary_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    """
    Export the variant summary to a CSV file.

    Creates the parent directory if it does not exist.

    @param df:   A DataFrame of experiment results.
    @param path: Output path.  Defaults to ``config.SUMMARY_CSV_PATH``.
    @returns:    The path the CSV was written to.
    """

    path = path or config.SUMMARY_CSV_PATH
    summary = summarize_variants(df)

    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)

    return path
