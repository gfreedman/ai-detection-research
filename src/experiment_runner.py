"""
Orchestrator: generate essays -> detect AI -> log results to JSONL.

This module implements the four-phase experimental pipeline:

1. **Human baselines** -- run real student essays through the detector to
   establish the false-positive rate.
2. **Gen-params sweep** (Tier 6) -- vary temperature/top_p while keeping the
   prompt fixed, to find the optimal generation parameters.
3. **Ablations** (Tiers 1-5) -- test every prompt variant in each dimension
   at the best temperature from the sweep.
4. **Composite prompts** -- combine the top performers from each tier and test
   the combined prompts with a higher N.

All results are appended to a JSONL file (one JSON object per line) so that
partial runs survive crashes and the file never needs to be held in memory.

Data Classes
------------
- :class:`RunRecord` -- one row of experimental output (a single JSONL line).

Functions
---------
- :func:`_log_record`       -- append a :class:`RunRecord` to the JSONL file.
- :func:`_build_record`     -- assemble a :class:`RunRecord` from generation +
  detection outputs.
- :func:`_short`            -- truncate a topic string for log readability.
- :func:`_log_phase_summary` -- emit an INFO log summarising a completed phase.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.detector_client import DetectorClient, DetectorResult
from src.gemini_client import GeminiClient, GenerationResult
from src.prompt_registry import (
    CompositePrompt,
    GenParamsVariant,
    PromptRegistry,
    PromptVariant,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class: one row of experimental output
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """
    One row of experimental output, written as a single JSONL line.

    Every field is a flat, JSON-serialisable value so the record can be dumped
    directly with ``json.dumps(asdict(record))``.

    @param timestamp:            ISO-8601 UTC timestamp of the run.
    @param phase:                Experiment phase (``"human_baseline"``,
                                 ``"gen_params_sweep"``, ``"ablation"``,
                                 ``"composite"``).
    @param dimension:            Prompt dimension (e.g. ``"persona"``,
                                 ``"gen_params"``, ``"composite"``).
    @param variant_id:           Variant identifier (e.g. ``"P1b"``).
    @param variant_label:        Human-readable variant name.
    @param topic:                Essay topic used for this run.
    @param run_index:            0-based run counter within this variant/topic.

    @param model:                Gemini model used (``None`` for human baselines).
    @param temperature:          Sampling temperature (``None`` for baselines).
    @param top_p:                Nucleus-sampling threshold (``None`` for baselines).
    @param top_k:                Top-k cutoff (``None`` for baselines).
    @param word_count:           Word count of the essay.
    @param generation_latency_ms: API latency in milliseconds (``None`` for baselines).
    @param generation_attempts:  Number of generation attempts (``None`` for baselines).

    @param essay_text:           The full essay text.

    @param detector:             Detector name (e.g. ``"gptzero"``, ``"zerogpt"``).
    @param overall_ai_prob:      Detector's overall AI probability (0-1).
    @param burstiness:           Burstiness score (``None`` if not reported).
    @param flagged_sentence_pct: Percentage of sentences flagged as AI.
    @param passes_threshold:     ``True`` if ``overall_ai_prob < DETECTION_PASS_THRESHOLD``.

    @param system_prompt:        Exact system prompt used (for reproducibility).
    @param user_prompt:          Exact user prompt used (for reproducibility).
    """

    timestamp: str
    phase: str
    dimension: str
    variant_id: str
    variant_label: str
    topic: str
    run_index: int

    # Generation metadata (None for human baselines)
    model: str | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    word_count: int | None
    generation_latency_ms: int | None
    generation_attempts: int | None

    # Essay text
    essay_text: str

    # Detection results
    detector: str
    overall_ai_prob: float
    burstiness: float | None
    flagged_sentence_pct: float
    passes_threshold: bool

    # Full prompts used (for reproducibility)
    system_prompt: str
    user_prompt: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_record(record: RunRecord, output_path: Path) -> None:
    """
    Append a :class:`RunRecord` as a JSON line to the output file.

    Creates the parent directory if it does not exist.

    @param record:      The record to serialise and append.
    @param output_path: Path to the JSONL file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _build_record(
    phase: str,
    dimension: str,
    variant_id: str,
    variant_label: str,
    topic: str,
    run_index: int,
    system_prompt: str,
    user_prompt: str,
    essay_text: str,
    gen_result: GenerationResult | None,
    det_result: DetectorResult,
    detector_name: str,
) -> RunRecord:
    """
    Assemble a :class:`RunRecord` from generation and detection outputs.

    For human baselines *gen_result* is ``None`` and all generation-related
    fields are set accordingly.

    @param phase:          Experiment phase identifier.
    @param dimension:      Prompt dimension name.
    @param variant_id:     Variant identifier.
    @param variant_label:  Human-readable variant label.
    @param topic:          Essay topic string.
    @param run_index:      0-based run counter.
    @param system_prompt:  System prompt that was used.
    @param user_prompt:    User prompt that was used.
    @param essay_text:     The essay text (generated or human-written).
    @param gen_result:     Generation metadata (``None`` for human baselines).
    @param det_result:     Detection result from the detector.
    @param detector_name:  Short name of the detector (e.g. ``"gptzero"``).
    @returns:              A fully populated :class:`RunRecord`.
    """

    return RunRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        phase=phase,
        dimension=dimension,
        variant_id=variant_id,
        variant_label=variant_label,
        topic=topic,
        run_index=run_index,

        # Generation metadata -- None for human baselines
        model=gen_result.model if gen_result else None,
        temperature=gen_result.params.temperature if gen_result else None,
        top_p=gen_result.params.top_p if gen_result else None,
        top_k=gen_result.params.top_k if gen_result else None,
        word_count=gen_result.word_count if gen_result else len(essay_text.split()),
        generation_latency_ms=gen_result.latency_ms if gen_result else None,
        generation_attempts=gen_result.attempts if gen_result else None,

        # Essay + detection
        essay_text=essay_text,
        detector=detector_name,
        overall_ai_prob=det_result.overall_ai_prob,
        burstiness=det_result.burstiness,
        flagged_sentence_pct=det_result.flagged_sentence_pct,
        passes_threshold=det_result.overall_ai_prob < config.DETECTION_PASS_THRESHOLD,

        # Prompts (for reproducibility)
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _short(topic: str, max_len: int = 40) -> str:
    """
    Truncate a topic string for log readability.

    @param topic:   The full topic string.
    @param max_len: Maximum character length before truncation.
    @returns:       The truncated string, with ``"..."`` appended if shortened.
    """

    return topic[:max_len] + "..." if len(topic) > max_len else topic


def _log_phase_summary(phase: str, records: list[RunRecord]) -> None:
    """
    Emit an INFO log line summarising the results of a completed phase.

    Reports total runs, mean AI probability, and pass rate.

    @param phase:   Phase name for the log prefix.
    @param records: All :class:`RunRecord` objects from the phase.
    """

    if not records:
        return

    probs = [r.overall_ai_prob for r in records]
    passing = sum(1 for r in records if r.passes_threshold)

    logger.info(
        "%s complete: %d runs, mean AI prob %.3f, pass rate %d/%d (%.0f%%)",
        phase,
        len(records),
        sum(probs) / len(probs),
        passing,
        len(records),
        passing / len(records) * 100,
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def _load_completed_keys(path: Path, phase: str) -> set[tuple[str, str, str, int]]:
    """
    Scan the JSONL results file for already-completed experiment runs.

    This enables **crash-resume**: if the experiment crashes mid-dimension
    (e.g. due to API quota exhaustion), restarting the same command will
    skip runs that already have results, avoiding duplicate records and
    wasted API credits.

    Each run is uniquely identified by the tuple
    ``(dimension, variant_id, topic, run_index)``.  Corrupt JSON lines
    are silently skipped so that a partially-written trailing line does
    not prevent resumption.

    @param path:  Path to the JSONL results file.
    @param phase: Experiment phase to filter on (e.g. ``"ablation"``).
    @returns:     A set of ``(dimension, variant_id, topic, run_index)``
                  tuples for all completed runs in that phase.
    """

    keys: set[tuple[str, str, str, int]] = set()
    if not path.exists():
        return keys
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip corrupt/partial trailing lines
            if r.get("phase") == phase:
                keys.add((
                    r.get("dimension", ""),
                    r.get("variant_id", ""),
                    r.get("topic", ""),
                    r.get("run_index", -1),
                ))
    return keys


class ExperimentRunner:
    """
    Runs the full experimental pipeline: generate -> detect -> log.

    The runner coordinates the :class:`GeminiClient`, a :class:`DetectorClient`,
    and the :class:`PromptRegistry` to execute each phase of the experiment.
    All results are appended to a JSONL file as they are produced.

    Usage::

        runner = ExperimentRunner(gemini, detector, registry)
        results = runner.run_full_pipeline(best_temperature=1.0, best_top_p=0.95)
    """

    def __init__(
        self,
        gemini: GeminiClient,
        detector: DetectorClient,
        registry: PromptRegistry,
        output_path: Path | None = None,
        rate_limit_delay: float = config.RATE_LIMIT_DELAY_S,
    ):
        """
        Initialise the experiment runner.

        @param gemini:           A configured :class:`GeminiClient` for essay generation.
        @param detector:         A configured :class:`DetectorClient` for AI detection.
        @param registry:         The :class:`PromptRegistry` with loaded taxonomy.
        @param output_path:      Path to the JSONL results file.  Defaults to
                                 ``config.RAW_RESULTS_PATH``.
        @param rate_limit_delay: Seconds to sleep between consecutive API calls
                                 to respect rate limits.
        """

        self._gemini = gemini
        self._detector = detector
        self._registry = registry
        self._output_path = output_path or config.RAW_RESULTS_PATH
        self._delay = rate_limit_delay

    # ── Phase 1: Human Baselines ───────────────────────────────────────────

    def run_human_baselines(
        self,
        baselines_dir: Path | None = None,
    ) -> list[RunRecord]:
        """
        Run human-written essays through the detector for calibration.

        Reads all ``.txt`` files from *baselines_dir*.  Each file is treated as
        one essay; the filename (without extension) becomes the topic identifier.

        @param baselines_dir: Directory containing ``.txt`` essay files.
                              Defaults to ``config.HUMAN_BASELINES_DIR``.
        @returns: A list of :class:`RunRecord` objects (one per essay).
        """

        bdir = baselines_dir or config.HUMAN_BASELINES_DIR
        txt_files = sorted(bdir.glob("*.txt"))

        if not txt_files:
            logger.warning(
                "No .txt files found in %s -- skipping human baselines.", bdir
            )
            return []

        records: list[RunRecord] = []

        for i, path in enumerate(txt_files):

            text = path.read_text().strip()
            topic_label = path.stem

            logger.info(
                "Human baseline %d/%d: %s", i + 1, len(txt_files), topic_label
            )

            # -- Detect the human essay -------------------------------------
            det = self._detector.check(text)

            # -- Build and persist the record -------------------------------
            record = _build_record(
                phase="human_baseline",
                dimension="human",
                variant_id="human_baseline",
                variant_label="Human-written",
                topic=topic_label,
                run_index=0,
                system_prompt="",
                user_prompt="",
                essay_text=text,
                gen_result=None,
                det_result=det,
                detector_name=self._detector.name,
            )

            _log_record(record, self._output_path)
            records.append(record)
            time.sleep(self._delay)

        logger.info(
            "Human baselines complete. %d essays, mean AI prob: %.3f",
            len(records),
            sum(r.overall_ai_prob for r in records) / len(records) if records else 0,
        )

        return records

    # ── Phase 2: Temperature / Gen-Params Sweep ────────────────────────────

    def run_gen_params_sweep(
        self,
        topics: list[str] | None = None,
        n_runs: int = config.RUNS_PER_VARIANT,
    ) -> list[RunRecord]:
        """
        Sweep Tier 6 generation parameters across all topics.

        Uses the baseline prompt (no persona, no texture, etc.) so the only
        independent variable is temperature/top_p.

        @param topics: Override the default topic list.
        @param n_runs: Number of runs per variant per topic.
        @returns:      A list of :class:`RunRecord` objects.
        """

        topics = topics or self._registry.topics
        gp_variants = self._registry.get_gen_params_variants()
        baseline_prompt = self._registry.get_baseline("persona")

        records: list[RunRecord] = []
        total = len(gp_variants) * len(topics) * n_runs
        done = 0

        for gp in gp_variants:
            for topic in topics:
                for run in range(n_runs):

                    done += 1
                    logger.info(
                        "[gen_params %s] topic=%s run=%d/%d (%d/%d total)",
                        gp.id, _short(topic), run + 1, n_runs, done, total,
                    )

                    user_prompt = baseline_prompt.user_prompt(topic)

                    # -- Generate the essay ---------------------------------
                    gen = self._gemini.generate(
                        user_prompt=user_prompt,
                        system_prompt=baseline_prompt.system_prompt,
                        temperature=gp.temperature,
                        top_p=gp.top_p,
                    )

                    # -- Run the detector -----------------------------------
                    det = self._detector.check(gen.text)

                    # -- Build and persist the record -----------------------
                    record = _build_record(
                        phase="gen_params_sweep",
                        dimension="gen_params",
                        variant_id=gp.id,
                        variant_label=gp.label,
                        topic=topic,
                        run_index=run,
                        system_prompt=baseline_prompt.system_prompt,
                        user_prompt=user_prompt,
                        essay_text=gen.text,
                        gen_result=gen,
                        det_result=det,
                        detector_name=self._detector.name,
                    )

                    _log_record(record, self._output_path)
                    records.append(record)
                    time.sleep(self._delay)

        _log_phase_summary("gen_params_sweep", records)
        return records

    # ── Phase 3: Per-Dimension Ablation ────────────────────────────────────

    def run_ablation(
        self,
        dimension: str,
        topics: list[str] | None = None,
        n_runs: int = config.RUNS_PER_VARIANT,
        temperature: float = config.DEFAULT_TEMPERATURE,
        top_p: float = config.DEFAULT_TOP_P,
    ) -> list[RunRecord]:
        """
        Test all variants in one prompt dimension across topics.

        Generation parameters are held fixed (ideally the best values from the
        Tier 6 sweep) so the only independent variable is the prompt variant.

        @param dimension:   Prompt dimension to ablate (e.g. ``"persona"``).
        @param topics:      Override the default topic list.
        @param n_runs:      Number of runs per variant per topic.
        @param temperature: Fixed temperature for all generations.
        @param top_p:       Fixed top_p for all generations.
        @returns:           A list of :class:`RunRecord` objects.
        """

        topics = topics or self._registry.topics
        variants = self._registry.get_dimension(dimension)

        # Resume support: skip records already in the JSONL.
        completed = _load_completed_keys(self._output_path, "ablation")

        records: list[RunRecord] = []
        total = len(variants) * len(topics) * n_runs
        done = 0
        skipped = 0

        for variant in variants:
            for topic in topics:
                for run in range(n_runs):

                    done += 1

                    if (dimension, variant.id, topic, run) in completed:
                        skipped += 1
                        continue

                    logger.info(
                        "[ablation %s/%s] topic=%s run=%d/%d (%d/%d total, %d skipped)",
                        dimension, variant.id, _short(topic),
                        run + 1, n_runs, done, total, skipped,
                    )

                    user_prompt = variant.user_prompt(topic)

                    # -- Generate the essay ---------------------------------
                    gen = self._gemini.generate(
                        user_prompt=user_prompt,
                        system_prompt=variant.system_prompt,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    # -- Run the detector -----------------------------------
                    det = self._detector.check(gen.text)

                    # -- Build and persist the record -----------------------
                    record = _build_record(
                        phase="ablation",
                        dimension=dimension,
                        variant_id=variant.id,
                        variant_label=variant.label,
                        topic=topic,
                        run_index=run,
                        system_prompt=variant.system_prompt,
                        user_prompt=user_prompt,
                        essay_text=gen.text,
                        gen_result=gen,
                        det_result=det,
                        detector_name=self._detector.name,
                    )

                    _log_record(record, self._output_path)
                    records.append(record)
                    time.sleep(self._delay)

        _log_phase_summary(f"ablation/{dimension}", records)
        return records

    def run_all_ablations(
        self,
        topics: list[str] | None = None,
        n_runs: int = config.RUNS_PER_VARIANT,
        temperature: float = config.DEFAULT_TEMPERATURE,
        top_p: float = config.DEFAULT_TOP_P,
    ) -> list[RunRecord]:
        """
        Run ablations for every prompt dimension (Tiers 1-5).

        Convenience wrapper that iterates over all prompt dimensions and
        delegates to :meth:`run_ablation` for each one.

        @param topics:      Override the default topic list.
        @param n_runs:      Number of runs per variant per topic.
        @param temperature: Fixed temperature for all generations.
        @param top_p:       Fixed top_p for all generations.
        @returns:           Combined list of :class:`RunRecord` objects.
        """

        all_records: list[RunRecord] = []

        for dim in self._registry.prompt_dimensions:
            records = self.run_ablation(
                dim,
                topics=topics,
                n_runs=n_runs,
                temperature=temperature,
                top_p=top_p,
            )
            all_records.extend(records)

        return all_records

    # ── Phase 4: Composite Prompts ─────────────────────────────────────────

    def run_composites(
        self,
        topics: list[str] | None = None,
        n_runs: int = config.RUNS_PER_COMPOSITE,
    ) -> list[RunRecord]:
        """
        Test composite prompts (combined best-of-each-tier) across all topics.

        Composites are defined in the taxonomy YAML after ablation results have
        been analysed.  If none are defined, this method returns an empty list.

        @param topics: Override the default topic list.
        @param n_runs: Number of runs per composite per topic (higher N for
                       final validation).
        @returns:      A list of :class:`RunRecord` objects.
        """

        topics = topics or self._registry.topics
        composites = self._registry.composites

        if not composites:
            logger.warning("No composite prompts defined -- skipping.")
            return []

        records: list[RunRecord] = []
        total = len(composites) * len(topics) * n_runs
        done = 0

        for ci, comp in enumerate(composites):
            for topic in topics:
                for run in range(n_runs):

                    done += 1
                    logger.info(
                        "[composite %d/%d '%s'] topic=%s run=%d/%d (%d/%d total)",
                        ci + 1, len(composites), comp.label,
                        _short(topic), run + 1, n_runs, done, total,
                    )

                    user_prompt = comp.user_prompt(topic)

                    # -- Generate the essay ---------------------------------
                    gen = self._gemini.generate(
                        user_prompt=user_prompt,
                        system_prompt=comp.system_prompt,
                        temperature=comp.temperature,
                        top_p=comp.top_p,
                    )

                    # -- Run the detector -----------------------------------
                    det = self._detector.check(gen.text)

                    # -- Build and persist the record -----------------------
                    record = _build_record(
                        phase="composite",
                        dimension="composite",
                        variant_id=f"composite_{ci}",
                        variant_label=comp.label,
                        topic=topic,
                        run_index=run,
                        system_prompt=comp.system_prompt,
                        user_prompt=user_prompt,
                        essay_text=gen.text,
                        gen_result=gen,
                        det_result=det,
                        detector_name=self._detector.name,
                    )

                    _log_record(record, self._output_path)
                    records.append(record)
                    time.sleep(self._delay)

        _log_phase_summary("composite", records)
        return records

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def best_params_from_sweep(
        sweep_records: list[RunRecord],
    ) -> tuple[float, float]:
        """
        Extract the best temperature and top_p from gen-params sweep records.

        Groups records by ``variant_id``, computes the mean ``overall_ai_prob``
        for each variant, and returns the (temperature, top_p) of the variant
        with the lowest mean detection probability.

        Falls back to the config defaults if *sweep_records* is empty.

        @param sweep_records: Records from :meth:`run_gen_params_sweep`.
        @returns: A ``(temperature, top_p)`` tuple.
        """

        if not sweep_records:
            return config.DEFAULT_TEMPERATURE, config.DEFAULT_TOP_P

        # Group AI probs by variant, keep the first-seen params per variant.
        variant_probs: dict[str, list[float]] = {}
        variant_params: dict[str, tuple[float, float]] = {}

        for r in sweep_records:
            variant_probs.setdefault(r.variant_id, []).append(r.overall_ai_prob)
            if r.variant_id not in variant_params:
                variant_params[r.variant_id] = (r.temperature, r.top_p)

        best_vid = min(
            variant_probs,
            key=lambda v: sum(variant_probs[v]) / len(variant_probs[v]),
        )

        return variant_params[best_vid]

    # ── Full Pipeline ──────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        best_temperature: float | None = None,
        best_top_p: float | None = None,
    ) -> dict[str, list[RunRecord]]:
        """
        Execute the full experiment in the prescribed order.

        The four phases run sequentially:

        1. Human baselines   -- calibrate the detector's false-positive rate.
        2. Gen-params sweep  -- find optimal temperature / top_p (Tier 6).
        3. Ablations         -- test all prompt variants at the best params.
           If *best_temperature* / *best_top_p* are ``None``, the optimal
           values are extracted automatically from the sweep results.
        4. Composite prompts -- validate combined best-of-each-tier prompts.

        @param best_temperature: Override temperature for ablations.  ``None``
                                 (default) = auto-select from sweep results.
        @param best_top_p:       Override top_p for ablations.  ``None``
                                 (default) = auto-select from sweep results.
        @returns: A dict keyed by phase name, each value a list of
                  :class:`RunRecord` objects.
        """

        results: dict[str, list[RunRecord]] = {}

        logger.info("=== Phase 1: Human Baselines ===")
        results["human_baselines"] = self.run_human_baselines()

        logger.info("=== Phase 2: Gen-Params Sweep ===")
        results["gen_params_sweep"] = self.run_gen_params_sweep()

        # Auto-select best params from sweep unless caller overrode.
        sweep_temp, sweep_top_p = self.best_params_from_sweep(
            results["gen_params_sweep"]
        )
        use_temp = best_temperature if best_temperature is not None else sweep_temp
        use_top_p = best_top_p if best_top_p is not None else sweep_top_p

        logger.info(
            "Ablation params: temperature=%.2f, top_p=%.2f%s",
            use_temp,
            use_top_p,
            " (from sweep)" if best_temperature is None else " (user override)",
        )

        logger.info("=== Phase 3: Ablations (Tiers 1-5) ===")
        results["ablations"] = self.run_all_ablations(
            temperature=use_temp,
            top_p=use_top_p,
        )

        logger.info("=== Phase 4: Composite Prompts ===")
        results["composites"] = self.run_composites()

        total = sum(len(v) for v in results.values())
        logger.info("=== Pipeline complete. %d total records logged. ===", total)

        return results
