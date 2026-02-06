"""Orchestrator: generate essays → detect AI → log results to JSONL."""

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


@dataclass
class RunRecord:
    """One row of experimental output, written as a single JSONL line."""

    timestamp: str
    phase: str  # "human_baseline", "gen_params_sweep", "ablation", "composite"
    dimension: str  # e.g. "persona", "gen_params", "composite"
    variant_id: str  # e.g. "P1b", "composite_1", "human_baseline"
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

    # Full system/user prompts used (for reproducibility)
    system_prompt: str
    user_prompt: str


def _log_record(record: RunRecord, output_path: Path) -> None:
    """Append a RunRecord as a JSON line to the output file."""
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
    return RunRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        phase=phase,
        dimension=dimension,
        variant_id=variant_id,
        variant_label=variant_label,
        topic=topic,
        run_index=run_index,
        model=gen_result.model if gen_result else None,
        temperature=gen_result.params.temperature if gen_result else None,
        top_p=gen_result.params.top_p if gen_result else None,
        top_k=gen_result.params.top_k if gen_result else None,
        word_count=gen_result.word_count if gen_result else len(essay_text.split()),
        generation_latency_ms=gen_result.latency_ms if gen_result else None,
        generation_attempts=gen_result.attempts if gen_result else None,
        essay_text=essay_text,
        detector=detector_name,
        overall_ai_prob=det_result.overall_ai_prob,
        burstiness=det_result.burstiness,
        flagged_sentence_pct=det_result.flagged_sentence_pct,
        passes_threshold=det_result.overall_ai_prob < config.DETECTION_PASS_THRESHOLD,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


class ExperimentRunner:
    """Runs the full experimental pipeline: generate → detect → log."""

    def __init__(
        self,
        gemini: GeminiClient,
        detector: DetectorClient,
        registry: PromptRegistry,
        output_path: Path | None = None,
        rate_limit_delay: float = config.RATE_LIMIT_DELAY_S,
    ):
        self._gemini = gemini
        self._detector = detector
        self._registry = registry
        self._output_path = output_path or config.RAW_RESULTS_PATH
        self._delay = rate_limit_delay

    # ── Phase 1: Human Baselines ───────────────────────────────────────────────

    def run_human_baselines(self, baselines_dir: Path | None = None) -> list[RunRecord]:
        """Run human-written essays through the detector for calibration.

        Reads all .txt files from the baselines directory. Each file is one essay.
        The filename (without extension) is used as the topic identifier.
        """
        bdir = baselines_dir or config.HUMAN_BASELINES_DIR
        txt_files = sorted(bdir.glob("*.txt"))
        if not txt_files:
            logger.warning("No .txt files found in %s — skipping human baselines.", bdir)
            return []

        records: list[RunRecord] = []
        for i, path in enumerate(txt_files):
            text = path.read_text().strip()
            topic_label = path.stem

            logger.info("Human baseline %d/%d: %s", i + 1, len(txt_files), topic_label)
            det = self._detector.check(text)

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

    # ── Phase 2: Temperature / Gen-Params Sweep ────────────────────────────────

    def run_gen_params_sweep(
        self,
        topics: list[str] | None = None,
        n_runs: int = config.RUNS_PER_VARIANT,
    ) -> list[RunRecord]:
        """Sweep Tier 6 generation parameters across all topics.

        Uses the baseline prompt (no persona, no texture, etc.) so the only
        variable is temperature/top_p.
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
                    gen = self._gemini.generate(
                        user_prompt=user_prompt,
                        system_prompt=baseline_prompt.system_prompt,
                        temperature=gp.temperature,
                        top_p=gp.top_p,
                    )
                    det = self._detector.check(gen.text)

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

    # ── Phase 3: Per-Dimension Ablation ────────────────────────────────────────

    def run_ablation(
        self,
        dimension: str,
        topics: list[str] | None = None,
        n_runs: int = config.RUNS_PER_VARIANT,
        temperature: float = config.DEFAULT_TEMPERATURE,
        top_p: float = config.DEFAULT_TOP_P,
    ) -> list[RunRecord]:
        """Test all variants in one prompt dimension across topics.

        Uses fixed generation parameters (ideally the best from the sweep).
        """
        topics = topics or self._registry.topics
        variants = self._registry.get_dimension(dimension)

        records: list[RunRecord] = []
        total = len(variants) * len(topics) * n_runs
        done = 0

        for variant in variants:
            for topic in topics:
                for run in range(n_runs):
                    done += 1
                    logger.info(
                        "[ablation %s/%s] topic=%s run=%d/%d (%d/%d total)",
                        dimension, variant.id, _short(topic), run + 1, n_runs, done, total,
                    )

                    user_prompt = variant.user_prompt(topic)
                    gen = self._gemini.generate(
                        user_prompt=user_prompt,
                        system_prompt=variant.system_prompt,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    det = self._detector.check(gen.text)

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
        """Run ablations for every prompt dimension (Tiers 1-5)."""
        all_records: list[RunRecord] = []
        for dim in self._registry.prompt_dimensions:
            records = self.run_ablation(
                dim, topics=topics, n_runs=n_runs,
                temperature=temperature, top_p=top_p,
            )
            all_records.extend(records)
        return all_records

    # ── Phase 4: Composite Prompts ─────────────────────────────────────────────

    def run_composites(
        self,
        topics: list[str] | None = None,
        n_runs: int = config.RUNS_PER_COMPOSITE,
    ) -> list[RunRecord]:
        """Test composite prompts (combined best-of-each-tier) across all topics."""
        topics = topics or self._registry.topics
        composites = self._registry.composites
        if not composites:
            logger.warning("No composite prompts defined — skipping.")
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
                    gen = self._gemini.generate(
                        user_prompt=user_prompt,
                        system_prompt=comp.system_prompt,
                        temperature=comp.temperature,
                        top_p=comp.top_p,
                    )
                    det = self._detector.check(gen.text)

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

    # ── Full Pipeline ──────────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        best_temperature: float = config.DEFAULT_TEMPERATURE,
        best_top_p: float = config.DEFAULT_TOP_P,
    ) -> dict[str, list[RunRecord]]:
        """Execute the full experiment in the prescribed order.

        1. Human baselines
        2. Gen-params sweep (Tier 6)
        3. Ablations for Tiers 1-5 at the given temperature/top_p
        4. Composite prompts

        Returns a dict keyed by phase name.
        """
        results: dict[str, list[RunRecord]] = {}

        logger.info("═══ Phase 1: Human Baselines ═══")
        results["human_baselines"] = self.run_human_baselines()

        logger.info("═══ Phase 2: Gen-Params Sweep ═══")
        results["gen_params_sweep"] = self.run_gen_params_sweep()

        logger.info("═══ Phase 3: Ablations (Tiers 1-5) ═══")
        results["ablations"] = self.run_all_ablations(
            temperature=best_temperature, top_p=best_top_p,
        )

        logger.info("═══ Phase 4: Composite Prompts ═══")
        results["composites"] = self.run_composites()

        total = sum(len(v) for v in results.values())
        logger.info("═══ Pipeline complete. %d total records logged. ═══", total)
        return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _short(topic: str, max_len: int = 40) -> str:
    """Truncate a topic string for log readability."""
    return topic[:max_len] + "..." if len(topic) > max_len else topic


def _log_phase_summary(phase: str, records: list[RunRecord]) -> None:
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
