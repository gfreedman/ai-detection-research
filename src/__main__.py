"""
CLI entrypoint for running experiments.

Usage::

    python -m src baselines                     # Phase 1
    python -m src sweep                         # Phase 2
    python -m src ablate                        # Phase 3 (all dimensions)
    python -m src ablate --dimension persona    # Phase 3 (one dimension)
    python -m src composite                     # Phase 4
    python -m src full                          # All phases sequentially
    python -m src full --temperature 1.0        # Override sweep auto-select

    # Use ZeroGPT instead of GPTZero:
    python -m src sweep --detector zerogpt
"""

from __future__ import annotations

import argparse
import logging
import sys

from src import config
from src.detector_client import GPTZeroClient, ZeroGPTClient
from src.experiment_runner import ExperimentRunner
from src.gemini_client import GeminiClient
from src.prompt_registry import PromptRegistry


def _build_runner(args: argparse.Namespace) -> ExperimentRunner:
    """Construct the runner from CLI args."""

    gemini = GeminiClient()
    registry = PromptRegistry()

    if args.detector == "zerogpt":
        detector = ZeroGPTClient()
    else:
        detector = GPTZeroClient()

    return ExperimentRunner(
        gemini=gemini,
        detector=detector,
        registry=registry,
    )


def cmd_baselines(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    records = runner.run_human_baselines()
    print(f"Done. {len(records)} human baseline records logged.")


def cmd_sweep(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    records = runner.run_gen_params_sweep()
    temp, top_p = ExperimentRunner.best_params_from_sweep(records)
    print(f"Done. {len(records)} sweep records logged.")
    print(f"Best params: temperature={temp}, top_p={top_p}")


def cmd_ablate(args: argparse.Namespace) -> None:
    runner = _build_runner(args)

    kwargs = {}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p

    if args.dimension:
        records = runner.run_ablation(args.dimension, **kwargs)
    else:
        records = runner.run_all_ablations(**kwargs)

    print(f"Done. {len(records)} ablation records logged.")


def cmd_composite(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    records = runner.run_composites()
    print(f"Done. {len(records)} composite records logged.")


def cmd_full(args: argparse.Namespace) -> None:
    runner = _build_runner(args)

    override_temp = args.temperature
    override_top_p = args.top_p

    results = runner.run_full_pipeline(
        best_temperature=override_temp,
        best_top_p=override_top_p,
    )

    total = sum(len(v) for v in results.values())
    print(f"Done. {total} total records logged.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="AI Detection Research -- Experiment Runner",
    )
    parser.add_argument(
        "--detector",
        choices=["gptzero", "zerogpt"],
        default="gptzero",
        help="Which detector to use (default: gptzero)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- baselines --
    sub.add_parser("baselines", help="Run human baselines through detector")

    # -- sweep --
    sub.add_parser("sweep", help="Run temperature/gen-params sweep (Tier 6)")

    # -- ablate --
    p_ablate = sub.add_parser("ablate", help="Run prompt ablations (Tiers 1-5)")
    p_ablate.add_argument("--dimension", help="Single dimension to ablate (default: all)")
    p_ablate.add_argument("--temperature", type=float, help="Fixed temperature")
    p_ablate.add_argument("--top-p", type=float, dest="top_p", help="Fixed top_p")

    # -- composite --
    sub.add_parser("composite", help="Run composite prompt tests")

    # -- full --
    p_full = sub.add_parser("full", help="Run full pipeline (all 4 phases)")
    p_full.add_argument("--temperature", type=float, help="Override ablation temperature")
    p_full.add_argument("--top-p", type=float, dest="top_p", help="Override ablation top_p")

    args = parser.parse_args()

    # -- Configure logging --
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- Dispatch --
    commands = {
        "baselines": cmd_baselines,
        "sweep": cmd_sweep,
        "ablate": cmd_ablate,
        "composite": cmd_composite,
        "full": cmd_full,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
