# Adversarial Robustness of AI Text Detectors

**Research Question:** How fragile are AI text detectors to prompt engineering, and what linguistic features most influence detection?

This project systematically tests whether prompt engineering alone — without post-processing or paraphrasing tools — can reduce AI detection rates. It applies the same adversarial testing methodology used in security research (red-teaming, penetration testing) to understand what AI detectors actually measure and how easily those signals can be manipulated.

## Key Findings

700 experiments. Gemini 2.0 Flash generating 500-word essays. ZeroGPT as the detector.

**Prompt engineering crushes temperature tuning.** Raising temperature from 0.7 to 1.5 only drops detection from 100% to 82%. A single prompt instruction ("vary sentence length dramatically") drops it to 9.5%.

**Three prompt dimensions independently beat the 15% detection threshold:**

| Rank | Variant | What it does | Mean AI % | Pass Rate | Cohen's d |
|------|---------|-------------|-----------|-----------|-----------|
| 1 | P3b — Varied sentence length | "Mix 4-word fragments with 35-word run-ons" | 9.5% | 84% | 3.80 |
| 2 | P2d — Irregular paragraphs | "2-3 short paragraphs and one long rambling one" | 10.5% | 80% | 3.32 |
| 3 | P1c — B+ student persona | System: "You write well but not perfectly" | 11.9% | 76% | 3.30 |

**Combining winners is devastating.** Effects are additive — the more dimensions you stack, the lower detection goes:

| Composite | Components | Mean AI % | Pass Rate |
|-----------|-----------|-----------|-----------|
| Top 3 | persona + structure + texture | 2.8% | **100%** |
| Top 3 + meta | + "night before it's due" | 2.0% | **100%** |
| All 5 winners | + obscure source reference | 1.2% | **100%** |

See [`prompts/winning_prompts.md`](prompts/winning_prompts.md) for the exact prompt text, and [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) for the full statistical analysis with visualizations.

## Hypothesis

AI text detectors primarily rely on **perplexity** (word predictability) and **burstiness** (variance in sentence complexity). Prompt engineering that increases lexical unpredictability and structural irregularity will reduce detection rates.

**Verdict:** Supported. The two most effective dimensions — varied sentence length (burstiness) and irregular paragraphs (structural regularity) — directly target these signals. All effect sizes are large (Cohen's d > 3.0).

## Experimental Design

### Prompt Taxonomy (Independent Variables)

Prompts are organized into 6 tiers, each tested independently via ablation.

**Variant ID scheme:** IDs follow the pattern `PxY` where `x` is the tier number (1-6) and `Y` is the variant letter. Variant `a` is always the baseline (no modification). For example, `P3b` = Tier 3 (Texture), variant b (varied sentence length). See [`prompts/taxonomy.yaml`](prompts/taxonomy.yaml) for the full definition of every variant.

| Tier | Dimension | Variants | What it tests |
|------|-----------|----------|---------------|
| 1 | Voice & Persona | P1a-P1d | Does adopting a student persona reduce detection? |
| 2 | Structure | P2a-P2d | Does non-standard essay structure help? |
| 3 | Linguistic Texture | P3a-P3e | Do hedging, errors, varied sentence length affect scores? |
| 4 | Content Specificity | P4a-P4d | Do personal anecdotes or obscure references help? |
| 5 | Meta-Instructions | P5a-P5d | Does "write like a human" actually work? |
| 6 | Generation Parameters | P6a-P6d | How does temperature/top_p affect detection? |

### Controls

- **5 fixed essay topics** tested with every prompt variant
- **Target word count:** 450-550 words
- **N = 5 runs** per variant — single runs prove nothing
- **Human baselines** — real student essays can be run through the detector for false positive calibration

### Statistical Rigor

- **Cohen's d** effect sizes for each variant vs. its baseline
- **Mann-Whitney U tests** for significance (all p < 0.001 for top performers)
- **Per-topic breakdowns** to check consistency across subject matter
- **700 total experiment records** (625 ablation + 75 composite)

## Setup

### Prerequisites

- Python 3.11+
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier is sufficient)
- A [ZeroGPT API key](https://zerogpt.com) (~$10 for sufficient credits)

### Installation

```bash
git clone <repo-url>
cd ai-detection-research
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env and add your API keys:
#   GEMINI_API_KEY=your_key_here
#   ZEROGPT_API_KEY=your_key_here
```

## Reproducing the Experiment

### CLI (recommended)

```bash
# Global flags (--detector, -v) must come BEFORE the subcommand
python -m src --detector zerogpt -v sweep        # Phase 2: temperature sweep
python -m src --detector zerogpt -v ablate \
    --temperature 1.5 --top-p 1.0                # Phase 3: all ablations
python -m src --detector zerogpt -v composite    # Phase 4: composite tests
```

### Python API

```python
from src.gemini_client import GeminiClient
from src.detector_client import ZeroGPTClient
from src.prompt_registry import PromptRegistry
from src.experiment_runner import ExperimentRunner

gemini = GeminiClient()
detector = ZeroGPTClient()
registry = PromptRegistry()
runner = ExperimentRunner(gemini, detector, registry)

# Run phases individually:
runner.run_gen_params_sweep()                          # Phase 2
runner.run_all_ablations(temperature=1.5, top_p=1.0)  # Phase 3
runner.run_composites()                                # Phase 4
```

### Crash Resume

Experiments automatically skip already-completed runs on restart. If a run crashes due to API quota exhaustion (common with both Gemini free tier and ZeroGPT), simply re-run the same command — it will pick up where it left off.

### Analyze Results

```bash
jupyter notebook notebooks/analysis.ipynb
```

Results are logged to `data/raw_results.jsonl` (append-only, one JSON object per run). The included dataset contains all 700 records from our experiment.

## Running Tests

```bash
pytest tests/ -v
```

79 tests covering all modules. All tests use mocks — no API keys needed.

## Project Structure

```
ai-detection-research/
├── src/
│   ├── __main__.py            # CLI entrypoint
│   ├── config.py              # Central config: thresholds, models, retry logic
│   ├── gemini_client.py       # Gemini wrapper with retry + word count validation
│   ├── detector_client.py     # Abstract base + ZeroGPT/GPTZero implementations
│   ├── experiment_runner.py   # Orchestrator: generate -> detect -> log
│   ├── prompt_registry.py     # Loads taxonomy.yaml into runnable prompt configs
│   └── analysis.py            # Score aggregation, stats, export helpers
├── prompts/
│   ├── taxonomy.yaml          # All prompt variants defined as structured data
│   └── winning_prompts.md     # The exact prompts that work best
├── data/
│   ├── human_baselines/       # Real student essays (control group)
│   └── raw_results.jsonl      # 700 experiment records
├── notebooks/
│   └── analysis.ipynb         # Full analysis with visualizations
└── tests/                     # 79 unit tests (fully mocked)
```

For the full experimental design rationale, hypothesis development, and implementation plan, see [`CLAUDE.md`](CLAUDE.md).

## Limitations

- **Single detector:** Tested against ZeroGPT only. Results may not transfer to GPTZero, Originality.ai, or other detectors. Cross-detector validation is an important next step.
- **Single LLM family:** Tested with Gemini 2.0 Flash only. Results may differ for GPT-4, Claude, or Llama.
- **English only:** All prompts and topics are in English.
- **Essay genre only:** Tested with 500-word high school essays. Detector behavior on code, creative writing, or technical prose may differ.
- **No human baselines:** We did not collect real student essays for false positive calibration. The framework supports this (add `.txt` files to `data/human_baselines/`).
- **Point-in-time snapshot:** Detectors are constantly updated. These results reflect ZeroGPT's behavior at the time of testing (February 2026).

## Ethics Statement

### What this project is

This is **adversarial robustness research** — the same class of work as red-teaming LLMs, testing spam filters, or penetration testing network security. The goal is to produce reproducible evidence about detector fragility so that:

1. **Detector developers** can identify and fix weaknesses in their classifiers
2. **Institutions** deploying detectors can make informed decisions about their reliability
3. **Researchers** studying AI detection have systematic data on failure modes

### Why this matters

AI text detectors are being deployed in high-stakes settings — university honor code enforcement, hiring screening, publishing gatekeeping — often with little transparency about their accuracy or failure modes. Research has shown that these detectors have [documented bias against non-native English speakers](https://arxiv.org/abs/2304.02819) (Liang et al., 2023). Understanding how and why detectors fail is essential for equity and fairness.

### What this project is not

This project is not a tool or guide for academic dishonesty. The prompts, methodology, and findings are published openly precisely because transparency serves the public interest — just as publishing security vulnerabilities (responsible disclosure) drives improvements in security systems.

### Principles

- All code, data, and methodology are published for transparency and reproducibility
- Findings should inform detector improvement, not circumvent academic integrity
- We report both what works and what doesn't — negative results are equally valuable
- Human baseline data contextualizes all claims (a "pass" rate means nothing without knowing the false positive rate)
