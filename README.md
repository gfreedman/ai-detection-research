# Adversarial Robustness of AI Text Detectors

**Research Question:** How fragile are AI text detectors (GPTZero) to prompt engineering, and what linguistic features most influence detection?

This project systematically tests whether prompt engineering alone — without post-processing or paraphrasing tools — can reduce AI detection rates. It applies the same adversarial testing methodology used in security research (red-teaming, penetration testing) to understand what AI detectors actually measure and how easily those signals can be manipulated.

## Hypothesis

AI text detectors primarily rely on **perplexity** (word predictability) and **burstiness** (variance in sentence complexity). Prompt engineering that increases lexical unpredictability and structural irregularity will reduce detection rates.

## Experimental Design

### Prompt Taxonomy (Independent Variables)

Prompts are organized into 6 tiers, each tested independently via ablation:

| Tier | Dimension | Variants | What it tests |
|------|-----------|----------|---------------|
| 1 | Voice & Persona | 4 | Does adopting a student persona reduce detection? |
| 2 | Structure | 4 | Does non-standard essay structure help? |
| 3 | Linguistic Texture | 5 | Do hedging, errors, varied sentence length affect scores? |
| 4 | Content Specificity | 4 | Do personal anecdotes or obscure references help? |
| 5 | Meta-Instructions | 4 | Does "write like a human" actually work? |
| 6 | Generation Parameters | 4 | How does temperature/top_p affect detection? |

### Controls

- **5 fixed essay topics** tested with every prompt variant
- **Target word count:** 450-550 words
- **N >= 5 runs** per variant (N=10 for composites) — single runs prove nothing
- **Human baselines** — real student essays run through GPTZero for false positive calibration

### Statistical Rigor

- **Cohen's d** effect sizes for each variant vs. its baseline
- **Mann-Whitney U tests** for significance
- **Cross-detector validation** against a second detector
- **Per-topic breakdowns** to check consistency

## Key Findings

*Findings will be populated after running the full experiment pipeline. See `notebooks/analysis.ipynb` for the full analysis.*

## Setup

### Prerequisites

- Python 3.11+
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier is sufficient)
- A [GPTZero API key](https://gptzero.me/docs)

### Installation

```bash
git clone https://github.com/your-username/ai-detection-research.git
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
#   GPTZERO_API_KEY=your_key_here
```

## Reproducing the Experiment

### 1. Add Human Baselines

Place real student essays as `.txt` files in `data/human_baselines/`. One essay per file.

### 2. Run the Full Pipeline

```python
from src.gemini_client import GeminiClient
from src.detector_client import GPTZeroClient
from src.prompt_registry import PromptRegistry
from src.experiment_runner import ExperimentRunner

gemini = GeminiClient()
detector = GPTZeroClient()
registry = PromptRegistry()
runner = ExperimentRunner(gemini, detector, registry)

# Run everything in order:
# 1. Human baselines → 2. Temperature sweep → 3. Ablations → 4. Composites
results = runner.run_full_pipeline()
```

Or run phases individually:

```python
# Phase 1: Calibrate with human essays
runner.run_human_baselines()

# Phase 2: Find optimal temperature
runner.run_gen_params_sweep()

# Phase 3: Test each prompt dimension (use best temp from Phase 2)
runner.run_all_ablations(temperature=1.0, top_p=0.95)

# Phase 4: Combine winners (after updating taxonomy.yaml composite_prompts)
runner.run_composites()
```

### 3. Analyze Results

Open `notebooks/analysis.ipynb` in Jupyter:

```bash
jupyter notebook notebooks/analysis.ipynb
```

Results are logged to `data/raw_results.jsonl` (append-only, one JSON object per run).

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
ai-detection-research/
├── src/
│   ├── config.py              # Central config: thresholds, models, retry logic
│   ├── gemini_client.py       # Gemini wrapper with retry + word count validation
│   ├── detector_client.py     # Abstract base + GPTZero implementation
│   ├── experiment_runner.py   # Orchestrator: generate → detect → log
│   ├── prompt_registry.py     # Loads taxonomy.yaml into runnable prompt configs
│   └── analysis.py            # Score aggregation, stats, export helpers
├── prompts/
│   └── taxonomy.yaml          # All prompt variants defined as structured data
├── data/
│   ├── human_baselines/       # Real student essays (control group)
│   ├── raw_results.jsonl      # Experimental output (append-only)
│   └── summary.csv            # Aggregated results
├── notebooks/
│   └── analysis.ipynb         # Reproducible analysis + visualizations
└── tests/                     # Unit tests for all modules
```

## Limitations

- **Single LLM family:** Tested with Gemini models only. Results may not transfer to GPT-4, Claude, or Llama.
- **English only:** All prompts and topics are in English. Detectors may behave differently for other languages.
- **Essay genre only:** Tested with 500-word high school essays. Detector behavior on code, creative writing, or technical prose may differ.
- **GPTZero-primary:** GPTZero is the primary detector. Cross-validation with a second detector is included but not exhaustive.
- **Point-in-time snapshot:** Detectors are constantly updated. These results reflect the detector behavior at the time of testing.

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
