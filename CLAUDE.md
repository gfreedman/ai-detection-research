# CLAUDE.md — Adversarial Robustness of AI Text Detectors

> **Note:** This is the original design document written before implementation began.
> The project was planned around GPTZero, but we pivoted to **ZeroGPT** after
> discovering that GPTZero's API pricing was prohibitive for the experiment scale.
> All references to GPTZero below reflect the original plan — see `README.md` for
> the actual methodology and results using ZeroGPT.

## Research Question

**How fragile are AI text detectors to prompt engineering, and what linguistic features most influence detection?**

This is adversarial robustness research — the same class of work as red-teaming LLMs or testing spam filters. The goal is to understand *what detectors actually measure* and *how easily those signals can be manipulated*, not to enable cheating.

---

## Why This Matters

AI detectors are being deployed in high-stakes settings (universities, hiring, publishing) with little transparency about their failure modes. Systematic adversarial testing is how the security and ML communities have always improved classifiers. This project produces reproducible evidence about detector fragility.

---

## Experimental Design

### Hypothesis

> AI text detectors primarily rely on **perplexity** (word predictability) and **burstiness** (variance in sentence complexity). Prompt engineering that increases lexical unpredictability and structural irregularity will reduce detection rates, even without post-processing.

### Variables

| Type | Variable | Details |
|------|----------|---------|
| **Independent** | Prompt template | Systematically varied across dimensions (see Prompt Taxonomy below) |
| **Independent** | Generation parameters | temperature, top_p, top_k — swept independently |
| **Independent** | Model | gemini-2.0-flash, gemini-1.5-pro (cross-model validation) |
| **Dependent** | `completely_generated_prob` | GPTZero's primary score (0-1) |
| **Dependent** | `overall_burstiness` | Sentence complexity variance |
| **Dependent** | Per-sentence flag rate | % of sentences individually flagged |
| **Control** | Essay topic | Fixed set of 5 topics, each tested with every prompt variant |
| **Control** | Word count | Target 450-550 words for all generations |
| **Control** | Human baselines | 5 actual student essays per topic for calibration |

### Statistical Rigor

- **N ≥ 5 generations per prompt variant** — single runs prove nothing. Report mean ± std.
- **Ablation study** — change one prompt dimension at a time against the baseline to isolate effects.
- **Multiple detectors** — GPTZero is the primary target but validate findings against at least one secondary detector (ZeroGPT, Originality.ai, or Sapling) to check if results generalize or are detector-specific.
- **Human baselines** — collect or source 5 real student essays per topic. Run them through GPTZero to establish the false positive rate. If GPTZero flags 30% of human essays, "passing" is less impressive.
- **Effect sizes** — don't just report p-values. Report Cohen's d or similar for each prompt dimension's impact.

---

## Prompt Taxonomy

Organize prompt variations into a systematic taxonomy rather than ad hoc iteration. Each dimension is a single variable to ablate.

### Tier 1: Voice & Persona
```
P1a: No persona (baseline)
P1b: "You are a 16-year-old named Alex writing for AP English."
P1c: "You are a B+ student who writes well but not perfectly."
P1d: Full persona with specific school, teacher name, personal context
```

### Tier 2: Structural Directives
```
P2a: No structure guidance (baseline)
P2b: "Don't use a 5-paragraph essay structure."
P2c: "Start in the middle of a thought. End abruptly."
P2d: "Use 2-3 very short paragraphs and one long rambling one."
```

### Tier 3: Linguistic Texture
```
P3a: No texture guidance (baseline)
P3b: "Vary sentence length dramatically — mix 4-word fragments with 35-word run-ons."
P3c: "Include hedging: 'I think', 'probably', 'sort of', 'I'm not sure but'."
P3d: "Use 1-2 minor grammatical errors and one awkward transition."
P3e: "Include a colloquialism or slang term."
```

### Tier 4: Content Specificity
```
P4a: Generic topic only (baseline)
P4b: "Reference a specific personal experience."
P4c: "Mention a real but obscure source you 'read in class'."
P4d: "Include a slightly wrong statistic that a student might misremember."
```

### Tier 5: Meta-Instructions
```
P5a: No meta-instructions (baseline)
P5b: "This is a rough first draft."
P5c: "You're writing this the night before it's due."
P5d: "Write like a human, not like an AI." (test if this naive approach works at all)
```

### Tier 6: Generation Parameters (not prompt text)
```
P6a: temperature=0.7, top_p=0.9 (baseline)
P6b: temperature=1.0, top_p=0.95
P6c: temperature=1.3, top_p=0.98
P6d: temperature=1.5, top_p=1.0
```

**Full factorial is too large.** Run ablations first (Tiers independently), then combine the top performers from each tier into composite prompts.

---

## Architecture

```
ai-detection-research/
├── CLAUDE.md                         # This file
├── README.md                         # Findings, setup, ethical framing
├── pyproject.toml                    # Use modern Python packaging
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Central config: thresholds, model names, retry logic
│   ├── gemini_client.py              # Thin Gemini wrapper — generation + metadata capture
│   ├── detector_client.py            # Abstract base + GPTZero impl (swap detectors easily)
│   ├── experiment_runner.py          # Orchestrator: generate → detect → log
│   ├── prompt_registry.py           # All prompt variants defined declaratively as data
│   └── analysis.py                   # Score aggregation, stats, export helpers
│
├── prompts/
│   └── taxonomy.yaml                 # Prompt taxonomy defined as structured YAML
│
├── data/
│   ├── human_baselines/              # Real student essays (control group)
│   ├── raw_results.jsonl             # One JSON object per run (append-only)
│   └── summary.csv                   # Aggregated results for notebook
│
├── notebooks/
│   └── analysis.ipynb                # Reproducible analysis + visualizations
│
└── tests/
    ├── test_gemini_client.py
    └── test_detector_client.py
```

### Key Design Decisions

1. **JSONL over JSON** for raw results — append-only, survives crashes mid-experiment, no need to hold full log in memory.
2. **Abstract detector interface** — `DetectorClient` base class with `check(text) → DetectorResult`. GPTZero is one implementation. Makes it trivial to add ZeroGPT or Originality.ai later.
3. **Prompt registry as data, not code** — define all prompt variants in `taxonomy.yaml` so the experiment runner just iterates over them. No prompt strings buried in Python.
4. **`pyproject.toml`** over `requirements.txt` — modern Python packaging standard.

---

## Implementation Plan

### Step 1: Scaffold + Config (30 min)

Create directory structure, `pyproject.toml`, `.env.example`, `.gitignore`, `config.py`.

`config.py` should centralize:
- API keys (from env vars)
- Model identifiers
- Rate limit delays (default 3s between calls)
- Detection pass threshold (default: `completely_generated_prob < 0.15`)
- Number of runs per variant (default: 5)

### Step 2: Gemini Client (30 min)

`gemini_client.py`:
- `generate(system_prompt, user_prompt, temperature, top_p, top_k) → GenerationResult`
- `GenerationResult` dataclass: `text`, `word_count`, `model`, `params`, `latency_ms`
- Retry logic with exponential backoff (3 retries)
- Validate word count is within target range; regenerate if not

### Step 3: Detector Client (45 min)

`detector_client.py`:
- Abstract `DetectorClient` with `check(text) → DetectorResult`
- `DetectorResult` dataclass: `overall_ai_prob`, `burstiness`, `per_sentence_scores`, `flagged_sentence_pct`, `raw_response`
- `GPTZeroClient(DetectorClient)` implementation
- Handle GPTZero API errors gracefully (rate limits return 429, quota exceeded, etc.)

**Important**: Check GPTZero's current API status before building. Their API access has changed multiple times. If unavailable or too expensive, pivot to:
1. **ZeroGPT** (free API available)
2. **Sapling AI Detector** (has an API)
3. **Selenium/Playwright automation of the GPTZero web UI** as a fallback (less ideal but works)

### Step 4: Prompt Registry (30 min)

`prompts/taxonomy.yaml`:
```yaml
dimensions:
  persona:
    P1a:
      label: "No persona"
      system_prompt: ""
      user_prompt_prefix: ""
    P1b:
      label: "Teen student"
      system_prompt: "You are Alex, a 16-year-old writing for AP English."
      user_prompt_prefix: ""
  # ... etc

topics:
  - "Write a 500-word essay about how social media affects teen mental health."
  - "Write a 500-word essay arguing whether school uniforms should be required."
  - "Write a 500-word essay about a book that changed your perspective."
  - "Write a 500-word essay about climate change and what your generation can do."
  - "Write a 500-word essay about whether standardized testing is fair."

composite_prompts:
  # Filled in after ablation results — combine top performers
  []
```

`prompt_registry.py` loads and resolves this YAML into runnable prompt configs.

### Step 5: Experiment Runner (1 hr)

`experiment_runner.py`:
```python
# Core loop pseudocode
def run_ablation(dimension: str, topic: str, n_runs: int = 5):
    """Test all variants in one dimension against one topic."""
    baseline = registry.get("baseline")
    variants = registry.get_dimension(dimension)

    for variant in [baseline] + variants:
        for run in range(n_runs):
            essay = gemini.generate(variant.system_prompt, variant.user_prompt(topic))
            scores = detector.check(essay.text)
            log_result(variant, topic, run, essay, scores)
            sleep(config.RATE_LIMIT_DELAY)

def run_composite(composite_prompt, topics, n_runs: int = 10):
    """Test a composite prompt across all topics with more runs."""
    for topic in topics:
        for run in range(n_runs):
            essay = gemini.generate(composite_prompt.system, composite_prompt.user(topic))
            scores = detector.check(essay.text)
            log_result(composite_prompt, topic, run, essay, scores)
            sleep(config.RATE_LIMIT_DELAY)
```

**Execution order:**
1. Run human baselines through GPTZero (calibration)
2. Run Tier 6 (temperature sweep) first — cheapest to test and most impactful
3. Run Tiers 1-5 ablations at best temperature from Tier 6
4. Identify top 2-3 performers per tier
5. Combine into composite prompts
6. Run composites with N=10 across all 5 topics
7. Validate winning composites on secondary detector

### Step 6: LLM-Assisted Prompt Evolution (Optional, 1 hr)

After the systematic ablation, optionally add a meta-prompting loop:

- Feed the winning composite prompt + its GPTZero scores back to Gemini
- Ask: "Given these AI detection scores, suggest 3 specific modifications to this prompt that would make the output harder to detect. Explain your reasoning."
- Test the suggestions
- Log whether LLM-suggested improvements outperform human-designed ones (interesting finding either way)

This is **Phase 2 of the research**, after the systematic ablation. Don't mix ad hoc iteration with controlled experimentation.

### Step 7: Analysis Notebook (1 hr)

`notebooks/analysis.ipynb` sections:

1. **Setup** — load JSONL, parse into DataFrame
2. **Human Baselines** — GPTZero scores on real essays. Establish false positive rate. This contextualizes all other results.
3. **Temperature Sweep** — plot detection prob vs. temperature. Find optimal range.
4. **Ablation Results** — for each dimension, box plot of detection prob across variants. Identify statistically significant improvements (Mann-Whitney U test or similar).
5. **Interaction Effects** — do persona + linguistic texture combine additively or is there synergy/interference?
6. **Composite Prompt Performance** — final scores with confidence intervals.
7. **Cross-Detector Validation** — do results hold on other detectors?
8. **Essay Comparison** — side-by-side examples with per-sentence highlighting showing which sentences get flagged and why.
9. **Findings Summary** — what actually works, what doesn't, what this tells us about how detectors work.

### Step 8: README + Ethics Statement (30 min)

README.md should include:
- Clear framing as adversarial robustness research
- Setup instructions (API keys, Python version, install)
- How to reproduce the full experiment
- Summary of key findings
- Limitations (single LLM family, English only, essay genre only)
- Ethics section (see below)

**Ethics statement must include:**
- This research tests detector robustness, analogous to penetration testing in security
- Findings should inform detector improvement, not enable academic dishonesty
- AI detectors have documented bias against non-native English speakers (cite Liang et al., 2023). Understanding their failure modes serves equity
- All code and data are published for transparency and reproducibility

### Step 9: Git + GitHub (15 min)

```bash
git add -A
git commit -m "Initial commit: experiment framework + prompt taxonomy"
# After running experiments:
git commit -m "Add ablation results and analysis notebook"
# After writeup:
git commit -m "Add findings to README, ethics statement"
```

---

## API Budget Estimate

| Phase | API Calls | Gemini Cost | GPTZero Cost |
|-------|-----------|-------------|-------------|
| Temperature sweep (4 variants × 5 topics × 5 runs) | 100 gen + 100 detect | Free tier | ~$5-10 |
| Ablation (5 tiers × ~4 variants × 5 topics × 5 runs) | 500 gen + 500 detect | Free tier | ~$25-50 |
| Composites (5 composites × 5 topics × 10 runs) | 250 gen + 250 detect | Free tier | ~$12-25 |
| Cross-model validation | 50 gen + 50 detect | May hit Pro pricing | ~$2-5 |
| **Total** | **~900 gen + 900 detect** | **~$0-5** | **~$45-90** |

Check GPTZero pricing before committing. If cost is prohibitive, reduce N per variant to 3, or use a free detector for initial exploration and GPTZero only for final validation.

---

## What Good Looks Like

The project is done when:

1. You can point to a specific prompt + parameter combination that drops detection below 15% **consistently** (mean across 5+ topics, N ≥ 5 runs each)
2. You know **which prompt dimensions matter most** and can rank them by effect size
3. You know whether results **transfer across detectors** or are GPTZero-specific
4. You have a **human baseline** showing how often GPTZero false-positives on real essays
5. Everything is reproducible — someone can clone the repo, add API keys, and replicate your results
6. The notebook tells a clear story from hypothesis → method → results → conclusions

---

## Common Pitfalls to Avoid

- **Don't iterate ad hoc.** It's tempting to just keep tweaking prompts based on vibes. The ablation structure is what makes this research instead of tinkering.
- **Don't test once and declare victory.** A single passing essay means nothing. You need statistical consistency.
- **Don't ignore human baselines.** If GPTZero flags 20% of human essays, your detector-evasion results need that context.
- **Don't hardcode GPTZero.** Their API could change or disappear. The abstract detector interface protects you.
- **Don't forget to log everything.** If you can't reproduce a result because you forgot what prompt you used, the run is wasted.
- **Don't commit API keys.** Use `.env` and `.env.example`. Double-check before pushing.
