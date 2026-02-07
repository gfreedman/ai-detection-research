"""
Central configuration for the AI detection research experiment.

This module is the single source of truth for all tuneable parameters, file paths,
API keys, and experimental constants.  Every other module imports from here rather
than hard-coding values, which keeps the experiment reproducible and easy to tweak
from one place.

Configuration values are loaded once at import time.  Environment variables (API
keys) are pulled from a ``.env`` file via ``python-dotenv``; everything else is a
plain Python constant.

Sections
--------
- **Paths**          -- project directories and output file locations.
- **API Keys**       -- credentials for Gemini, GPTZero, and ZeroGPT.
- **Gemini**         -- model identifiers and default generation parameters.
- **Detection**      -- detector endpoints and the pass/fail threshold.
- **Experiment**     -- run counts, rate-limit delays, and essay topics.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: read .env into os.environ before any env-var lookups.
# ---------------------------------------------------------------------------
load_dotenv()


# ── Paths ──────────────────────────────────────────────────────────────────────
# All paths are derived from PROJECT_ROOT so the project stays relocatable.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HUMAN_BASELINES_DIR = DATA_DIR / "human_baselines"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RAW_RESULTS_PATH = DATA_DIR / "raw_results.jsonl"
SUMMARY_CSV_PATH = DATA_DIR / "summary.csv"


# ── API Keys ───────────────────────────────────────────────────────────────────
# Loaded from environment variables.  An empty string means "not configured".
# Client classes validate at init time and raise ValueError if missing.
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GPTZERO_API_KEY = os.environ.get("GPTZERO_API_KEY", "")
ZEROGPT_API_KEY = os.environ.get("ZEROGPT_API_KEY", "")


# ── Gemini Generation ─────────────────────────────────────────────────────────
# Model identifiers and the Tier 6 (P6a) baseline generation parameters.
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_VALIDATION_MODEL = "gemini-1.5-pro"  # For cross-model validation

# Default generation parameters (Tier 6 baseline -- P6a)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40

# Word-count target range for essay generation.
# If a generation falls outside [WORD_COUNT_MIN, WORD_COUNT_MAX] the client
# re-generates up to MAX_REGENERATION_ATTEMPTS times, keeping the attempt
# whose word count is closest to TARGET_WORD_COUNT.
TARGET_WORD_COUNT = 500
WORD_COUNT_MIN = 450
WORD_COUNT_MAX = 550
MAX_REGENERATION_ATTEMPTS = 3

# Retry logic for transient API failures (applies to both Gemini and detectors).
# Backoff schedule: RETRY_BASE_DELAY_S * 2^retry  -->  2 s, 4 s, 8 s.
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 2.0


# ── Detection ──────────────────────────────────────────────────────────────────
# Detector API endpoints and the threshold that separates "passes as human"
# from "detected as AI".
# ---------------------------------------------------------------------------

GPTZERO_API_URL = "https://api.gptzero.me/v2/predict/text"
ZEROGPT_API_URL = "https://api.zerogpt.com/api/detect/detectText"

# An essay "passes" (i.e. reads as human-written) when its overall AI
# probability falls below this value.
DETECTION_PASS_THRESHOLD = 0.15  # completely_generated_prob < 0.15


# ── Experiment Parameters ──────────────────────────────────────────────────────
# Run counts and rate-limiting.
# ---------------------------------------------------------------------------

RUNS_PER_VARIANT = 5          # N per ablation cell
RUNS_PER_COMPOSITE = 10       # N per composite prompt test
RATE_LIMIT_DELAY_S = 3.0      # Seconds between consecutive API calls


# ── Topics (control variable) ─────────────────────────────────────────────────
# Fixed set of five essay prompts.  Every prompt variant is tested against every
# topic so that results are not confounded by topic choice.
# ---------------------------------------------------------------------------

ESSAY_TOPICS = [
    "Write a 500-word essay about how social media affects teen mental health.",
    "Write a 500-word essay arguing whether school uniforms should be required.",
    "Write a 500-word essay about a book that changed your perspective.",
    "Write a 500-word essay about climate change and what your generation can do.",
    "Write a 500-word essay about whether standardized testing is fair.",
]
