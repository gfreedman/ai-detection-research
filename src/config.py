"""Central configuration for the AI detection research experiment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HUMAN_BASELINES_DIR = DATA_DIR / "human_baselines"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RAW_RESULTS_PATH = DATA_DIR / "raw_results.jsonl"
SUMMARY_CSV_PATH = DATA_DIR / "summary.csv"

# ── API Keys ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GPTZERO_API_KEY = os.environ.get("GPTZERO_API_KEY", "")

# ── Gemini Generation ─────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_VALIDATION_MODEL = "gemini-1.5-pro"  # For cross-model validation

# Default generation parameters (Tier 6 baseline — P6a)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40

# Word count target range
TARGET_WORD_COUNT = 500
WORD_COUNT_MIN = 450
WORD_COUNT_MAX = 550
MAX_REGENERATION_ATTEMPTS = 3

# Retry logic for API calls
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 2.0  # Exponential backoff: 2s, 4s, 8s

# ── Detection ──────────────────────────────────────────────────────────────────
# GPTZero API endpoint
GPTZERO_API_URL = "https://api.gptzero.me/v2/predict/text"

# Threshold: below this = "passes" as human-written
DETECTION_PASS_THRESHOLD = 0.15  # completely_generated_prob < 0.15

# ── Experiment Parameters ──────────────────────────────────────────────────────
RUNS_PER_VARIANT = 5          # N per ablation cell
RUNS_PER_COMPOSITE = 10       # N per composite prompt test
RATE_LIMIT_DELAY_S = 3.0      # Seconds between API calls

# ── Topics (control variable) ─────────────────────────────────────────────────
ESSAY_TOPICS = [
    "Write a 500-word essay about how social media affects teen mental health.",
    "Write a 500-word essay arguing whether school uniforms should be required.",
    "Write a 500-word essay about a book that changed your perspective.",
    "Write a 500-word essay about climate change and what your generation can do.",
    "Write a 500-word essay about whether standardized testing is fair.",
]
