"""Central configuration: paths, scoring weights, UI palette. Change values here only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"            # original datasets (never modified)
DATA_DIRS = [DATASET_DIR, ROOT / "data" / "jobs", ROOT / "data" / "resumes"]
PROCESSED_DIR = ROOT / "data" / "processed"   # processed copies live here
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
HISTORY_FILE = PROCESSED_DIR / "analysis_history.json"

# ---- Interpretable weighted score (weights are renormalised if a component is unavailable)
SCORE_WEIGHTS = {"skills": 0.50, "text": 0.25, "experience": 0.15, "education": 0.10}
PREFERRED_SKILL_WEIGHT = 0.5      # preferred skills count half as much as required ones
MIN_MATCHED_SKILLS = 3            # recommendations need >= this many matched skills (avoids 1-skill jobs scoring 100%)
TEXT_SIM_FULL_SCORE = 0.50        # TF-IDF cosine >= this value counts as 100% text similarity

# ---- Classification
TARGET_ROLES = ["Software Developer", "Data Analyst", "Data Scientist",
                "Machine Learning Engineer", "Web Developer", "Cyber Security Analyst"]
MIN_SAMPLES_PER_CLASS = 50        # classes below this are flagged as unreliable
RANDOM_STATE = 42
TEST_SIZE = 0.2

PALETTE = {"primary": "#6366F1", "secondary": "#8B5CF6", "background": "#F8FAFC",
           "dark": "#111827", "success": "#22C55E", "warning": "#F59E0B", "critical": "#EF4444"}
