"""Project configuration settings."""

from pathlib import Path


RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "diagnosis"
ID_COLUMNS = ["id"]


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
EXPLAINABILITY_DIR = OUTPUT_DIR / "explainability"
