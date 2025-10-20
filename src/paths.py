"""Utility helpers to work with project paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPARK_DATA_PATH = DATA_DIR / "props_model_sparked.parquet"
SQL_DATA_PATH = DATA_DIR / "props_model.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
DUCKDB_SUMMARY_PATH = REPORTS_DIR / "duckdb_summary.json"
FEATURE_STATS_PATH = REPORTS_DIR / "feature_stats.json"
SAMPLES_DIR = PROJECT_ROOT / "samples"

for directory in [DATA_DIR, RAW_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
