"""Paths and shared constants used across the project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

OBSERVATION_FREQUENCY_DAYS = {
    "1D": 1,
    "1M": 30, "Monthly": 30, "mensual": 30, "1 month": 30, "M": 30,
    "2M": 60,
    "3M": 90, "Quarterly": 90, "trimestral": 90, "Q": 90, "3 months": 90,
    "6M": 180, "Semi-Annual": 180, "semestral": 180, "6 months": 180,
    "1Y": 365, "12M": 365, "Y": 365, "Annual": 365, "anual": 365,
}

CATEGORICAL_COLUMNS = ["product_type", "basket_type", "counterparty", "trader_id"]

LEAKAGE_OR_ID_COLUMNS = ["rfq_id", "start_date", "end_date", "avg_duration_months", "executed"]
