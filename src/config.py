"""
config.py — Central configuration for Inventi.
"""

import os


# ── Repository paths ──────────────────────────────────────────────────────────

def _find_repo_root() -> str:
    """Find the project root from the location of this file."""

    candidate = os.path.dirname(os.path.abspath(__file__))

    for _ in range(5):
        candidate = os.path.dirname(candidate)

        if os.path.exists(
            os.path.join(candidate, "requirements.txt")
        ):
            return candidate

    return os.getcwd()


ROOT_DIR = _find_repo_root()

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

RAW_DATA_PATH = os.path.join(
    DATA_DIR,
    "raw",
    "train.csv"
)

FEATURED_DATA_PATH = os.path.join(
    DATA_DIR,
    "processed",
    "featured_data.csv"
)


# ── XGBoost model ─────────────────────────────────────────────────────────────

MODEL_PATHS = {
    "XGBoost": os.path.join(
        MODELS_DIR,
        "xgb_model.joblib"
    ),
}

PREDS_PATHS = {
    "XGBoost": os.path.join(
        MODELS_DIR,
        "xgb_preds.joblib"
    ),
    "actual": os.path.join(
        MODELS_DIR,
        "y_val_actual.joblib"
    ),
}


BEST_MODEL = "XGBoost"


# ── Feature configuration ────────────────────────────────────────────────────

FEATURE_COLS = [
    "store",
    "item",

    "year",
    "month",
    "week",
    "day",
    "dayofweek",
    "is_weekend",

    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",

    "store_avg_sales",
    "item_avg_sales",
]


TARGET_COL = "sales"
DATE_COL = "date"

NUM_STORES = 10
NUM_ITEMS = 50


# ── Inventory defaults ───────────────────────────────────────────────────────

INV_DEFAULTS = {
    "ordering_cost": 50.0,
    "holding_cost_per_unit": 2.0,
    "lead_time_days": 7,
    "service_level_z": 1.65,
}