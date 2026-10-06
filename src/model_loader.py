"""
model_loader.py

Loads the XGBoost model and saved validation predictions for Inventi.
"""

import os
import numpy as np
import streamlit as st

from src.config import MODEL_PATHS, PREDS_PATHS


# ── XGBoost model ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_xgb_model():
    """
    Load the trained XGBoost model.

    Uses the JSON model when available because it is more
    portable across XGBoost versions.
    """

    import xgboost as xgb

    json_path = os.path.join(
        os.path.dirname(MODEL_PATHS["XGBoost"]),
        "xgb_model.json"
    )

    joblib_path = MODEL_PATHS["XGBoost"]

    if os.path.exists(json_path):
        model = xgb.XGBRegressor()
        model.load_model(json_path)
        return model

    if os.path.exists(joblib_path):
        import joblib
        return joblib.load(joblib_path)

    return None


# ── Saved validation predictions ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_saved_predictions() -> dict:
    """
    Load saved XGBoost validation predictions and actual values.

    Returns:
        {
            "XGBoost": np.ndarray,
            "actual": np.ndarray
        }
    """

    import joblib

    predictions = {}

    for name, path in PREDS_PATHS.items():

        if os.path.exists(path):
            arr = joblib.load(path)
            predictions[name] = np.asarray(arr).flatten()

    return predictions


# ── Unified prediction interface ─────────────────────────────────────────────

def predict_tree(
    model_name: str,
    X
) -> np.ndarray:
    """
    Run prediction using the XGBoost model.

    Predictions are clipped to zero because demand cannot be negative.
    """

    if model_name != "XGBoost":
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Inventi uses XGBoost as its forecasting model."
        )

    model = load_xgb_model()

    if model is None:
        raise FileNotFoundError(
            "XGBoost model file not found."
        )

    predictions = model.predict(X)

    return np.clip(
        np.asarray(predictions),
        0,
        None
    )