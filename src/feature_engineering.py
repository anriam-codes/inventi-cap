"""
feature_engineering.py

Feature engineering pipeline for Inventi.
Uses only historical information available before each prediction date.
"""

import numpy as np
import pandas as pd

from src.config import TARGET_COL, DATE_COL, FEATURE_COLS


def engineer_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Build all features required by the XGBoost model.

    If train_df is provided, store/item averages are calculated
    only from the training data to avoid target leakage.
    """

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(["store", "item", DATE_COL]).reset_index(drop=True)

    # Time features
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["week"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["day"] = df[DATE_COL].dt.day
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Lag features
    grp = df.groupby(["store", "item"])[TARGET_COL]

    for lag in [1, 7, 14, 28]:
        df[f"sales_lag_{lag}"] = grp.shift(lag)

    # Rolling mean features
    shifted = grp.shift(1)

    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = (
            shifted.rolling(window).mean().values
        )

    # Aggregate features
    # Use training data only when supplied to avoid validation leakage.
    if train_df is not None:
        train_df = train_df.copy()

        store_avg = (
            train_df.groupby("store")[TARGET_COL]
            .mean()
            .rename("store_avg_sales")
            .reset_index()
        )

        item_avg = (
            train_df.groupby("item")[TARGET_COL]
            .mean()
            .rename("item_avg_sales")
            .reset_index()
        )
    else:
        # Used when processing a standalone dataset.
        # For model training/validation, always pass train_df.
        store_avg = (
            df.groupby("store")[TARGET_COL]
            .mean()
            .rename("store_avg_sales")
            .reset_index()
        )

        item_avg = (
            df.groupby("item")[TARGET_COL]
            .mean()
            .rename("item_avg_sales")
            .reset_index()
        )

    df = df.merge(store_avg, on="store", how="left")
    df = df.merge(item_avg, on="item", how="left")

    return df


def get_model_ready(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select model features and remove rows with missing values.
    """

    available = [
        c for c in FEATURE_COLS
        if c in df.columns
    ]

    df_clean = df[available].dropna()

    return df_clean, available


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Calculate forecasting metrics.
    """

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = np.mean(np.abs(y_true - y_pred))

    rmse = np.sqrt(
        np.mean((y_true - y_pred) ** 2)
    )

    mask = y_true != 0

    if mask.any():
        mape = np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        ) * 100
    else:
        mape = 0.0

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
    }