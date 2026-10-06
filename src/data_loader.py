"""
data_loader.py — Data loading and feature preparation for Inventi.

The feature set matches the XGBoost model:
store, item, time features, lag features, rolling features,
store_avg_sales, and item_avg_sales.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    RAW_DATA_PATH,
    FEATURED_DATA_PATH,
    DATE_COL,
    TARGET_COL,
    FEATURE_COLS,
)


# ── Raw data ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    if not os.path.exists(RAW_DATA_PATH):
        st.error(
            f"Raw data not found: `{RAW_DATA_PATH}`\n\n"
            "Ensure `data/raw/train.csv` exists."
        )
        st.stop()

    df = pd.read_csv(
        RAW_DATA_PATH,
        parse_dates=[DATE_COL]
    )

    df.sort_values(DATE_COL, inplace=True)

    return df


# ── Feature data ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_featured_data() -> pd.DataFrame:
    """
    Load pre-built features if available.

    Otherwise build them from the raw dataset.
    """

    if os.path.exists(FEATURED_DATA_PATH):
        df = pd.read_csv(
            FEATURED_DATA_PATH,
            parse_dates=[DATE_COL]
        )

        df.sort_values(DATE_COL, inplace=True)

        return df

    st.info(
        "Building features from train.csv. "
        "This may take some time on the first run.",
        icon="ℹ️",
    )

    raw = load_raw_data()

    return _build_features(raw)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features while preventing aggregate leakage.

    Store/item averages are calculated from the training period only
    and then applied to the complete dataset.
    """

    df = df.copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df = df.sort_values(
        ["store", "item", DATE_COL]
    ).reset_index(drop=True)

    # ── Time features ────────────────────────────────────────────────────────

    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["week"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["day"] = df[DATE_COL].dt.day
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # ── Lag features ─────────────────────────────────────────────────────────

    grp = df.groupby(["store", "item"])[TARGET_COL]

    for lag in [1, 7, 14, 28]:
        df[f"sales_lag_{lag}"] = grp.shift(lag)

    # ── Rolling mean features ────────────────────────────────────────────────

    shifted = grp.shift(1)

    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = (
            shifted.rolling(window).mean().values
        )

    # ── Training-only aggregate features ─────────────────────────────────────

    train_mask = df[DATE_COL] < "2017-01-01"

    train_data = df.loc[train_mask]

    store_avg = (
        train_data.groupby("store")[TARGET_COL]
        .mean()
        .rename("store_avg_sales")
        .reset_index()
    )

    item_avg = (
        train_data.groupby("item")[TARGET_COL]
        .mean()
        .rename("item_avg_sales")
        .reset_index()
    )

    df = df.merge(
        store_avg,
        on="store",
        how="left"
    )

    df = df.merge(
        item_avg,
        on="item",
        how="left"
    )

    return df


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return _build_features(df)


# ── Train / Validation split ─────────────────────────────────────────────────

def train_val_split(
    df: pd.DataFrame,
    val_start: str = "2017-01-01"
):
    train = df[df[DATE_COL] < val_start].copy()

    validation = df[df[DATE_COL] >= val_start].copy()

    return train, validation


# ── XGBoost data ──────────────────────────────────────────────────────────────

def get_X_y(df: pd.DataFrame):
    available = [
        column
        for column in FEATURE_COLS
        if column in df.columns
    ]

    subset = df[
        available + [TARGET_COL]
    ].dropna()

    return (
        subset[available],
        subset[TARGET_COL]
    )


# ── Upload validation ─────────────────────────────────────────────────────────

def validate_uploaded_csv(
    df: pd.DataFrame
) -> tuple[bool, str]:

    required = {
        DATE_COL,
        "store",
        "item"
    }

    missing = required - set(df.columns)

    if missing:
        return False, f"Missing required columns: {missing}"

    try:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    except Exception:
        return False, "Could not parse 'date' column as datetime."

    if df["store"].nunique() > 10:
        return False, "store values exceed the training range."

    if df["item"].nunique() > 50:
        return False, "item values exceed the training range."

    return True, "OK"