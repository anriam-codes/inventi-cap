"""
pages/forecast.py

Inventi — Demand Forecasting Dashboard.
Uses the trained XGBoost model on the 2017 validation period.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import (
    BEST_MODEL,
    NUM_STORES,
    NUM_ITEMS,
    FEATURE_COLS,
    TARGET_COL,
    DATE_COL,
)

from src.data_loader import (
    load_featured_data,
    train_val_split,
)

from src.model_loader import (
    predict_tree,
    load_saved_predictions,
)


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("📈 Demand Forecast")

st.markdown(
    "View XGBoost demand predictions and compare them with actual sales."
)


# ── Controls ──────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)

with col1:
    store = st.selectbox(
        "Store",
        list(range(1, NUM_STORES + 1)),
    )

with col2:
    item = st.selectbox(
        "Item",
        list(range(1, NUM_ITEMS + 1)),
    )

with col3:
    n_show = st.slider(
        "Points to show",
        min_value=30,
        max_value=365,
        value=150,
    )


# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner("Loading forecasting data..."):
    df = load_featured_data()


# ── Selected store/item ────────────────────────────────────────────────────────

subset = df[
    (df["store"] == store)
    & (df["item"] == item)
].copy()

subset = subset.sort_values(DATE_COL)


if subset.empty:

    st.error(
        "No data found for the selected store and item."
    )

    st.stop()


# ── Validation period ─────────────────────────────────────────────────────────

_, val_df = train_val_split(subset)

val_df = val_df.dropna(
    subset=FEATURE_COLS + [TARGET_COL]
).copy()


if val_df.empty:

    st.error(
        "No valid validation data available for this store/item."
    )

    st.stop()


X_val = val_df[FEATURE_COLS]
y_val = val_df[TARGET_COL].to_numpy()


# ── Predictions ───────────────────────────────────────────────────────────────

with st.spinner("Generating XGBoost predictions..."):

    predictions = predict_tree(
        BEST_MODEL,
        X_val,
    )


predictions = np.asarray(
    predictions
).flatten()


# ── Metrics for selected item ─────────────────────────────────────────────────

mae = np.mean(
    np.abs(
        y_val - predictions
    )
)

rmse = np.sqrt(
    np.mean(
        (y_val - predictions) ** 2
    )
)

non_zero = y_val != 0

if non_zero.any():

    mape = (
        np.mean(
            np.abs(
                (
                    y_val[non_zero]
                    - predictions[non_zero]
                )
                / y_val[non_zero]
            )
        )
        * 100
    )

else:

    mape = 0.0


# ── Metrics ───────────────────────────────────────────────────────────────────

st.subheader(
    f"Store {store} · Item {item}"
)

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "MAE",
    f"{mae:.2f}",
)

metric2.metric(
    "RMSE",
    f"{rmse:.2f}",
)

metric3.metric(
    "MAPE",
    f"{mape:.2f}%",
)


# ── Forecast chart ────────────────────────────────────────────────────────────

display_n = min(
    n_show,
    len(val_df),
)

display_df = val_df.tail(
    display_n
).copy()

display_predictions = predictions[-display_n:]


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=display_df[DATE_COL],
        y=display_df[TARGET_COL],
        mode="lines",
        name="Actual Sales",
    )
)

fig.add_trace(
    go.Scatter(
        x=display_df[DATE_COL],
        y=display_predictions,
        mode="lines",
        name="XGBoost Forecast",
    )
)

fig.update_layout(
    title=(
        f"Demand Forecast — "
        f"Store {store} · Item {item}"
    ),
    xaxis_title="Date",
    yaxis_title="Sales",
    hovermode="x unified",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ── Forecast table ────────────────────────────────────────────────────────────

st.subheader("Forecast Details")

forecast_table = display_df[
    [DATE_COL, TARGET_COL]
].copy()

forecast_table["forecast"] = display_predictions

forecast_table["error"] = (
    forecast_table[TARGET_COL]
    - forecast_table["forecast"]
)

forecast_table["absolute_error"] = (
    forecast_table["error"]
    .abs()
)

forecast_table = forecast_table.rename(
    columns={
        DATE_COL: "Date",
        TARGET_COL: "Actual Sales",
        "forecast": "Forecast",
        "error": "Forecast Error",
        "absolute_error": "Absolute Error",
    }
)

st.dataframe(
    forecast_table,
    use_container_width=True,
    hide_index=True,
)


# ── Download ──────────────────────────────────────────────────────────────────

csv_data = forecast_table.to_csv(
    index=False
)

st.download_button(
    "⬇️ Download Forecast CSV",
    data=csv_data,
    file_name=(
        f"forecast_store{store}"
        f"_item{item}.csv"
    ),
    mime="text/csv",
)


# ── Model information ─────────────────────────────────────────────────────────

with st.expander("ℹ️ Model Information"):

    st.write(
        "**Model:** XGBoost"
    )

    st.write(
        "**Validation period:** 2017"
    )

    st.write(
        f"**Features:** {len(FEATURE_COLS)}"
    )

    st.write(
        "The model uses historical sales, "
        "lag features, rolling averages, "
        "time features, and store/item aggregates."
    )