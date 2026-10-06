"""pages/home.py — Landing / overview page."""

import streamlit as st
from src.config import MODEL_METRICS, BEST_MODEL


def render():
    st.title("Inventi — Predictive Inventory Planning & Replenishment")
    st.markdown(
        "**Predict demand, assess inventory risk, and recommend how much to reorder.**"
    )
    st.markdown("---")

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model",    BEST_MODEL)
    c2.metric("Best RMSE",     f"{MODEL_METRICS[BEST_MODEL]['RMSE']:.4f}")
    c3.metric("Best MAPE",     f"{MODEL_METRICS[BEST_MODEL]['MAPE']:.2f}%")
    c4.metric("Validation Period","2017")

    st.markdown("---")

    # ── Project overview ─────────────────────────────────────────────────────
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.subheader("🗂️ Project Pipeline")
        st.markdown("""
        ```
        Historical Sales (train.csv)
             │
             ▼
        Feature Engineering
        (Calendar · Lag · Rolling · Aggregate)
             │
             ▼
        XGBoost Demand Forecast
        RMSE: 7.91   MAPE: 12.44%
             │
             ▼
        Forecast Error → Inventory Risk
             │
             ▼
        Safety Stock · Reorder Point
             │
             ▼
        Recommended Order Quantity
        ```
        """)

    with col2:
        st.subheader("📂 Dataset Facts")
        st.markdown("""
        | Attribute | Value |
        |-----------|-------|
        | Rows | 913,000 |
        | Columns | 4 |
        | Stores | 10 |
        | Items | 50 |
        | Date Range | 2013–2017 |
        | Missing Values | None |
        | Engineered Features | 19 |
        """)

    st.markdown("---")

    # ── Navigation guide ─────────────────────────────────────────────────────
    st.subheader("🚀 What you can do here")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.info("**🤖 Forecast**\nSelect a store and item to see demand predictions.")
    with g2:
        st.info("**🏭 Inventory Planner**\nTurn forecasts into safety stock, reorder point and order quantity.")
    with g3:
        st.info("**🔍 SHAP Explainer**\nSee why the model predicted a given demand.")

    st.markdown("---")
    st.caption("Built with Streamlit · XGBoost · SHAP · Plotly")


render()
