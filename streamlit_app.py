"""
Inventi — Predictive Inventory Planning & Replenishment

Main Streamlit application.
"""

import importlib.util
import os

import streamlit as st


# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Inventi",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("📦 Inventi")
st.sidebar.caption("Predict → Assess → Replenish")

PAGES = {
    "🏠 Home": "pages/home.py",
    "📈 Forecast": "pages/forecast.py",
    "📦 Inventory Planner": "pages/inventory.py",
    "🔍 SHAP Explainer": "pages/shap_explainer.py",
}

page = st.sidebar.radio(
    "Navigate",
    list(PAGES.keys()),
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Forecast Model\n\n"
    "**XGBoost**"
)


# ── Page loader ────────────────────────────────────────────────────────────────

def load_page(path: str):
    """Load and execute a Streamlit page."""

    abs_path = os.path.join(
        os.path.dirname(__file__),
        path,
    )

    if not os.path.exists(abs_path):
        st.error(f"Page not found: `{path}`")
        return

    spec = importlib.util.spec_from_file_location(
        "inventi_page",
        abs_path,
    )

    if spec is None or spec.loader is None:
        st.error(f"Could not load page: `{path}`")
        return

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)


load_page(PAGES[page])