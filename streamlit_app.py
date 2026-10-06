"""
Inventi — Predictive Inventory Planning & Replenishment (Streamlit App)
Entry point: runs the multi-page navigation shell.
"""

import streamlit as st

st.set_page_config(
    page_title="Inventi",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.title("📦 Inventi")
st.sidebar.caption("Predict → Assess → Replenish")

PAGES = {
    " Home":               "pages/home.py",
    " Forecast":           "pages/forecast.py",
    " Model Comparison":   "pages/model_comparison.py",
    " Inventory Planner":  "pages/inventory.py",
    " SHAP Explainer":     "pages/shap_explainer.py",
}

page = st.sidebar.radio("Navigate", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.info("Forecast model: XGBoost")

# ── Dynamic page loader ─────────────────────────────────────────────────────
import importlib.util, sys, os

def load_page(path: str):
    abs_path = os.path.join(os.path.dirname(__file__), path)
    spec = importlib.util.spec_from_file_location("page_module", abs_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

load_page(PAGES[page])
