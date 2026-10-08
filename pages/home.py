import streamlit as st

st.title("📦 Inventi")

st.subheader(
    "Predictive Inventory Planning & Replenishment System"
)

st.write(
    "Inventi combines machine-learning demand forecasting "
    "with inventory planning to convert predicted demand "
    "into actionable replenishment decisions."
)

st.markdown("---")

st.subheader("System Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Forecast Model", "XGBoost")

with col2:
    st.metric("Features", "17")

with col3:
    st.metric("Validation Period", "2017")

with col4:
    st.metric("Decision Layer", "Inventory")

st.markdown("---")

st.subheader("Inventi Decision Pipeline")

st.markdown(
    """
    **Historical Sales**
    → **Feature Engineering**
    → **XGBoost Forecast**
    → **Forecast Error**
    → **Safety Stock**
    → **Reorder Point**
    → **Recommended Order**
    → **Inventory Risk**
    """
)

st.markdown("---")

st.subheader("XGBoost Validation Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "6.08")

with col2:
    st.metric("RMSE", "7.91")

with col3:
    st.metric("MAPE", "12.44%")
                
st.caption("Validation performed on 2017 demand data.")

st.markdown("---")

st.info(
    "Use the Forecast page to inspect demand predictions "
    "and the Inventory Planner page to generate "
    "replenishment decisions."
)