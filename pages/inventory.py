"""
pages/inventory.py

Inventi — Inventory Planning & Replenishment Dashboard.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    INV_DEFAULTS,
    NUM_ITEMS,
)

from src.data_loader import (
    load_featured_data,
    train_val_split,
)

from src.model_loader import (
    predict_tree,
)

from src.inventory import (
    compute_inventory_plan,
    compute_order_quantity,
    classify_inventory_risk,
)

from src.plots import (
    plot_eoq_by_item,
    plot_rop_safety,
    plot_inventory_risk,
)


# ── Page title ────────────────────────────────────────────────────────────────

st.title("📦 Inventory Planner")

st.markdown(
    """
Convert **XGBoost demand forecasts** into inventory decisions using
forecast uncertainty, safety stock, reorder points and recommended
replenishment quantities.
"""
)


# ── Parameters ────────────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚙️ Inventory Parameters")

ordering_cost = st.sidebar.number_input(
    "Ordering Cost",
    min_value=1.0,
    value=float(
        INV_DEFAULTS["ordering_cost"]
    ),
)

holding_cost = st.sidebar.number_input(
    "Holding Cost / Unit / Year",
    min_value=0.1,
    value=float(
        INV_DEFAULTS["holding_cost_per_unit"]
    ),
)

lead_time = st.sidebar.number_input(
    "Lead Time (days)",
    min_value=1,
    max_value=90,
    value=int(
        INV_DEFAULTS["lead_time_days"]
    ),
)

service_level = st.sidebar.selectbox(
    "Service Level",
    [
        "90% (Z=1.28)",
        "95% (Z=1.65)",
        "99% (Z=2.33)",
    ],
    index=1,
)

z_map = {
    "90% (Z=1.28)": 1.28,
    "95% (Z=1.65)": 1.65,
    "99% (Z=2.33)": 2.33,
}

z_score = z_map[service_level]


# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner("Loading demand data..."):

    df = load_featured_data()

    _, val_df = train_val_split(df)


# Keep only rows that contain all model features.

val_df = val_df.dropna(
    subset=[
        "store",
        "item",
        "sales",
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
).copy()


# ── Generate XGBoost predictions ──────────────────────────────────────────────

feature_cols = [
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


with st.spinner("Generating XGBoost demand predictions..."):

    predictions = predict_tree(
        "XGBoost",
        val_df[feature_cols],
    )


val_df["y_pred"] = np.asarray(
    predictions
).flatten()

val_df["y_pred"] = val_df[
    "y_pred"
].clip(lower=0)


# Forecast error is essential for inventory uncertainty.

val_df["error"] = (
    val_df["sales"]
    - val_df["y_pred"]
)


# ── Status ────────────────────────────────────────────────────────────────────

st.success(
    f"Using XGBoost forecasts on {len(val_df):,} validation observations."
)


# ── Inventory plan ────────────────────────────────────────────────────────────

with st.spinner("Calculating inventory requirements..."):

    inv_df = compute_inventory_plan(
        forecast_df=val_df,
        ordering_cost=ordering_cost,
        holding_cost=holding_cost,
        lead_time_days=int(lead_time),
        z=z_score,
    )


# ── KPI section ───────────────────────────────────────────────────────────────

st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Items Analyzed",
        len(inv_df),
    )

with c2:
    st.metric(
        "Avg Daily Demand",
        f"{inv_df['avg_daily_demand'].mean():.1f}",
    )

with c3:
    st.metric(
        "Avg Safety Stock",
        f"{inv_df['safety_stock'].mean():.1f}",
    )

with c4:
    st.metric(
        "Avg Reorder Point",
        f"{inv_df['ROP'].mean():.1f}",
    )


# ── Charts ────────────────────────────────────────────────────────────────────

st.markdown("---")

st.subheader("Inventory Overview")

top_n = st.slider(
    "Items to display",
    min_value=5,
    max_value=50,
    value=20,
)


col1, col2 = st.columns(2)

with col1:

    st.plotly_chart(
        plot_eoq_by_item(
            inv_df,
            top_n,
        ),
        use_container_width=True,
    )

with col2:

    st.plotly_chart(
        plot_rop_safety(
            inv_df,
            top_n,
        ),
        use_container_width=True,
    )


# ── Inventory table ───────────────────────────────────────────────────────────

st.markdown("---")

st.subheader("📋 Inventory Requirements")

display_cols = [
    "item",
    "annual_forecast_demand",
    "avg_daily_demand",
    "error_std",
    "cv",
    "EOQ",
    "safety_stock",
    "ROP",
]

display_names = {
    "item": "Item",
    "annual_forecast_demand": "Annual Forecast Demand",
    "avg_daily_demand": "Avg Daily Demand",
    "error_std": "Forecast Error Std",
    "cv": "CV",
    "EOQ": "EOQ",
    "safety_stock": "Safety Stock",
    "ROP": "Reorder Point",
}

display_df = (
    inv_df[display_cols]
    .rename(columns=display_names)
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# ── Replenishment decision ────────────────────────────────────────────────────

st.markdown("---")

st.subheader("🎯 Replenishment Decision")

col1, col2 = st.columns(2)

with col1:

    selected_item = st.selectbox(
        "Select Item",
        list(range(1, NUM_ITEMS + 1)),
    )

with col2:

    current_stock = st.number_input(
        "Current Stock (units)",
        min_value=0.0,
        value=100.0,
        step=1.0,
    )


selected_row = inv_df[
    inv_df["item"] == selected_item
]


if not selected_row.empty:

    row = selected_row.iloc[0]

    recommended_order = compute_order_quantity(
        current_stock=current_stock,
        reorder_point=row["ROP"],
    )

    risk = classify_inventory_risk(
        current_stock=current_stock,
        reorder_point=row["ROP"],
        avg_daily_demand=row["avg_daily_demand"],
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Forecast / Day",
            f"{row['avg_daily_demand']:.1f}",
        )

    with col2:
        st.metric(
            "Safety Stock",
            f"{row['safety_stock']:.1f}",
        )

    with col3:
        st.metric(
            "Reorder Point",
            f"{row['ROP']:.1f}",
        )

    with col4:
        st.metric(
            "Recommended Order",
            f"{recommended_order:.0f}",
        )

    st.markdown("### Inventory Status")

    if risk == "Stockout":
        st.error(
            "🔴 STOCKOUT — immediate replenishment required."
        )

    elif risk == "Reorder":
        st.warning(
            "🟠 REORDER — current stock is below the reorder point."
        )

    elif risk == "Overstock":
        st.warning(
            "🟡 OVERSTOCK — inventory is significantly above the reorder point."
        )

    else:
        st.success(
            "🟢 HEALTHY — current inventory is within the desired range."
        )

    st.markdown(
        f"""
**Decision explanation**

- Current stock: **{current_stock:.0f} units**
- Reorder point: **{row['ROP']:.0f} units**
- Safety stock: **{row['safety_stock']:.0f} units**
- Lead time: **{lead_time} days**
- Recommended order: **{recommended_order:.0f} units**
"""
    )


# ── Download ──────────────────────────────────────────────────────────────────

st.markdown("---")

csv = display_df.to_csv(
    index=False
)

st.download_button(
    "⬇️ Download Inventory Plan",
    data=csv,
    file_name="inventi_inventory_plan.csv",
    mime="text/csv",
)


# ── Formula reference ─────────────────────────────────────────────────────────

with st.expander("📖 Inventory Formula Reference"):

    st.markdown(
        """
**EOQ**

Economic Order Quantity determines an efficient order size
based on annual demand, ordering cost and holding cost.

**Safety Stock**

Safety stock protects against demand uncertainty derived from
forecast error.

**Reorder Point**

The reorder point combines expected demand during supplier lead
time with safety stock.

**Recommended Order**

Order quantity is the amount required to bring current stock
back to the reorder point.
"""
    )