"""
plots.py

Reusable Plotly charts for Inventi.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_eoq_by_item(
    inv_df: pd.DataFrame,
    top_n: int = 20
) -> go.Figure:

    data = (
        inv_df
        .nlargest(top_n, "EOQ")
        .sort_values("EOQ")
    )

    fig = px.bar(
        data,
        x="EOQ",
        y="item",
        orientation="h",
        title=f"EOQ — Top {top_n} Items",
        labels={
            "item": "Item",
            "EOQ": "Economic Order Quantity",
        },
        template="plotly_dark",
    )

    return fig


def plot_rop_safety(
    inv_df: pd.DataFrame,
    top_n: int = 20
) -> go.Figure:

    data = (
        inv_df
        .nlargest(top_n, "ROP")
        .sort_values("ROP")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=data["item"].astype(str),
            y=data["ROP"],
            name="Reorder Point",
        )
    )

    fig.add_trace(
        go.Bar(
            x=data["item"].astype(str),
            y=data["safety_stock"],
            name="Safety Stock",
        )
    )

    fig.update_layout(
        barmode="group",
        title=f"Reorder Point & Safety Stock — Top {top_n} Items",
        xaxis_title="Item",
        yaxis_title="Units",
        template="plotly_dark",
    )

    return fig


def plot_inventory_risk(
    inv_df: pd.DataFrame
) -> go.Figure:

    if "risk_status" not in inv_df.columns:
        return go.Figure()

    counts = (
        inv_df["risk_status"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "risk_status",
        "count",
    ]

    fig = px.bar(
        counts,
        x="risk_status",
        y="count",
        title="Inventory Risk Status",
        labels={
            "risk_status": "Risk Status",
            "count": "Number of Items",
        },
        template="plotly_dark",
    )

    return fig