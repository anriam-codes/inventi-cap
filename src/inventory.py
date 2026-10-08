"""
inventory.py

Inventory decision engine for Inventi.

Flow:
Forecast → Forecast Error → Demand Variability
→ Safety Stock → Reorder Point → Order Quantity → Risk
"""

import numpy as np
import pandas as pd

from src.config import INV_DEFAULTS


def compute_eoq(
    annual_demand: float,
    ordering_cost: float = INV_DEFAULTS["ordering_cost"],
    holding_cost: float = INV_DEFAULTS["holding_cost_per_unit"],
) -> float:
    """
    Economic Order Quantity.

    EOQ = sqrt(2DS / H)
    """

    if annual_demand <= 0 or holding_cost <= 0:
        return 0.0

    return float(
        np.sqrt(
            (2 * annual_demand * ordering_cost)
            / holding_cost
        )
    )


def compute_safety_stock(
    demand_std: float,
    lead_time_days: int = INV_DEFAULTS["lead_time_days"],
    z: float = INV_DEFAULTS["service_level_z"],
) -> float:
    """
    Safety Stock based on demand variability.

    SS = Z × σ × sqrt(L)
    """

    if demand_std <= 0 or lead_time_days <= 0:
        return 0.0

    return float(
        z * demand_std * np.sqrt(lead_time_days)
    )


def compute_rop(
    avg_daily_demand: float,
    lead_time_days: int = INV_DEFAULTS["lead_time_days"],
    safety_stock: float = 0.0,
) -> float:
    """
    Reorder Point.

    ROP = Lead-Time Demand + Safety Stock
    """

    return float(
        (avg_daily_demand * lead_time_days)
        + safety_stock
    )


def compute_order_quantity(
    current_stock: float,
    reorder_point: float,
) -> float:
    """
    Recommended quantity to order.

    Order Quantity = max(0, ROP - Current Stock)
    """

    return float(
        max(0, reorder_point - current_stock)
    )


def classify_inventory_risk(
    current_stock: float,
    reorder_point: float,
    avg_daily_demand: float,
) -> str:
    """
    Classify current inventory condition.
    """

    if avg_daily_demand <= 0:
        return "No Demand"

    if current_stock <= 0:
        return "Stockout"

    if current_stock < reorder_point:
        return "Reorder"

    if current_stock > reorder_point * 2:
        return "Overstock"

    return "Healthy"


def compute_inventory_plan(
    forecast_df: pd.DataFrame,
    current_stock: float | None = None,
    ordering_cost: float = INV_DEFAULTS["ordering_cost"],
    holding_cost: float = INV_DEFAULTS["holding_cost_per_unit"],
    lead_time_days: int = INV_DEFAULTS["lead_time_days"],
    z: float = INV_DEFAULTS["service_level_z"],
) -> pd.DataFrame:
    """
    Convert demand forecasts into inventory decisions.

    Required columns:
        date
        store
        item
        y_pred

    Optional columns:
        error

    If current_stock is not supplied, the engine calculates
    the inventory requirements without making a stock decision.
    """

    required = {
        "date",
        "store",
        "item",
        "y_pred",
    }

    missing = required - set(forecast_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = forecast_df.copy()

    df["y_pred"] = pd.to_numeric(
        df["y_pred"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["y_pred"]
    )

    # ---------------------------------------------------------
    # Demand statistics
    # ---------------------------------------------------------

    daily_item = (
        df.groupby(
            ["date", "item"]
        )["y_pred"]
        .sum()
        .reset_index()
    )

    item_stats = (
        daily_item
        .groupby("item")
        .agg(
            annual_forecast_demand=("y_pred", "sum"),
            avg_daily_demand=("y_pred", "mean"),
            forecast_std=("y_pred", "std"),
            days_of_data=("y_pred", "count"),
        )
        .reset_index()
    )

    item_stats["forecast_std"] = (
        item_stats["forecast_std"]
        .fillna(0)
    )

    # ---------------------------------------------------------
    # Forecast-error uncertainty
    # ---------------------------------------------------------

    if "error" in df.columns:

        error_stats = (
            df.groupby("item")["error"]
            .std()
            .fillna(0)
            .rename("error_std")
            .reset_index()
        )

        item_stats = item_stats.merge(
            error_stats,
            on="item",
            how="left",
        )

    else:
        item_stats["error_std"] = (
            item_stats["forecast_std"]
        )

    item_stats["error_std"] = (
        item_stats["error_std"]
        .fillna(0)
    )

    # ---------------------------------------------------------
    # Variability
    # ---------------------------------------------------------

    item_stats["cv"] = np.where(
        item_stats["avg_daily_demand"] > 0,
        item_stats["error_std"]
        / item_stats["avg_daily_demand"],
        0,
    )

    # ---------------------------------------------------------
    # EOQ
    # ---------------------------------------------------------

    item_stats["EOQ"] = (
        item_stats["annual_forecast_demand"]
        .apply(
            lambda d: compute_eoq(
                d,
                ordering_cost,
                holding_cost,
            )
        )
    )

    # ---------------------------------------------------------
    # Safety Stock
    # ---------------------------------------------------------

    item_stats["safety_stock"] = (
        item_stats["error_std"]
        .apply(
            lambda s: compute_safety_stock(
                s,
                lead_time_days,
                z,
            )
        )
    )

    # ---------------------------------------------------------
    # Reorder Point
    # ---------------------------------------------------------

    item_stats["ROP"] = item_stats.apply(
        lambda row: compute_rop(
            row["avg_daily_demand"],
            lead_time_days,
            row["safety_stock"],
        ),
        axis=1,
    )

    # ---------------------------------------------------------
    # Current stock + recommendation
    # ---------------------------------------------------------

    if current_stock is not None:

        item_stats["current_stock"] = float(
            current_stock
        )

        item_stats["recommended_order_qty"] = (
            item_stats.apply(
                lambda row: compute_order_quantity(
                    row["current_stock"],
                    row["ROP"],
                ),
                axis=1,
            )
        )

        item_stats["risk_status"] = (
            item_stats.apply(
                lambda row: classify_inventory_risk(
                    row["current_stock"],
                    row["ROP"],
                    row["avg_daily_demand"],
                ),
                axis=1,
            )
        )

    # ---------------------------------------------------------
    # Clean output
    # ---------------------------------------------------------

    numeric_columns = [
        "annual_forecast_demand",
        "avg_daily_demand",
        "forecast_std",
        "error_std",
        "cv",
        "EOQ",
        "safety_stock",
        "ROP",
    ]

    if current_stock is not None:
        numeric_columns.extend([
            "current_stock",
            "recommended_order_qty",
        ])

    for column in numeric_columns:
        item_stats[column] = (
            item_stats[column]
            .round(2)
        )

    return (
        item_stats
        .sort_values("item")
        .reset_index(drop=True)
    )