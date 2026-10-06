# Inventi — Predictive Inventory Planning & Replenishment

A lightweight, explainable decision-support system that answers:

> **What is likely to be demanded, what is the current inventory risk, and how much should be reordered?**

```
Historical Sales → Feature Engineering → XGBoost Demand Forecast
        → Forecast Error → Inventory Risk → Safety Stock + Reorder Point
        → Recommended Order Quantity → Dashboard + Explanation
```

**Predict → Assess → Replenish.**

> Status: under active development (rebuilt from the open-source DemandSense project).

---

## Method

| Stage | Approach |
|-------|----------|
| Forecasting | XGBoost on calendar, lag, rolling-mean and aggregate features. Chronological split: train 2013–2016, validate 2017 |
| Metrics | MAE, RMSE, MAPE |
| Safety stock | `SS = Z · σ_d · √L` (σ_d = demand variability / forecast error, L = lead time) |
| Reorder point | `ROP = lead-time demand + SS` |
| Order quantity | `max(0, ROP − current stock)` |
| Risk status | Rule-based: STOCKOUT RISK / HEALTHY / OVERSTOCK |
| Explainability | SHAP (secondary) |

### Important data limitation
The Kaggle Store Item Demand dataset contains **sales history only**. It has no real on-hand inventory or supplier lead times.
Current stock, lead time, service level, ordering cost and holding cost are therefore **user-entered parameters** in the app.
Any default stock value is a simulated/derived demonstration value, not real inventory data.

## Dataset
Kaggle — [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only):
913,000 daily records, 10 stores, 50 items, 2013–2017 (`date, store, item, sales`).

## Features
| Category | Features |
|----------|----------|
| Time | year, month, week, day, dayofweek, is_weekend |
| Lag | sales_lag_1, _7, _14, _28 |
| Rolling | rolling_mean_7, _14, _28 (shifted by one day to avoid leakage) |
| Aggregate | store_avg_sales, item_avg_sales |

The trained model also receives `store` and `item` as numeric inputs (17 features in total).

## Running locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Requires `data/raw/train.csv` and the model files in `models/`. If `data/processed/featured_data.csv` is absent,
features are built from `train.csv` on first load (cached afterwards).

## Contribution
Integration of machine-learning demand forecasting, inventory-risk assessment, and explainable replenishment
decisions into a lightweight predictive inventory planning system. We do not claim a new forecasting or inventory algorithm.

## References
1. Alsaif (2026), *A Systematic Benchmarking Pipeline for Demand Forecasting and Inventory-Aware Supply Chain Decision Support*, Frontiers in AI.
2. Gao & Sarvghadi (2026), *From Prediction to Diagnostic Support: A Data-Driven System for Retail Demand Forecasting and Inventory Risk Assessment*, Information 17(9):909.
3. *A Hybrid Learning Framework for Forecasting Uncertainty and Adaptive Inventory Planning in Retail Supply Chains* (2026).
4. Nguyen, Dang & Le (2025), *Inventory Demand Forecasting Using XGBoost and LightGBM Algorithms: A Case Study of Grupo Bimbo*.
5. Schett et al. (2026), *Integrated demand forecasting and reinforcement learning for order point optimization in inventory planning*, Int. J. Production Economics.
