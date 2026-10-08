"""
pages/shap_explainer.py

Explain individual XGBoost demand predictions using SHAP.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.config import NUM_STORES, NUM_ITEMS
from src.data_loader import load_featured_data, train_val_split
from src.model_loader import load_xgb_model


# ============================================================
# Helpers
# ============================================================

def get_model_features(model):
    """Get the exact features used when the XGBoost model was trained."""

    if hasattr(model, "feature_names_in_"):
        names = model.feature_names_in_
        if names is not None:
            return list(names)

    try:
        names = model.get_booster().feature_names
        if names:
            return list(names)
    except Exception:
        pass

    return None


def align_features(df, model_features):
    """Select model features in the exact training order."""

    missing = [f for f in model_features if f not in df.columns]

    if missing:
        st.error(
            f"Missing model features: {missing}"
        )
        return pd.DataFrame()

    X = df[model_features].copy()

    # Ensure numeric values
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    return X


def get_shap_values(model, X):
    """
    Compute SHAP values robustly across SHAP/XGBoost versions.
    """

    import shap

    # Method 1: TreeExplainer
    try:
        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            X,
            check_additivity=False
        )

        expected_value = explainer.expected_value

        # Some SHAP versions return a list
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Some versions return an Explanation-like object
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        shap_values = np.asarray(shap_values)

        if np.ndim(expected_value) > 0:
            expected_value = float(np.asarray(expected_value).flatten()[0])
        else:
            expected_value = float(expected_value)

        return shap_values, expected_value, "TreeExplainer"

    except Exception as first_error:

        # Method 2: shap.Explainer fallback
        try:
            explainer = shap.Explainer(model)

            explanation = explainer(X)

            shap_values = explanation.values

            if np.ndim(shap_values) == 3:
                shap_values = shap_values[:, :, 0]

            expected_value = explanation.base_values

            if np.ndim(expected_value) > 1:
                expected_value = expected_value[:, 0]

            expected_value = float(np.asarray(expected_value).flatten()[0])

            return shap_values, expected_value, "SHAP Explainer"

        except Exception as second_error:
            raise RuntimeError(
                f"TreeExplainer failed: {str(first_error)[:300]} | "
                f"SHAP Explainer failed: {str(second_error)[:300]}"
            )


# ============================================================
# Main renderer
# ============================================================

def render():

    st.title("🔍 Demand Explainer (SHAP)")

    st.markdown(
        "**Why did the model predict this demand value?** "
        "SHAP breaks down every prediction into the contribution "
        "of each feature."
    )

    # --------------------------------------------------------
    # SHAP availability
    # --------------------------------------------------------

    try:
        import shap
        shap_available = True
        shap_version = getattr(shap, "__version__", "unknown")

    except ImportError:
        shap_available = False
        shap_version = None

    if not shap_available:
        st.error(
            "SHAP is not installed in the current environment."
        )

        st.code(
            "python -m pip install shap",
            language="bash"
        )

        return

    st.sidebar.markdown("## ⚙️ Controls")

    store = st.sidebar.selectbox(
        "Store",
        list(range(1, NUM_STORES + 1))
    )

    item = st.sidebar.selectbox(
        "Item",
        list(range(1, NUM_ITEMS + 1))
    )

    n_explain = st.sidebar.slider(
        "Rows to explain",
        min_value=1,
        max_value=50,
        value=10
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    with st.spinner("Loading XGBoost model..."):

        model = load_xgb_model()

    if model is None:
        st.error(
            "XGBoost model not found. "
            "Ensure models/xgb_model.joblib exists."
        )
        return

    # --------------------------------------------------------
    # Model features
    # --------------------------------------------------------

    model_features = get_model_features(model)

    if model_features:

        st.sidebar.success(
            f"Model uses {len(model_features)} features"
        )

    else:

        st.sidebar.warning(
            "Could not read model feature names."
        )

        return

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    with st.spinner("Loading validation data..."):

        df = load_featured_data()

    _, val_df = train_val_split(df)

    subset = val_df[
        (val_df["store"] == store) &
        (val_df["item"] == item)
    ].copy()

    if subset.empty:

        st.warning(
            "No validation data for this store/item combination."
        )

        return

    # --------------------------------------------------------
    # Align features
    # --------------------------------------------------------

    X_explain = align_features(
        subset,
        model_features
    )

    if X_explain.empty:

        st.error(
            "No valid rows remain after feature alignment."
        )

        return

    # Keep only requested rows
    X_explain = X_explain.head(n_explain)

    # Actual values aligned to same indexes
    y_explain = subset.loc[
        X_explain.index,
        "sales"
    ]

    st.caption(
        f"Explaining **{len(X_explain)} rows** "
        f"with **{X_explain.shape[1]} features**."
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(X_explain)

    predictions = np.asarray(predictions).reshape(-1)

    predictions = np.clip(predictions, 0, None)

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Computing SHAP explanations..."
        ):

            shap_values, base_value, shap_method = get_shap_values(
                model,
                X_explain
            )

        if shap_values.shape != X_explain.shape:

            raise ValueError(
                f"SHAP shape {shap_values.shape} "
                f"does not match feature shape {X_explain.shape}"
            )

        shap_df = pd.DataFrame(
            shap_values,
            columns=X_explain.columns,
            index=X_explain.index
        )

        st.success(
            f"SHAP explanations generated successfully "
            f"using {shap_method}."
        )

        # ====================================================
        # Global SHAP Importance
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🌐 Global SHAP Feature Importance"
        )

        st.caption(
            "Average absolute SHAP value shows how strongly "
            "each feature influences demand predictions."
        )

        mean_shap = (
            shap_df
            .abs()
            .mean()
            .sort_values(ascending=True)
        )

        fig_global = go.Figure(
            go.Bar(
                x=mean_shap.values,
                y=mean_shap.index,
                orientation="h",
                marker_color="#3498DB"
            )
        )

        fig_global.update_layout(
            title="Mean |SHAP| per Feature",
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title="Feature",
            template="plotly_dark",
            height=550
        )

        st.plotly_chart(
            fig_global,
            use_container_width=True
        )

        # ====================================================
        # Single Prediction
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🔬 Single-Prediction Explanation"
        )

        row_idx = st.slider(
            "Select validation row",
            min_value=0,
            max_value=len(X_explain) - 1,
            value=0
        )

        row_X = X_explain.iloc[row_idx]

        row_shap = shap_df.iloc[row_idx]

        pred_value = float(
            predictions[row_idx]
        )

        actual_value = float(
            y_explain.iloc[row_idx]
        )

        # ----------------------------------------------------
        # Prediction metrics
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Actual Sales",
            f"{actual_value:.1f}"
        )

        c2.metric(
            "Predicted Demand",
            f"{pred_value:.1f}"
        )

        c3.metric(
            "Base Demand",
            f"{base_value:.1f}"
        )

        c4.metric(
            "Prediction Error",
            f"{actual_value - pred_value:+.1f}"
        )

        # ----------------------------------------------------
        # SHAP contribution chart
        # ----------------------------------------------------

        top_features = (
            row_shap
            .abs()
            .sort_values(ascending=False)
            .head(12)
            .index
        )

        contribution = row_shap[
            top_features
        ].sort_values()

        feature_values = [
            row_X[f]
            for f in contribution.index
        ]

        labels = [
            f"{feature}<br>"
            f"<sup>value = {value:.2f}</sup>"
            for feature, value
            in zip(contribution.index, feature_values)
        ]

        colors = [
            "#E74C3C" if value > 0
            else "#3498DB"
            for value in contribution.values
        ]

        fig_waterfall = go.Figure()

        fig_waterfall.add_trace(
            go.Bar(
                x=contribution.values,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[
                    f"{value:+.2f}"
                    for value in contribution.values
                ],
                textposition="outside"
            )
        )

        fig_waterfall.add_vline(
            x=0,
            line_dash="dash",
            line_color="white"
        )

        fig_waterfall.update_layout(
            title=(
                f"SHAP Contributions — "
                f"Store {store}, Item {item}, Row {row_idx}"
            ),
            xaxis_title="SHAP Contribution to Predicted Demand",
            yaxis_title="Feature",
            template="plotly_dark",
            height=550
        )

        st.plotly_chart(
            fig_waterfall,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        positive = (
            row_shap[row_shap > 0]
            .sort_values(ascending=False)
            .head(3)
        )

        negative = (
            row_shap[row_shap < 0]
            .sort_values()
            .head(3)
        )

        st.subheader("💡 Prediction Explanation")

        st.markdown(
            f"""
            The model starts from a baseline demand of
            **{base_value:.2f} units** and adjusts that value
            using the feature contributions.
            
            **Final predicted demand: {pred_value:.2f} units**
            """
        )

        if not positive.empty:

            st.markdown("**Features increasing demand:**")

            for feature, value in positive.items():

                st.write(
                    f"🔴 `{feature}` → **+{value:.2f} units**"
                )

        if not negative.empty:

            st.markdown("**Features decreasing demand:**")

            for feature, value in negative.items():

                st.write(
                    f"🔵 `{feature}` → **{value:.2f} units**"
                )

        # ====================================================
        # SHAP Heatmap
        # ====================================================

        st.markdown("---")

        st.subheader(
            f"🗺️ SHAP Heatmap — {len(X_explain)} Rows"
        )

        top_heat_features = (
            shap_df
            .abs()
            .mean()
            .nlargest(12)
            .index
            .tolist()
        )

        heat_df = shap_df[
            top_heat_features
        ].T

        heat_df.columns = [
            f"Row {i}"
            for i in range(len(heat_df.columns))
        ]

        fig_heat = px.imshow(
            heat_df,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            template="plotly_dark",
            aspect="auto",
            title="SHAP Contribution Heatmap"
        )

        st.plotly_chart(
            fig_heat,
            use_container_width=True
        )

        # ====================================================
        # Prediction table
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🔮 Predictions vs Actual"
        )

        result = pd.DataFrame({
            "actual": y_explain.values,
            "predicted": predictions.round(2)
        })

        result["error"] = (
            result["actual"] -
            result["predicted"]
        ).round(2)

        st.dataframe(
            result,
            use_container_width=True
        )

        # ====================================================
        # SHAP explanation reference
        # ====================================================

        with st.expander(
            "📖 How to Read SHAP Values"
        ):

            st.markdown(
                """
                | SHAP Result | Meaning |
                |---|---|
                | 🔴 Positive SHAP | Feature increased predicted demand |
                | 🔵 Negative SHAP | Feature decreased predicted demand |
                | Larger absolute value | Greater influence on the prediction |
                | Base demand | Model's average starting prediction |
                | Final prediction | Base demand + feature contributions |
                """
            )

            st.latex(
                r"\text{Prediction} "
                r"= \text{Base Value} "
                r"+ \sum_i \text{SHAP}_i"
            )

    except Exception as e:

        # ====================================================
        # Diagnostic fallback
        # ====================================================

        st.error(
            "SHAP calculation failed."
        )

        st.code(
            str(e),
            language="text"
        )

        st.info(
            "The model and feature pipeline are working, "
            "but the installed SHAP/XGBoost versions are "
            "not compatible with the current model."
        )

        # Native importance so page still remains useful

        st.markdown("---")

        st.subheader(
            "📊 XGBoost Native Feature Importance"
        )

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "feature": model_features,
            "importance": importances
        }).sort_values(
            "importance",
            ascending=True
        ).tail(20)

        fig = go.Figure(
            go.Bar(
                x=importance_df["importance"],
                y=importance_df["feature"],
                orientation="h"
            )
        )

        fig.update_layout(
            title="Top 20 XGBoost Features",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# Run page
# ============================================================

render()