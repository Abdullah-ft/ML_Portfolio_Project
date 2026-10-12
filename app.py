"""Streamlit app for ML Portfolio Project (classification + regression)."""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")


@st.cache_resource
def load_bundle(name: str):
    pipe = joblib.load(os.path.join(MODELS_DIR, f"{name}_model.joblib"))
    with open(os.path.join(MODELS_DIR, f"{name}_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return pipe, meta


def render_inputs(meta: dict) -> dict:
    values = {}
    for feat in meta["features"]:
        name = feat["name"]
        ftype = feat["type"]
        if ftype == "number":
            step = 1.0 if feat.get("int") else None
            values[name] = st.number_input(
                name.replace("_", " ").title(),
                min_value=feat["min"],
                max_value=feat["max"],
                value=feat["default"],
                step=step,
            )
        elif ftype == "binary":
            values[name] = st.selectbox(
                name.replace("_", " ").title(),
                options=feat["choices"],
                index=feat["choices"].index(feat["default"]),
            )
        elif ftype == "category":
            values[name] = st.selectbox(
                name.replace("_", " ").title(),
                options=feat["choices"],
                index=feat["choices"].index(feat["default"]),
            )
    return values


def main():
    st.set_page_config(
        page_title="ML Portfolio — Returns & Campaign Revenue",
        page_icon="📊",
        layout="wide",
    )
    st.title("ML Portfolio Project")
    st.caption(
        "Predict e-commerce return risk and influencer campaign revenue "
        "(models from ML_Project.ipynb)."
    )

    if not os.path.isdir(MODELS_DIR):
        st.error(
            "Trained models not found. Run `python train_models.py` locally, "
            "or ensure the `models/` folder is committed for Streamlit Cloud."
        )
        st.stop()

    tab_clf, tab_reg = st.tabs(["Return risk (classification)", "Campaign revenue (regression)"])

    with tab_clf:
        pipe_c, meta_c = load_bundle("return_risk")
        extra = meta_c.get("extra", {})
        st.subheader("E-commerce return risk")
        st.write(
            f"**Model:** {extra.get('model_name', '—')} · "
            f"Test F1: {extra.get('metrics', {}).get('F1', 0):.3f} · "
            f"ROC-AUC: {extra.get('metrics', {}).get('ROC-AUC', 0):.3f}"
        )
        with st.form("clf_form"):
            inputs_c = render_inputs(meta_c)
            threshold = st.slider(
                "Decision threshold (flag if P(return) ≥ threshold)",
                0.05,
                0.95,
                0.50,
                0.05,
            )
            submitted_c = st.form_submit_button("Predict return risk")

        if submitted_c:
            row = pd.DataFrame([inputs_c])
            proba = float(pipe_c.predict_proba(row)[0, 1])
            label = int(proba >= threshold)
            st.metric("P(order returned)", f"{proba:.1%}")
            if label == 1:
                st.warning("Prediction: **Likely returned** — consider fit guidance or review.")
            else:
                st.success("Prediction: **Likely kept**.")

    with tab_reg:
        pipe_r, meta_r = load_bundle("campaign_revenue")
        extra_r = meta_r.get("extra", {})
        st.subheader("Influencer campaign revenue (14-day trackable USD)")
        m = extra_r.get("metrics", {})
        st.write(
            f"**Model:** {extra_r.get('model_name', '—')} · "
            f"Test MAE: ${m.get('MAE', 0):,.0f} · "
            f"RMSE: ${m.get('RMSE', 0):,.0f} · "
            f"R²: {m.get('R2', 0):.3f}"
        )
        with st.form("reg_form"):
            inputs_r = render_inputs(meta_r)
            submitted_r = st.form_submit_button("Predict revenue")

        if submitted_r:
            row = pd.DataFrame([inputs_r])
            revenue = float(pipe_r.predict(row)[0])
            mae = m.get("MAE", 0)
            st.metric("Expected campaign revenue (USD)", f"${revenue:,.0f}")
            st.info(f"Typical error band: about ± ${mae:,.0f} (test-set MAE).")

    with st.expander("About"):
        st.markdown(
            """
            - **Classification target:** `return_risk_label` (1 = returned).
            - **Regression target:** `campaign_revenue_usd` within 14 days of the post.
            - Preprocessing and models are bundled in scikit-learn `Pipeline` objects
              (same design as the notebook).
            """
        )


if __name__ == "__main__":
    main()
