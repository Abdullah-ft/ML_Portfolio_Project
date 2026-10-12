import json, joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="ML Portfolio App", page_icon="🛍️", layout="wide")

@st.cache_resource
def load(prefix):
    return (joblib.load(f"models/{prefix}_pipeline.joblib"),
            json.load(open(f"models/{prefix}_meta.json")))

clf, meta_c = load("clf")
reg, meta_r = load("reg")

def input_form(meta, key):
    vals, cols = {}, st.columns(2)
    for i, f in enumerate(meta["features"]):
        k, name = f"{key}_{f['name']}", f["name"]
        with cols[i % 2]:
            if f["type"] == "number" and f["int"]:
                vals[name] = st.number_input(name, int(f["min"]), int(f["max"]),
                                             int(f["default"]), step=1, key=k)
            elif f["type"] == "number":
                vals[name] = st.number_input(name, float(f["min"]), float(f["max"]),
                                             float(f["default"]), key=k)
            elif f["type"] == "binary":
                vals[name] = st.selectbox(name, [0, 1], index=0, key=k)
            else:
                vals[name] = st.selectbox(name, f["choices"],
                                          index=f["choices"].index(f["default"]), key=k)
    return pd.DataFrame([vals])

st.title("🛍️📈 ML Portfolio App")
tab_c, tab_r = st.tabs(["Return risk (Classification)", "Campaign revenue (Regression)"])

with tab_c:
    X = input_form(meta_c, "clf")
    if st.button("Predict return risk", type="primary"):
        p = float(clf.predict_proba(X)[0, 1])
        st.metric("Probability of return", f"{p:.1%}")
        st.progress(p)
        (st.error if p >= 0.5 else st.success)(
            "⚠️ Likely RETURNED" if p >= 0.5 else "✅ Likely KEPT")

with tab_r:
    X = input_form(meta_r, "reg")
    if st.button("Predict revenue", type="primary"):
        y = float(reg.predict(X)[0])
        st.metric("Predicted 14-day revenue", f"${y:,.0f}")
        st.caption(f"Typical error ≈ ±${meta_r['extra']['mae']:,.0f} (test MAE)")