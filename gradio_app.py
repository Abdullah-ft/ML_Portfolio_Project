import json, joblib, pandas as pd, gradio as gr

def load(prefix):
    return (joblib.load(f"models/{prefix}_pipeline.joblib"),
            json.load(open(f"models/{prefix}_meta.json")))

clf, meta_c = load("clf")
reg, meta_r = load("reg")

def make_inputs(meta):
    comps = []
    for f in meta["features"]:
        if f["type"] == "number":
            comps.append(gr.Number(label=f["name"], value=f["default"],
                                   minimum=f["min"], maximum=f["max"],
                                   precision=0 if f["int"] else None))
        elif f["type"] == "binary":
            comps.append(gr.Radio([0, 1], label=f["name"], value=f["default"]))
        else:
            comps.append(gr.Dropdown(f["choices"], label=f["name"], value=f["default"]))
    return comps

def make_predict(pipe, meta, kind):
    names = [f["name"] for f in meta["features"]]
    def predict(*vals):
        X = pd.DataFrame([dict(zip(names, vals))])
        if kind == "clf":
            p = float(pipe.predict_proba(X)[0, 1])
            verdict = "⚠️ Likely RETURNED" if p >= 0.5 else "✅ Likely KEPT"
            return {"Returned": p, "Kept": 1 - p}, verdict
        return f"${pipe.predict(X)[0]:,.0f}  (typical error ≈ ±${meta['extra']['mae']:,.0f})"
    return predict

with gr.Blocks(title="ML Portfolio App") as demo:
    gr.Markdown("# 🛍️📈 ML Portfolio App")
    with gr.Tab("Return risk (Classification)"):
        ins_c = make_inputs(meta_c)
        btn_c = gr.Button("Predict", variant="primary")
        label_c = gr.Label(label="Probability")
        verdict_c = gr.Textbox(label="Verdict")
        btn_c.click(make_predict(clf, meta_c, "clf"), ins_c, [label_c, verdict_c])
    with gr.Tab("Campaign revenue (Regression)"):
        ins_r = make_inputs(meta_r)
        btn_r = gr.Button("Predict", variant="primary")
        out_r = gr.Textbox(label="Predicted 14-day revenue")
        btn_r.click(make_predict(reg, meta_r, "reg"), ins_r, out_r)

if __name__ == "__main__":
    demo.launch()