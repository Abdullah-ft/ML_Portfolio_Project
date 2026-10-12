"""Shared training utilities (from ML_Project.ipynb)."""

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RS = 42
np.random.seed(RS)

CLF_FILE = "ecommerce_return_classification.csv"
REG_FILE = "influencer_campaign_regression.csv"

num_c = ["cart_value_usd", "past_return_rate", "delivery_days_taken", "discount_pct_applied"]
bin_c = ["size_bracketing_flag", "newsletter_subscriber"]
cat_c = [
    "product_category",
    "fit_review_sentiment",
    "browser_used",
    "checkout_device",
    "packaging_color",
    "loyalty_card_color",
]

num_r = [
    "creator_followers_k",
    "engagement_rate_pct",
    "paid_boost_spend_usd",
    "avg_watch_time_sec",
    "promo_discount_pct",
    "caption_word_count",
    "hashtags_used_count",
]
cat_r = [
    "content_format",
    "audience_intent_tier",
    "creator_phone_os",
    "contract_payment_terms",
    "posting_day_weather",
]


def build_preprocessor(num_cols, bin_cols, cat_cols):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("bin", "passthrough", bin_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def make_pipe(model, num, binr, cat):
    return Pipeline([("prep", build_preprocessor(num, binr, cat)), ("model", model)])


def build_meta(X_ref, num_cols, bin_cols, cat_cols, extra=None):
    feats = []
    for c in num_cols:
        s = X_ref[c]
        feats.append(
            {
                "name": c,
                "type": "number",
                "min": float(s.min()),
                "max": float(s.max()),
                "default": float(s.median()),
                "int": bool(pd.api.types.is_integer_dtype(s)),
            }
        )
    for c in bin_cols:
        feats.append({"name": c, "type": "binary", "choices": [0, 1], "default": 0})
    for c in cat_cols:
        vals = sorted(X_ref[c].unique().tolist())
        feats.append(
            {"name": c, "type": "category", "choices": vals, "default": X_ref[c].mode()[0]}
        )
    return {"features": feats, "extra": extra or {}}


def _sig_classification(df):
    rows = []
    target = "return_risk_label"
    for c in num_c:
        a = df.loc[df[target] == 1, c]
        b = df.loc[df[target] == 0, c]
        stat, p = stats.mannwhitneyu(a, b)
        rows.append({"feature": c, "p_value": p})
    for c in bin_c + cat_c:
        ct = pd.crosstab(df[c], df[target])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        rows.append({"feature": c, "p_value": p})
    sig = pd.DataFrame(rows).set_index("feature")
    return sig.index[sig["p_value"] < 0.05].tolist()


def _sig_regression(df):
    rows = []
    target = "campaign_revenue_usd"
    for c in num_r:
        r_, p = stats.pearsonr(df[c], df[target])
        rows.append({"feature": c, "p_value": p})
    for c in cat_r:
        groups = [g[target].values for _, g in df.groupby(c)]
        f_, p = stats.f_oneway(*groups)
        rows.append({"feature": c, "p_value": p})
    sig = pd.DataFrame(rows).set_index("feature")
    return sig.index[sig["p_value"] < 0.05].tolist()


def train_classification(clf_path: str):
    df_c = pd.read_csv(clf_path)
    target = "return_risk_label"
    df_c_clean = df_c.drop_duplicates().reset_index(drop=True)
    selected_c = _sig_classification(df_c_clean)

    X_all_c = df_c_clean.drop(columns=[target])
    y_c = df_c_clean[target]
    X_train_all_c, X_test_all_c, y_train_c, y_test_c = train_test_split(
        X_all_c, y_c, test_size=0.20, stratify=y_c, random_state=RS
    )

    X_train_all_c, X_test_all_c = X_train_all_c.copy(), X_test_all_c.copy()
    for c in num_c:
        q1, q3 = X_train_all_c[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        X_train_all_c[c] = X_train_all_c[c].clip(lo, hi)
        X_test_all_c[c] = X_test_all_c[c].clip(lo, hi)

    X_train_c = X_train_all_c[selected_c]
    X_test_c = X_test_all_c[selected_c]
    sel_num_c = [c for c in num_c if c in selected_c]
    sel_bin_c = [c for c in bin_c if c in selected_c]
    sel_cat_c = [c for c in cat_c if c in selected_c]

    models_c = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
    }
    param_grids_c = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [5, 8, None],
            "model__min_samples_leaf": [1, 5, 10],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        },
    }
    scoring_c = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

    searches_c = {}
    for name, grid in param_grids_c.items():
        t0 = time.time()
        pipe = make_pipe(clone(models_c[name]), sel_num_c, sel_bin_c, sel_cat_c)
        gs = GridSearchCV(pipe, grid, scoring=scoring_c, refit="f1", cv=skf, n_jobs=-1)
        gs.fit(X_train_c, y_train_c)
        searches_c[name] = gs
        print(
            f"[clf] {name:20s} | best CV F1 = {gs.best_score_:.4f} | {time.time() - t0:.0f}s"
        )

    complexity = {"Logistic Regression": 1, "Random Forest": 2, "Gradient Boosting": 3}
    compare_rows = []
    for name, gs in searches_c.items():
        i = gs.best_index_
        compare_rows.append(
            {
                "Model": name,
                "CV F1": gs.cv_results_["mean_test_f1"][i],
                "Complexity (1=simple)": complexity[name],
            }
        )
    compare_c = pd.DataFrame(compare_rows).set_index("Model")
    best_cv = compare_c["CV F1"].max()
    candidates = compare_c[compare_c["CV F1"] >= best_cv - 0.005]
    final_name_c = candidates["Complexity (1=simple)"].idxmin()
    final_pipe_c = searches_c[final_name_c].best_estimator_

    proba = final_pipe_c.predict_proba(X_test_c)[:, 1]
    pred = final_pipe_c.predict(X_test_c)
    metrics = {
        "Accuracy": accuracy_score(y_test_c, pred),
        "Precision": precision_score(y_test_c, pred, zero_division=0),
        "Recall": recall_score(y_test_c, pred, zero_division=0),
        "F1": f1_score(y_test_c, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test_c, proba),
    }
    print(f"[clf] Final model: {final_name_c} | test: {metrics}")

    meta = build_meta(
        X_train_c,
        sel_num_c,
        sel_bin_c,
        sel_cat_c,
        extra={
            "task": "classification",
            "target_label": "return_risk_label",
            "model_name": final_name_c,
            "metrics": metrics,
            "feature_columns": selected_c,
        },
    )
    return final_pipe_c, meta


def train_regression(reg_path: str):
    df_r = pd.read_csv(reg_path)
    target = "campaign_revenue_usd"
    df_r_clean = df_r.drop_duplicates().reset_index(drop=True)
    selected_r = _sig_regression(df_r_clean)

    X_all_r = df_r_clean.drop(columns=[target])
    y_r = df_r_clean[target]
    X_train_all_r, X_test_all_r, y_train_r, y_test_r = train_test_split(
        X_all_r, y_r, test_size=0.20, random_state=RS
    )
    X_train_r = X_train_all_r[selected_r]
    X_test_r = X_test_all_r[selected_r]
    sel_num_r = [c for c in num_r if c in selected_r]
    sel_cat_r = [c for c in cat_r if c in selected_r]

    models_r = {
        "Ridge Regression": Ridge(random_state=RS),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RS, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RS),
    }
    param_grids_r = {
        "Ridge Regression": {"model__alpha": [0.01, 0.1, 1, 10, 100, 1000]},
        "Random Forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [5, 8, None],
            "model__min_samples_leaf": [1, 5, 10],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        },
    }
    scoring_r = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=RS)

    searches_r = {}
    for name, grid in param_grids_r.items():
        t0 = time.time()
        pipe = make_pipe(clone(models_r[name]), sel_num_r, [], sel_cat_r)
        gs = GridSearchCV(pipe, grid, scoring=scoring_r, refit="rmse", cv=kf, n_jobs=-1)
        gs.fit(X_train_r, y_train_r)
        searches_r[name] = gs
        print(f"[reg] {name:18s} | best CV RMSE = {-gs.best_score_:,.1f} | {time.time() - t0:.0f}s")

    complexity_r = {"Ridge Regression": 1, "Random Forest": 2, "Gradient Boosting": 3}
    compare_rows = []
    for name, gs in searches_r.items():
        i = gs.best_index_
        compare_rows.append(
            {
                "Model": name,
                "CV RMSE": -gs.cv_results_["mean_test_rmse"][i],
                "Complexity (1=simple)": complexity_r[name],
            }
        )
    compare_r = pd.DataFrame(compare_rows).set_index("Model")
    best_rmse = compare_r["CV RMSE"].min()
    candidates = compare_r[compare_r["CV RMSE"] <= best_rmse * 1.01]
    final_name_r = candidates["Complexity (1=simple)"].idxmin()
    final_pipe_r = searches_r[final_name_r].best_estimator_

    pred = final_pipe_r.predict(X_test_r)
    metrics = {
        "MAE": mean_absolute_error(y_test_r, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_r, pred))),
        "R2": r2_score(y_test_r, pred),
    }
    print(f"[reg] Final model: {final_name_r} | test: {metrics}")

    meta = build_meta(
        X_train_r,
        sel_num_r,
        [],
        sel_cat_r,
        extra={
            "task": "regression",
            "target_label": "campaign_revenue_usd",
            "model_name": final_name_r,
            "metrics": metrics,
            "feature_columns": selected_r,
        },
    )
    return final_pipe_r, meta


def save_artifacts(out_dir: str, clf_pipe, clf_meta, reg_pipe, reg_meta):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf_pipe, os.path.join(out_dir, "return_risk_model.joblib"))
    joblib.dump(reg_pipe, os.path.join(out_dir, "campaign_revenue_model.joblib"))
    with open(os.path.join(out_dir, "return_risk_meta.json"), "w", encoding="utf-8") as f:
        json.dump(clf_meta, f, indent=2)
    with open(os.path.join(out_dir, "campaign_revenue_meta.json"), "w", encoding="utf-8") as f:
        json.dump(reg_meta, f, indent=2, default=str)


def train_and_save(models_dir: str | None = None):
    root = os.path.dirname(os.path.abspath(__file__))
    models_dir = models_dir or os.path.join(root, "models")
    clf_path = os.path.join(root, CLF_FILE)
    reg_path = os.path.join(root, REG_FILE)
    if not os.path.isfile(clf_path) or not os.path.isfile(reg_path):
        raise FileNotFoundError(f"CSV files must be in {root}")

    print("Training classification model (GridSearchCV)...")
    clf_pipe, clf_meta = train_classification(clf_path)
    print("Training regression model (GridSearchCV)...")
    reg_pipe, reg_meta = train_regression(reg_path)
    save_artifacts(models_dir, clf_pipe, clf_meta, reg_pipe, reg_meta)
    print(f"Saved models and metadata to {models_dir}")
    return models_dir


if __name__ == "__main__":
    train_and_save()
