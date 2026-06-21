"""
model.py — Train and evaluate TWO severity classifiers.

Model 1: Priority classifier  (target: priority_high = High/Low)
  - Uses event_cause, time, location, veh_type, corridor-freq, road-closure flag.
  - Corridor target-encoding REMOVED (it trivially reconstructs the label).
  - Expected honest F1: 0.75–0.88 on test window.

Model 2: Road closure classifier  (target: requires_road_closure_bool)
  - Genuinely hard: imbalanced 8.3% TRUE / 91.7% FALSE.
  - More operationally direct: "does this event need a physical barricade?"
  - Treated with SMOTE-style class_weight and PR-AUC optimisation.

Compute: CPU-only. Both models train in < 60 seconds on 8k rows.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, average_precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
)
from xgboost import XGBClassifier

from src.config import XGB_PARAMS, MODEL_PATH

CLOSURE_MODEL_PATH = MODEL_PATH.replace("severity_xgb", "closure_xgb")


# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X, y, threshold=0.5, label="", pos_label=1):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    f1_pos = f1_score(y, preds, pos_label=pos_label, zero_division=0)
    prauc = average_precision_score(y, proba)
    report = classification_report(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)
    print(f"\n{'='*55}\n{label}  (threshold={threshold:.3f})")
    print(f"F1 (pos): {f1_pos:.4f}  |  PR-AUC: {prauc:.4f}")
    print(report)
    return {"f1_pos": f1_pos, "prauc": prauc, "cm": cm.tolist(), "threshold": threshold}


def tune_threshold(model, X_val, y_val, target_recall=0.80):
    proba = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, proba, pos_label=1)
    best_thresh, best_prec = 0.5, 0.0
    for p, r, t in zip(prec[:-1], rec[:-1], thresholds):
        if r >= target_recall and p > best_prec:
            best_thresh = float(t)
            best_prec = p
    print(f"[model] Threshold → {best_thresh:.3f}  (prec≈{best_prec:.3f} @ recall≥{target_recall})")
    return best_thresh


def _make_xgb(scale_pos):
    params = {k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"}
    params["scale_pos_weight"] = scale_pos
    return XGBClassifier(**params)


# ─────────────────────────────────────────────────────────────────────────────
def train_priority_model(X_train, y_train, X_val, y_val, X_test, y_test, save_dir):
    """Model 1: priority_high classification."""
    print("\n" + "█"*55)
    print("  MODEL 1 — Priority Severity (High vs Low)")
    print("█"*55)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    evaluate(dummy, X_val, y_val, label="Dummy baseline (val)")

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    evaluate(lr, X_val, y_val, label="LogReg baseline (val)")

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = _make_xgb(scale_pos)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    threshold = tune_threshold(xgb, X_val, y_val, target_recall=0.80)
    val_res  = evaluate(xgb, X_val,  y_val,  threshold, "XGBoost Priority (val)")
    test_res = evaluate(xgb, X_test, y_test, threshold, "XGBoost Priority (TEST)")

    # Plots
    _plot_importances(xgb, X_train, save_dir, "priority_feature_importance.png",
                      "Priority Model — Feature Importances (Top 20)")
    _plot_cm(test_res["cm"], ["Low", "High"], save_dir, "priority_confusion_matrix.png",
             "Priority Model — Test Confusion Matrix")

    joblib.dump({"model": xgb, "threshold": threshold}, MODEL_PATH)
    return xgb, threshold, {"val": val_res, "test": test_res, "threshold": threshold}


def train_closure_model(X_train, y_train, X_val, y_val, X_test, y_test, save_dir):
    """Model 2: requires_road_closure classification (imbalanced: ~8% TRUE)."""
    print("\n" + "█"*55)
    print("  MODEL 2 — Road Closure Needed? (True vs False)")
    print(f"  Class balance — TRUE: {y_train.sum()} | FALSE: {(y_train==0).sum()}")
    print("█"*55)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    evaluate(dummy, X_val, y_val, label="Dummy baseline (val)", pos_label=1)

    scale_pos = (y_train == 0).sum() / max(y_train.sum(), 1)
    print(f"[model] scale_pos_weight = {scale_pos:.1f} (handles imbalance)")
    xgb = _make_xgb(scale_pos)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # For imbalanced: tune for recall≥0.60 (missing a closure = under-deployed)
    threshold = tune_threshold(xgb, X_val, y_val, target_recall=0.60)
    val_res  = evaluate(xgb, X_val,  y_val,  threshold, "XGBoost Closure (val)")
    test_res = evaluate(xgb, X_test, y_test, threshold, "XGBoost Closure (TEST)")

    _plot_importances(xgb, X_train, save_dir, "closure_feature_importance.png",
                      "Closure Model — Feature Importances (Top 20)")
    _plot_cm(test_res["cm"], ["No Closure", "Closure"], save_dir,
             "closure_confusion_matrix.png", "Closure Model — Test Confusion Matrix")

    joblib.dump({"model": xgb, "threshold": threshold}, CLOSURE_MODEL_PATH)
    return xgb, threshold, {"val": val_res, "test": test_res, "threshold": threshold}


# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       X_train_c=None, y_train_c=None,
                       X_val_c=None,   y_val_c=None,
                       X_test_c=None,  y_test_c=None,
                       save_dir=None):
    """
    Train both models. If closure arrays are provided, trains Model 2 as well.
    Returns (priority_xgb, priority_threshold, closure_xgb, closure_threshold, all_results)
    """
    save_dir = save_dir or os.path.dirname(MODEL_PATH)
    os.makedirs(save_dir, exist_ok=True)

    priority_xgb, p_thresh, p_res = train_priority_model(
        X_train, y_train, X_val, y_val, X_test, y_test, save_dir
    )

    closure_xgb, c_thresh, c_res = None, None, {}
    if X_train_c is not None:
        closure_xgb, c_thresh, c_res = train_closure_model(
            X_train_c, y_train_c, X_val_c, y_val_c, X_test_c, y_test_c, save_dir
        )

    all_results = {"priority": p_res, "closure": c_res}
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    print(f"\n[model] All artifacts saved to {save_dir}/")
    return priority_xgb, p_thresh, closure_xgb, c_thresh, all_results


# ─────────────────────────────────────────────────────────────────────────────
def _plot_importances(model, X_train, save_dir, filename, title):
    imp = pd.Series(model.feature_importances_, index=X_train.columns)
    top = imp.nlargest(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    top.sort_values().plot.barh(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Gain importance")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=120)
    plt.close(fig)


def _plot_cm(cm_list, labels, save_dir, filename, title):
    cm = np.array(cm_list)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def load_model(path=None):
    path = path or MODEL_PATH
    artifact = joblib.load(path)
    return artifact["model"], artifact["threshold"]


def predict(model, threshold, X: pd.DataFrame):
    proba = model.predict_proba(X)[:, 1]
    labels = (proba >= threshold).astype(int)
    return labels, proba


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    from src.config import CLEAN_PARQUET
    from src.features import build_features, FeatureBuilder
    from src.split import time_split

    df = pd.read_parquet(CLEAN_PARQUET)
    train, val, test = time_split(df)

    # Model 1: priority
    builder, X_tr, y_tr, X_v, y_v, X_te, y_te = build_features(train, val, test)
    # Model 2: closure (same builder, same X, different y)
    _, y_tr_c = builder.transform_secondary(train)
    _, y_v_c  = builder.transform_secondary(val)
    _, y_te_c = builder.transform_secondary(test)

    train_and_evaluate(
        X_tr, y_tr, X_v, y_v, X_te, y_te,
        X_tr, y_tr_c, X_v, y_v_c, X_te, y_te_c
    )
    joblib.dump(builder, MODEL_PATH.replace("severity_xgb.pkl", "severity_xgb_builder.pkl"))
