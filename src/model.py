"""
model.py — Train, evaluate, and save the severity classifier.

Pipeline:
  Baseline (DummyClassifier + LogReg) → XGBoost (primary)
  Threshold tuned on validation set for High-recall ≥ 0.80
  Reports F1, PR-AUC, confusion matrix on held-out test window.

Compute: CPU-only, ~seconds on 8k rows. No GPU required.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, average_precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
)
from xgboost import XGBClassifier

from src.config import XGB_PARAMS, MODEL_PATH


# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X, y, threshold=0.5, label=""):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    f1_high = f1_score(y, preds, pos_label=1, zero_division=0)
    prauc = average_precision_score(y, proba)
    report = classification_report(y, preds, target_names=["Low", "High"], zero_division=0)
    cm = confusion_matrix(y, preds)
    print(f"\n{'='*50}\n{label} (threshold={threshold:.2f})")
    print(f"F1 (High): {f1_high:.4f}  |  PR-AUC: {prauc:.4f}")
    print(report)
    return {"f1_high": f1_high, "prauc": prauc, "cm": cm.tolist(), "threshold": threshold}


def tune_threshold_for_recall(model, X_val, y_val, target_recall=0.80):
    """Find lowest threshold that achieves target recall on High class."""
    proba = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, proba, pos_label=1)
    # prec, rec, thresholds: len(thresholds) == len(prec) - 1
    best_thresh = 0.5
    best_prec = 0.0
    for p, r, t in zip(prec[:-1], rec[:-1], thresholds):
        if r >= target_recall and p > best_prec:
            best_thresh = float(t)
            best_prec = p
    print(f"[model] Threshold tuned: {best_thresh:.3f} "
          f"→ estimated precision={best_prec:.3f} at recall≥{target_recall}")
    return best_thresh


# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       save_dir=None):
    results = {}
    save_dir = save_dir or os.path.dirname(MODEL_PATH)
    os.makedirs(save_dir, exist_ok=True)

    # ── Baseline 1: always predict majority ──────────────────────────────────
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    results["dummy_val"] = evaluate(dummy, X_val, y_val, label="Dummy (val)")

    # ── Baseline 2: Logistic Regression ──────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    results["lr_val"] = evaluate(lr, X_val, y_val, label="LogReg (val)")

    # ── Primary: XGBoost ─────────────────────────────────────────────────────
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_params = {**XGB_PARAMS, "scale_pos_weight": scale_pos}
    xgb_params.pop("use_label_encoder", None)   # removed in XGB ≥2.0

    xgb = XGBClassifier(**xgb_params)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Tune threshold on validation
    threshold = tune_threshold_for_recall(xgb, X_val, y_val, target_recall=0.80)
    results["xgb_val"] = evaluate(xgb, X_val, y_val, threshold=threshold, label="XGBoost (val)")
    results["xgb_test"] = evaluate(xgb, X_test, y_test, threshold=threshold, label="XGBoost (TEST)")
    results["tuned_threshold"] = threshold

    # ── Feature importance plot ───────────────────────────────────────────────
    importances = pd.Series(xgb.feature_importances_, index=X_train.columns)
    top20 = importances.nlargest(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    top20.sort_values().plot.barh(ax=ax)
    ax.set_title("XGBoost Feature Importances (Gain, Top 20)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=120)
    plt.close(fig)

    # ── Confusion matrix plot (test) ─────────────────────────────────────────
    cm = np.array(results["xgb_test"]["cm"])
    fig2, ax2 = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["Low", "High"]).plot(ax=ax2)
    ax2.set_title("XGBoost — Test Confusion Matrix")
    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig2)

    # ── Save model + results ──────────────────────────────────────────────────
    joblib.dump({"model": xgb, "threshold": threshold}, MODEL_PATH)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        # cm arrays → lists for JSON
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    print(f"\n[model] Saved → {MODEL_PATH}")
    print(f"[model] Plots  → {save_dir}/feature_importance.png | confusion_matrix.png")
    print(f"[model] Metrics → {save_dir}/metrics.json")
    return xgb, threshold, results


# ─────────────────────────────────────────────────────────────────────────────
def load_model(path=None):
    """Load saved model artifact. Returns (xgb_model, threshold)."""
    path = path or MODEL_PATH
    artifact = joblib.load(path)
    return artifact["model"], artifact["threshold"]


def predict(model, threshold, X: pd.DataFrame):
    """Returns (labels, probabilities)."""
    proba = model.predict_proba(X)[:, 1]
    labels = (proba >= threshold).astype(int)
    return labels, proba


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.config import CLEAN_PARQUET
    from src.preprocessing import load_and_clean, save_clean
    from src.features import build_features
    from src.split import time_split

    df = pd.read_parquet(CLEAN_PARQUET)
    train, val, test = time_split(df)
    builder, X_train, y_train, X_val, y_val, X_test, y_test = build_features(
        train, val_df=val, test_df=test
    )
    train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
