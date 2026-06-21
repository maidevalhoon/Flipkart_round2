"""
realtime_sim.py — Simulated real-time event replay.

Replays test-window events in chronological order, printing predictions
with optional sleep delay to mimic a live feed. Clearly labelled SIMULATED.

Usage:
    python -m src.realtime_sim --delay 0.5 --n 50
    # or from Colab notebook cell:
    from src.realtime_sim import simulate_stream
"""
import time
import argparse
import pandas as pd
import joblib
import numpy as np

from src.config import MODEL_PATH, CLEAN_PARQUET
from src.recommend import recommend, fallback_recommend
from src.config import RULES


def simulate_stream(
    test_df: pd.DataFrame,
    builder,
    model,
    threshold: float,
    delay: float = 0.0,
    n: int = None,
    callback=None,
):
    """
    Generator yielding prediction dicts for each row in test_df (chronological).

    Parameters
    ----------
    test_df  : cleaned test-window DataFrame
    builder  : fitted FeatureBuilder
    model    : XGBClassifier
    threshold: decision threshold
    delay    : seconds to sleep between events (0 = as fast as possible)
    n        : max events to replay (None = all)
    callback : optional callable(result_dict) for Streamlit integration
    """
    from src.model import predict as model_predict

    test_sorted = test_df.sort_values("start_datetime").copy()
    if n:
        test_sorted = test_sorted.head(n)

    X_test, _ = builder.transform(test_sorted)

    labels, probas = model_predict(model, threshold, X_test)

    for i, (idx, row) in enumerate(test_sorted.iterrows()):
        label = "High" if labels[i] == 1 else "Low"
        proba = float(probas[i])

        # Dead-band fallback
        if 0.4 <= proba <= 0.6:
            rec = fallback_recommend(row.get("event_cause", "other"),
                                     requires_road_closure=bool(row.get("requires_road_closure_bool", False)))
        else:
            rec = recommend({
                "severity": label,
                "probability": proba,
                "event_cause": row.get("event_cause", "other"),
                "requires_road_closure": bool(row.get("requires_road_closure_bool", False)),
                "is_corridor": 1 if str(row.get("corridor", "")).lower() != "non-corridor" else 0,
                "event_type": row.get("event_type", "unplanned"),
                "hour": int(row.get("hour", 12)) if "hour" in row.index else 12,
                "hour_bucket": "midday",
                "dup_cluster_size": int(row.get("dup_cluster_size", 1)),
                "corridor_name": str(row.get("corridor", "")),
            })

        result = {
            "[SIMULATED]": True,
            "row_idx": int(idx),
            "timestamp": str(row.get("start_datetime", "")),
            "event_cause": row.get("event_cause", "unknown"),
            "predicted_severity": label,
            "probability": round(proba, 4),
            "actual_priority": "High" if row.get("priority_high", -1) == 1 else "Low",
            "recommendation": rec,
        }

        if callback:
            callback(result)
        else:
            _print_result(result)

        yield result

        if delay > 0:
            time.sleep(delay)


def _print_result(r: dict):
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"[SIMULATED LIVE EVENT] {r['timestamp']}")
    print(f"  Cause     : {r['event_cause']}")
    print(f"  Predicted : {r['predicted_severity']}  (p={r['probability']:.3f})")
    print(f"  Actual    : {r['actual_priority']}")
    rec = r["recommendation"]
    print(f"  Manpower  : {rec['manpower_count']}")
    print(f"  Barricades: {rec['barricade_count']}  — {rec['barricade_placement']}")
    print(f"  Diversion : {'YES' if rec['diversion_suggested'] else 'no'}  {rec['diversion_note']}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between events")
    parser.add_argument("--n", type=int, default=30, help="Number of events to replay")
    args = parser.parse_args()

    from src.preprocessing import load_and_clean
    from src.features import FeatureBuilder
    from src.split import time_split
    from src.config import RAW_CSV

    df = load_and_clean(RAW_CSV)
    _, _, test_df = time_split(df)

    artifact = joblib.load(MODEL_PATH)
    model, threshold = artifact["model"], artifact["threshold"]

    # Re-fit builder on full train for streaming (minimal; reuse saved builder in production)
    train_df, _, _ = time_split(df)
    builder = FeatureBuilder()
    builder.fit_transform(train_df)

    print("\n" + "="*55)
    print("  SIMULATED REAL-TIME REPLAY — test window")
    print("  (This is NOT a live feed — historical data replayed)")
    print("="*55)

    for _ in simulate_stream(test_df, builder, model, threshold,
                              delay=args.delay, n=args.n):
        pass
