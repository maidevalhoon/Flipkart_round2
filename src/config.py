"""
Central configuration — all magic numbers and paths live here.
Modify this file to tune thresholds without touching model/pipeline code.
"""
import os

# ── Paths (relative to project root; overridden by COLAB_ROOT env var) ──────
_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_CSV = os.path.join(_ROOT, "data", "raw", "events.csv")
CLEAN_PARQUET = os.path.join(_ROOT, "data", "processed", "clean.parquet")
FEATURES_PARQUET = os.path.join(_ROOT, "data", "processed", "features.parquet")
MODEL_PATH = os.path.join(_ROOT, "models", "severity_xgb.pkl")
FEEDBACK_LOG = os.path.join(_ROOT, "feedback_log.csv")

# ── Bengaluru geographic bounding box ────────────────────────────────────────
GEO_BBOX = {"lat_min": 12.7, "lat_max": 13.25, "lon_min": 77.3, "lon_max": 77.9}

# ── Temporal split boundaries (ISO date strings) ─────────────────────────────
TRAIN_END   = "2024-02-15"
VAL_END     = "2024-03-10"
# test = anything after VAL_END

# ── Feature definitions ───────────────────────────────────────────────────────
TARGET_COL = "priority_high"   # 1 = High, 0 = Low

# Columns available at report time (forecast-safe)
FEATURE_COLS = [
    "event_type_planned",
    "requires_road_closure_bool",
    "is_corridor",
    "hour", "hour_bucket", "day_of_week", "is_weekend", "month", "is_peak_hour",
    "lat", "lon",
    "geo_cluster",
    "dup_cluster_size",
    # one-hot encoded (appended dynamically by features.py):
    # cause_*, veh_type_*, corridor_enc, zone_enc, police_enc
]

# event_cause values to collapse into "other" (fewer than MIN_CAUSE_COUNT training rows)
MIN_CAUSE_COUNT = 20

# Peak hours (IST) based on observed data distribution
PEAK_HOURS = {5, 6, 19, 20, 21, 22}

# Hour buckets: label → [start_hour_inclusive, end_hour_inclusive]
HOUR_BUCKETS = {
    "night":   (0, 4),
    "morning": (5, 10),
    "midday":  (11, 15),
    "evening": (16, 20),
    "late":    (21, 23),
}

# Columns to drop unconditionally (empty, audit, leakage)
DROP_COLS = [
    "map_file", "comment", "meta_data",
    "cargo_material", "reason_breakdown", "age_of_truck",
    "route_path", "direction",
    "assigned_to_police_id", "citizen_accident_id",
    "resolved_at_address", "resolved_at_latitude", "resolved_at_longitude",
    "resolved_by_id",
    "kgid", "created_by_id", "last_modified_by_id", "closed_by_id",
    "client_id",
    # post-event leakage:
    "status", "authenticated", "modified_datetime",
    "end_datetime", "closed_datetime", "resolved_datetime",
    "created_date",
    "end_address",
    "endlatitude", "endlongitude",
]

# ── Geo clustering (KMeans k) ─────────────────────────────────────────────────
GEO_N_CLUSTERS = 15

# ── Historical rolling window (days) for hotspot features ────────────────────
ROLLING_DAYS = 7

# ── XGBoost hyperparameters (baseline; tune via RandomizedSearchCV) ──────────
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
}

# Decision threshold tuned on validation set (updated by model.py)
DEFAULT_THRESHOLD = 0.5

# ── Recommendation engine rules (all thresholds centralised here) ─────────────
RULES = {
    "base_manpower": {"High": 4, "Low": 1},
    "closure_bonus_manpower": 2,
    "crowd_bonus_manpower": 2,      # public_event, procession, protest, vip_movement
    "corridor_bonus_manpower": 1,
    "peak_hour_bonus_manpower": 1,  # only when High severity
    "hotspot_bonus_per_2_dups": 1,  # per 2 dup_cluster_size, capped at 3
    "hotspot_cap": 3,

    "base_barricades": {"High_no_closure": 2, "Low_no_closure": 0, "closure": 4},
    "crowd_barricade_bonus": 2,

    "diversion_causes": {"public_event", "procession", "protest", "vip_movement", "congestion"},

    # Fallback: event_cause → default High-rate lookup (from EDA)
    "cause_high_rate": {
        "vehicle_breakdown": 0.66,
        "congestion": 0.69,
        "construction": 0.63,
        "others": 0.60,
        "pot_holes": 0.56,
        "water_logging": 0.59,
        "accident": 0.46,
        "road_conditions": 0.55,
        "tree_fall": 0.33,
        "public_event": 0.50,
        "procession": 0.32,
        "vip_movement": 0.35,
        "protest": 0.40,
        "default": 0.50,
    },
    "fallback_high_threshold": 0.55,  # cause_high_rate >= this → predict High in fallback
}
