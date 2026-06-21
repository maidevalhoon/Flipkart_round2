"""
features.py — Feature engineering (leakage-free).

FeatureBuilder follows sklearn fit/transform pattern:
  - fit_transform(train_df) → (X_train, y_train)  [fits encoders on train]
  - transform(df) → (X, y)                         [applies frozen encoders]

All encoders fitted exclusively on training data.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from src.config import (
    PEAK_HOURS, HOUR_BUCKETS, MIN_CAUSE_COUNT, GEO_N_CLUSTERS,
    ROLLING_DAYS, TARGET_COL
)


class FeatureBuilder:
    def __init__(self):
        self._fitted = False
        self._geo_kmeans = None
        self._cause_map = None        # rare causes → "other"
        self._target_encoders = {}    # col → {cat: mean_target}
        self._train_cols = None       # column order fixed after fit

    # ─────────────────────────────────────────────────────────────────────────
    def fit_transform(self, train_df: pd.DataFrame):
        df = train_df.copy()
        y = df[TARGET_COL].values

        # 1. Cause rareness map (based on train only)
        cause_counts = df["event_cause"].value_counts()
        rare = set(cause_counts[cause_counts < MIN_CAUSE_COUNT].index)
        self._cause_map = rare
        df["event_cause"] = df["event_cause"].apply(
            lambda x: "other" if x in rare else x
        )

        # 2. Geo KMeans (fit on train lat/lon)
        coords = df[["lat", "lon"]].values
        self._geo_kmeans = KMeans(n_clusters=GEO_N_CLUSTERS, random_state=42, n_init=10)
        df["geo_cluster"] = self._geo_kmeans.fit_predict(coords)

        # 3. Target-encode high-cardinality categoricals (corridor, zone, police_station)
        for col in ["corridor", "zone", "gba_identifier", "police_station"]:
            enc = df.groupby(col)[TARGET_COL].mean().to_dict()
            overall = df[TARGET_COL].mean()
            self._target_encoders[col] = (enc, overall)
            df[f"{col}_enc"] = df[col].map(enc).fillna(overall)

        df = _build_all_features(df)
        df = _drop_raw(df)
        X = df.drop(columns=[TARGET_COL])
        self._train_cols = X.columns.tolist()
        self._fitted = True
        return X, y

    # ─────────────────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame):
        df = df.copy()
        y = df[TARGET_COL].values if TARGET_COL in df.columns else None

        # Apply rare-cause map
        df["event_cause"] = df["event_cause"].apply(
            lambda x: "other" if x in self._cause_map else x
        )

        # Geo cluster
        coords = df[["lat", "lon"]].values
        df["geo_cluster"] = self._geo_kmeans.predict(coords)

        # Target encoding (frozen from train)
        for col, (enc, overall) in self._target_encoders.items():
            df[f"{col}_enc"] = df[col].map(enc).fillna(overall)

        df = _build_all_features(df)
        df = _drop_raw(df)

        X = df.drop(columns=[TARGET_COL], errors="ignore")
        # Align columns to training schema
        X = X.reindex(columns=self._train_cols, fill_value=0)
        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (pure functions, no state)
# ─────────────────────────────────────────────────────────────────────────────

def _hour_bucket(hour: int) -> str:
    for name, (lo, hi) in HOUR_BUCKETS.items():
        if lo <= hour <= hi:
            return name
    return "unknown"


def _build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    # ── Event type ───────────────────────────────────────────────────────────
    df["event_type_planned"] = (df["event_type"] == "planned").astype(int)

    # ── Road closure ─────────────────────────────────────────────────────────
    df["requires_road_closure_int"] = df["requires_road_closure_bool"].astype(int)

    # ── Corridor flag ────────────────────────────────────────────────────────
    df["is_corridor"] = (df["corridor"].str.lower() != "non-corridor").astype(int)

    # ── Temporal features (IST already applied at preprocessing) ─────────────
    dt = df["start_datetime"].dt
    df["hour"]        = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["month"]       = dt.month
    df["is_weekend"]  = (dt.dayofweek >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].apply(lambda h: int(h in PEAK_HOURS))
    df["hour_bucket"] = df["hour"].apply(_hour_bucket)

    # ── One-hot: event_cause ─────────────────────────────────────────────────
    df = pd.get_dummies(df, columns=["event_cause"], prefix="cause", dtype=int)

    # ── One-hot: veh_type ────────────────────────────────────────────────────
    df = pd.get_dummies(df, columns=["veh_type"], prefix="veh", dtype=int)

    # ── One-hot: hour_bucket ─────────────────────────────────────────────────
    df = pd.get_dummies(df, columns=["hour_bucket"], prefix="hbkt", dtype=int)

    return df


def _drop_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that were only needed to derive features."""
    to_drop = [
        "event_type", "corridor", "zone", "gba_identifier", "police_station",
        "junction", "requires_road_closure_bool",
        "start_datetime",
        "veh_no", "description", "address",
    ]
    return df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
def build_features(train_df, val_df=None, test_df=None):
    """Convenience wrapper returning (builder, X_train, y_train, ...)."""
    builder = FeatureBuilder()
    X_train, y_train = builder.fit_transform(train_df)
    results = [builder, X_train, y_train]
    for df in [val_df, test_df]:
        if df is not None:
            X, y = builder.transform(df)
            results += [X, y]
    return tuple(results)
