"""
features.py — Feature engineering (leakage-free).

FeatureBuilder follows sklearn fit/transform pattern:
  - fit_transform(train_df) → (X_train, y_train)  [fits encoders on train only]
  - transform(df) → (X, y)                         [applies frozen encoders]

Design note on corridor:
  Target-encoding corridor vs priority was removed because corridor perfectly
  predicts priority in this dataset (Non-corridor=0% High, named=100% High).
  Keeping it would make both models trivially accurate and impress nobody.
  Instead we use frequency encoding (log-scaled event count per corridor) which
  preserves the signal that busy corridors matter without reconstructing the label.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from src.config import (
    PEAK_HOURS, HOUR_BUCKETS, MIN_CAUSE_COUNT, GEO_N_CLUSTERS, TARGET_COL
)

SECONDARY_TARGET = "requires_road_closure_bool"


class FeatureBuilder:
    def __init__(self):
        self._fitted = False
        self._geo_kmeans = None
        self._cause_map = None           # rare causes → "other"
        self._freq_encoders = {}         # col → {cat: log_count} (frequency, not target)
        self._dup_cluster_map = {}       # (lat_r2, lon_r2, cause) → count from train only
        self._train_cols = None

    # ─────────────────────────────────────────────────────────────────────────
    def fit_transform(self, train_df: pd.DataFrame, target_col: str = TARGET_COL):
        df = train_df.copy()
        y = df[target_col].values

        # 1. Cause rareness map (train only)
        cause_counts = df["event_cause"].value_counts()
        rare = set(cause_counts[cause_counts < MIN_CAUSE_COUNT].index)
        self._cause_map = rare
        df["event_cause"] = df["event_cause"].apply(
            lambda x: "other" if x in rare else x
        )

        # 2. Geo KMeans on train lat/lon
        coords = df[["lat", "lon"]].values
        self._geo_kmeans = KMeans(n_clusters=GEO_N_CLUSTERS, random_state=42, n_init=10)
        df["geo_cluster"] = self._geo_kmeans.fit_predict(coords)

        # 3. Frequency encoding (NOT target encoding) for high-cardinality categoricals.
        #    Value = log1p(event count for that category in train).
        #    This encodes "how active is this corridor/zone/station" without
        #    reconstructing the label.
        for col in ["corridor", "zone", "gba_identifier", "police_station"]:
            freq = df[col].value_counts().to_dict()
            overall = np.log1p(len(df) / max(df[col].nunique(), 1))  # fallback
            self._freq_encoders[col] = (
                {k: np.log1p(v) for k, v in freq.items()},
                overall
            )
            df[f"{col}_freq"] = df[col].map(
                {k: np.log1p(v) for k, v in freq.items()}
            ).fillna(overall)

        # 4. dup_cluster_size: compute on train only, freeze for val/test
        df["lat_r2"] = df["lat"].round(2)
        df["lon_r2"] = df["lon"].round(2)
        cluster_counts = df.groupby(["lat_r2", "lon_r2", "event_cause"]).size().to_dict()
        self._dup_cluster_map = cluster_counts
        df["dup_cluster_size"] = df.apply(
            lambda r: cluster_counts.get((r["lat_r2"], r["lon_r2"], r["event_cause"]), 1),
            axis=1
        )
        df = df.drop(columns=["lat_r2", "lon_r2"])

        df = _build_all_features(df)
        df = _drop_raw(df)
        X = df.drop(columns=[target_col, SECONDARY_TARGET], errors="ignore")
        self._train_cols = X.columns.tolist()
        self._fitted = True
        return X, y

    # ─────────────────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame, target_col: str = TARGET_COL):
        df = df.copy()
        y = df[target_col].values if target_col in df.columns else None

        df["event_cause"] = df["event_cause"].apply(
            lambda x: "other" if x in self._cause_map else x
        )

        coords = df[["lat", "lon"]].values
        df["geo_cluster"] = self._geo_kmeans.predict(coords)

        # Frequency encoding (frozen from train)
        for col, (enc, overall) in self._freq_encoders.items():
            df[f"{col}_freq"] = df[col].map(enc).fillna(overall)

        # dup_cluster_size: look up from train-computed map (no future leakage)
        df["lat_r2"] = df["lat"].round(2)
        df["lon_r2"] = df["lon"].round(2)
        df["dup_cluster_size"] = df.apply(
            lambda r: self._dup_cluster_map.get(
                (r["lat_r2"], r["lon_r2"], r["event_cause"]), 1
            ),
            axis=1
        )
        df = df.drop(columns=["lat_r2", "lon_r2"])

        df = _build_all_features(df)
        df = _drop_raw(df)
        X = df.drop(columns=[target_col, SECONDARY_TARGET], errors="ignore")
        X = X.reindex(columns=self._train_cols, fill_value=0)
        return X, y

    def transform_secondary(self, df: pd.DataFrame):
        """Return (X, y) for the road-closure target using the same frozen features."""
        df2 = df.copy()
        y2 = df2[SECONDARY_TARGET].astype(int).values if SECONDARY_TARGET in df2.columns else None
        X, _ = self.transform(df2, target_col=TARGET_COL)
        return X, y2


# ─────────────────────────────────────────────────────────────────────────────
def _hour_bucket(hour: int) -> str:
    for name, (lo, hi) in HOUR_BUCKETS.items():
        if lo <= hour <= hi:
            return name
    return "unknown"


def _build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df["event_type_planned"] = (df["event_type"] == "planned").astype(int)
    df["requires_road_closure_int"] = df["requires_road_closure_bool"].astype(int)
    df["is_corridor"] = (df["corridor"].str.lower() != "non-corridor").astype(int)
    df["has_junction"] = df["has_junction"].fillna(0).astype(int)

    dt = df["start_datetime"].dt
    df["hour"]         = dt.hour
    df["day_of_week"]  = dt.dayofweek
    df["month"]        = dt.month
    df["is_weekend"]   = (dt.dayofweek >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].apply(lambda h: int(h in PEAK_HOURS))
    df["hour_bucket"]  = df["hour"].apply(_hour_bucket)

    df = pd.get_dummies(df, columns=["event_cause"], prefix="cause", dtype=int)
    df = pd.get_dummies(df, columns=["veh_type"],    prefix="veh",   dtype=int)
    df = pd.get_dummies(df, columns=["hour_bucket"], prefix="hbkt",  dtype=int)

    return df


def _drop_raw(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [
        "event_type", "corridor", "zone", "gba_identifier", "police_station",
        "junction", "start_datetime",
        "veh_no", "description", "address",
    ]
    return df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
def build_features(train_df, val_df=None, test_df=None, target_col=TARGET_COL):
    """Convenience wrapper. Returns (builder, X_tr, y_tr [, X_v, y_v, X_t, y_t])."""
    builder = FeatureBuilder()
    X_train, y_train = builder.fit_transform(train_df, target_col=target_col)
    results = [builder, X_train, y_train]
    for df in [val_df, test_df]:
        if df is not None:
            X, y = builder.transform(df, target_col=target_col)
            results += [X, y]
    return tuple(results)
