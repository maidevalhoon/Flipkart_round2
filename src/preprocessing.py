"""
preprocessing.py — Load and clean the raw event CSV.

Outputs a cleaned DataFrame with:
- Geo-filtered rows (Bengaluru bbox, no 0/0)
- Binary target `priority_high` (1=High, 0=Low)
- Normalised categorical columns
- Parsed start_datetime (UTC → IST-aware)
- Post-event leakage columns removed
"""
import pandas as pd
import numpy as np
import pyarrow
from src.config import GEO_BBOX, DROP_COLS


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[preprocessing] Loaded {len(df)} rows × {len(df.columns)} cols")

    # ── 1. Drop dead-weight columns ──────────────────────────────────────────
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # ── 2. Target: priority → binary ─────────────────────────────────────────
    df = df[df["priority"].notna() & (df["priority"].str.strip() != "")].copy()
    df["priority_high"] = (df["priority"].str.strip().str.lower() == "high").astype(int)
    df = df.drop(columns=["priority"])

    # ── 3. Parse start_datetime; convert UTC → IST ───────────────────────────
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df["start_datetime"] = df["start_datetime"].dt.tz_convert("Asia/Kolkata")
    df = df[df["start_datetime"].notna()].copy()

    # ── 4. Lat/lon → float; geo-filter ───────────────────────────────────────
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.drop(columns=["latitude", "longitude"], errors="ignore")

    pre = len(df)
    df = df[
        df["lat"].between(GEO_BBOX["lat_min"], GEO_BBOX["lat_max"]) &
        df["lon"].between(GEO_BBOX["lon_min"], GEO_BBOX["lon_max"])
    ].copy()
    print(f"[preprocessing] Geo filter removed {pre - len(df)} rows "
          f"({pre - len(df)}/{pre} = {(pre-len(df))/pre*100:.1f}%)")

    # ── 5. event_type ─────────────────────────────────────────────────────────
    df["event_type"] = df["event_type"].str.strip().str.lower()

    # ── 6. event_cause: lowercase, strip, merge casing dupes ─────────────────
    df["event_cause"] = (
        df["event_cause"]
        .str.strip()
        .str.lower()
        .replace({"fog / low visibility": "fog_low_visibility", "test_demo": "other"})
    )

    # ── 7. requires_road_closure → bool (kept as BOTH feature AND secondary target)
    df["requires_road_closure_bool"] = (
        df["requires_road_closure"]
        .astype(str).str.strip().str.upper() == "TRUE"
    )
    df = df.drop(columns=["requires_road_closure"], errors="ignore")
    # Note: requires_road_closure_bool remains in df as:
    #   - a feature for the priority model (known at report time)
    #   - the target for the road-closure model (Model 2)

    # ── 8. corridor: fill missing → "unknown" ────────────────────────────────
    df["corridor"] = df["corridor"].fillna("unknown").str.strip()

    # ── 9. zone / gba_identifier: fill missing ───────────────────────────────
    df["zone"] = df["zone"].fillna("unknown").str.strip()
    df["gba_identifier"] = df["gba_identifier"].fillna("unknown").str.strip()

    # ── 10. police_station: strip ────────────────────────────────────────────
    df["police_station"] = df["police_station"].fillna("unknown").str.strip()

    # ── 11. junction: missing flag + raw value ────────────────────────────────
    df["has_junction"] = df["junction"].notna().astype(int)
    df["junction"] = df["junction"].fillna("none").str.strip()

    # ── 12. veh_type: fill missing → "none" ──────────────────────────────────
    df["veh_type"] = df["veh_type"].fillna("none").str.strip().str.lower()

    # ── 13. dup_cluster_size: count repeat (lat-round2, lon-round2, cause) ───
    df["lat_r2"] = df["lat"].round(2)
    df["lon_r2"] = df["lon"].round(2)
    dc = df.groupby(["lat_r2", "lon_r2", "event_cause"])["id"].transform("count")
    df["dup_cluster_size"] = dc.fillna(1).astype(int)
    df = df.drop(columns=["lat_r2", "lon_r2"])

    # ── 14. Drop remaining string ID / audit columns ──────────────────────────
    for col in ["id", "veh_no", "description", "address", "map_file",
                "direction", "route_path", "assigned_to_police_id",
                "citizen_accident_id", "police_station_id"]:
        df = df.drop(columns=[col], errors="ignore")

    print(f"[preprocessing] Clean output: {len(df)} rows × {len(df.columns)} cols")
    print(f"[preprocessing] Target balance — High: {df['priority_high'].sum()} "
          f"| Low: {(df['priority_high']==0).sum()}")
    return df.reset_index(drop=True)


def save_clean(df: pd.DataFrame, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[preprocessing] Saved clean parquet → {path}")


if __name__ == "__main__":
    from src.config import RAW_CSV, CLEAN_PARQUET
    df = load_and_clean(RAW_CSV)
    save_clean(df, CLEAN_PARQUET)
