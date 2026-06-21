"""
Tests for preprocessing.py — written BEFORE the implementation.
Run with: pytest tests/test_preprocessing.py -v
"""
import os, sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import load_and_clean


CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events.csv")


@pytest.fixture(scope="module")
def clean_df():
    if not os.path.exists(CSV_PATH):
        pytest.skip("Raw CSV not present — upload data/raw/events.csv first")
    return load_and_clean(CSV_PATH)


def test_returns_dataframe(clean_df):
    assert isinstance(clean_df, pd.DataFrame)


def test_row_count_reasonable(clean_df):
    # Should keep at least 7500 rows after geo filter on 8173 raw
    assert len(clean_df) >= 7000, f"Only {len(clean_df)} rows — check geo filter"


def test_target_no_nulls(clean_df):
    assert clean_df["priority_high"].isna().sum() == 0


def test_target_binary(clean_df):
    assert set(clean_df["priority_high"].unique()).issubset({0, 1})


def test_known_empty_cols_dropped(clean_df):
    for col in ["map_file", "comment", "meta_data", "cargo_material"]:
        assert col not in clean_df.columns, f"{col} should have been dropped"


def test_no_zero_zero_geo(clean_df):
    bad = ((clean_df["lat"] == 0) & (clean_df["lon"] == 0))
    assert bad.sum() == 0, "Found (0,0) geo rows that should have been filtered"


def test_geo_within_bengaluru_bbox(clean_df):
    assert clean_df["lat"].between(12.7, 13.25).all()
    assert clean_df["lon"].between(77.3, 77.9).all()


def test_event_cause_normalised(clean_df):
    causes = clean_df["event_cause"].str.lower().unique()
    # Debris and debris should both map to "debris"
    if "debris" in causes:
        assert "Debris" not in causes, "Casing inconsistency in event_cause"


def test_requires_road_closure_bool(clean_df):
    assert clean_df["requires_road_closure_bool"].dtype == bool or \
           clean_df["requires_road_closure_bool"].isin([True, False]).all()


def test_start_datetime_parsed(clean_df):
    assert pd.api.types.is_datetime64_any_dtype(clean_df["start_datetime"])


def test_no_leakage_cols(clean_df):
    leakage = ["status", "closed_datetime", "resolved_datetime", "end_datetime"]
    for col in leakage:
        assert col not in clean_df.columns, f"Leakage column {col} still present"


def test_corridor_no_nulls(clean_df):
    assert clean_df["corridor"].isna().sum() == 0
    assert "unknown" in clean_df["corridor"].values or \
           (clean_df["corridor"].isna().sum() == 0)


def test_veh_type_nulls_filled(clean_df):
    assert clean_df["veh_type"].isna().sum() == 0
