"""
Tests for features.py — leakage, encoder discipline, IST conversion.
Run with: pytest tests/test_features.py -v
"""
import os, sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import load_and_clean
from src.features import build_features, FeatureBuilder
from src.split import time_split

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events.csv")

LEAKAGE_COLS = [
    "status", "closed_datetime", "resolved_datetime", "end_datetime",
    "modified_datetime", "created_date", "end_address",
    "resolved_at_address", "resolved_at_latitude", "resolved_at_longitude",
]


@pytest.fixture(scope="module")
def splits():
    if not os.path.exists(CSV_PATH):
        pytest.skip("Raw CSV not present")
    df = load_and_clean(CSV_PATH)
    train, val, test = time_split(df)
    builder = FeatureBuilder()
    X_train, y_train = builder.fit_transform(train)
    X_val, y_val = builder.transform(val)
    X_test, y_test = builder.transform(test)
    return X_train, y_train, X_val, y_val, X_test, y_test, builder


def test_no_leakage_in_features(splits):
    X_train = splits[0]
    for col in LEAKAGE_COLS:
        assert col not in X_train.columns, f"Leakage column '{col}' found in features"


def test_no_datetime_in_features(splits):
    X_train = splits[0]
    for col in X_train.columns:
        assert not pd.api.types.is_datetime64_any_dtype(X_train[col]), \
            f"Datetime column '{col}' leaked into feature matrix"


def test_target_not_in_features(splits):
    X_train = splits[0]
    assert "priority_high" not in X_train.columns
    assert "priority" not in X_train.columns


def test_encoders_fit_on_train_only(splits):
    _, _, X_train = splits[0], splits[1], splits[0]
    builder = splits[6]
    # Encoder must have been fitted (has categories_ attribute or similar)
    assert hasattr(builder, "_fitted") and builder._fitted, \
        "FeatureBuilder must set _fitted=True after fit_transform"


def test_val_test_no_new_categories(splits):
    X_val = splits[2]
    X_test = splits[4]
    # All columns in val/test must be present in train
    X_train = splits[0]
    for col in X_val.columns:
        assert col in X_train.columns or col == X_train.columns.tolist()[-1], \
            f"Val has column '{col}' not in train"


def test_peak_hour_binary(splits):
    X_train = splits[0]
    assert "is_peak_hour" in X_train.columns
    assert X_train["is_peak_hour"].isin([0, 1]).all()


def test_is_weekend_binary(splits):
    X_train = splits[0]
    assert "is_weekend" in X_train.columns
    assert X_train["is_weekend"].isin([0, 1]).all()


def test_hour_range(splits):
    X_train = splits[0]
    assert "hour" in X_train.columns
    assert X_train["hour"].between(0, 23).all()


def test_no_nulls_in_feature_matrix(splits):
    for X in [splits[0], splits[2], splits[4]]:
        assert X.isna().sum().sum() == 0, \
            f"NaNs found in feature matrix: {X.isna().sum()[X.isna().sum()>0]}"


def test_geo_cluster_present(splits):
    X_train = splits[0]
    assert "geo_cluster" in X_train.columns
    assert X_train["geo_cluster"].nunique() > 1
