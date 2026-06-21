"""
split.py — Time-based train/val/test split.

NO random splitting — this is temporal data. Future events must not appear
in training. Boundaries defined in config.py.
"""
import pandas as pd
from src.config import TRAIN_END, VAL_END


def time_split(df: pd.DataFrame):
    """
    Returns (train_df, val_df, test_df) with non-overlapping time windows.
    Asserts ordering: train.max < val.min, val.max < test.min.
    """
    df = df.sort_values("start_datetime").copy()

    train_end = pd.Timestamp(TRAIN_END, tz="Asia/Kolkata")
    val_end   = pd.Timestamp(VAL_END,   tz="Asia/Kolkata")

    train = df[df["start_datetime"] <= train_end].copy()
    val   = df[(df["start_datetime"] > train_end) & (df["start_datetime"] <= val_end)].copy()
    test  = df[df["start_datetime"] > val_end].copy()

    # Sanity checks — fail loud if windows overlap
    assert len(train) > 0, "Train split is empty — check TRAIN_END in config.py"
    assert len(val)   > 0, "Val split is empty — check VAL_END in config.py"
    assert len(test)  > 0, "Test split is empty — check VAL_END in config.py"
    assert train["start_datetime"].max() < val["start_datetime"].min(), \
        "Train/val window overlap detected"
    assert val["start_datetime"].max() < test["start_datetime"].min(), \
        "Val/test window overlap detected"

    print(f"[split] Train: {len(train)} rows  "
          f"{train['start_datetime'].min().date()} → {train['start_datetime'].max().date()}")
    print(f"[split] Val:   {len(val)} rows  "
          f"{val['start_datetime'].min().date()} → {val['start_datetime'].max().date()}")
    print(f"[split] Test:  {len(test)} rows  "
          f"{test['start_datetime'].min().date()} → {test['start_datetime'].max().date()}")

    return train, val, test


if __name__ == "__main__":
    from src.config import CLEAN_PARQUET
    df = pd.read_parquet(CLEAN_PARQUET)
    train, val, test = time_split(df)
