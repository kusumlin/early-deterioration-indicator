"""
Clean layer — validates physiological ranges, flags and drops bad rows,
and standardises column types. Mirrors real-world clinical data QA.
"""

import pandas as pd

VALID_RANGES = {
    "hr":   (20,  250),
    "rr":   (2,   60),
    "sbp":  (40,  260),
    "temp": (32,  44),
    "spo2": (50,  100),
    "loc":  (0,   1),
    "age":  (0,   120),
}

REQUIRED_COLS = ["patient_id", "age", "hr", "rr", "sbp", "temp", "spo2", "loc"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    # Drop rows missing required vitals
    df = df.dropna(subset=REQUIRED_COLS)

    # Flag out-of-range values and remove
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in VALID_RANGES.items():
        in_range = df[col].between(lo, hi)
        bad = (~in_range).sum()
        if bad:
            print(f"[Clean] Dropping {bad} rows: {col} out of range [{lo}, {hi}]")
        mask &= in_range

    df = df[mask].copy()

    # Type coercion
    int_cols = ["patient_id", "age", "hr", "rr", "sbp", "loc"]
    for col in int_cols:
        df[col] = df[col].astype(int)
    df["temp"] = df["temp"].round(1)
    df["spo2"] = df["spo2"].round(1)

    removed = original_len - len(df)
    print(f"[Clean] {len(df)} records retained | {removed} removed")
    return df.reset_index(drop=True)
