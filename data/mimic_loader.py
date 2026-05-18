"""
MIMIC-IV Data Loader
--------------------
Maps MIMIC-IV CSV files to the EDI feature schema:
  HR, RR, SBP, Temp (C), SpO2, LOC (GCS-derived), Age

HOW TO GET THE DATA:
  1. Go to https://physionet.org/content/mimiciv/
  2. Complete CITI ethics training (~2 hrs, free)
  3. Sign the data use agreement
  4. Download and unzip — you'll get a folder like mimic-iv-2.2/
  5. Run: python pipeline.py --source mimic --mimic-path /path/to/mimic-iv-2.2

FILES USED (from MIMIC-IV):
  hosp/patients.csv     — demographics (age, death date)
  icu/icustays.csv      — ICU stay records
  icu/chartevents.csv   — vital sign measurements (large file, ~30 GB)

MIMIC-IV itemids used:
  220045  Heart Rate
  220210  Respiratory Rate
  220179  Systolic BP (non-invasive)
  220050  Systolic BP (arterial)
  223762  Temperature (Celsius)
  223761  Temperature (Fahrenheit) — converted to C
  220277  SpO2
  220739  GCS Eye Opening
  223900  GCS Verbal Response
  223901  GCS Motor Response
"""

import os
import pandas as pd
import numpy as np

VITALS_ITEMIDS = {
    "hr":   [220045],
    "rr":   [220210],
    "sbp":  [220179, 220050],
    "temp": [223762, 223761],  # C first, then F
    "spo2": [220277],
    "gcs":  [220739, 223900, 223901],  # eye + verbal + motor = total GCS
}

VALID_RANGES = {
    "hr":   (20,  250),
    "rr":   (2,   60),
    "sbp":  (40,  260),
    "temp": (34,  42),
    "spo2": (50,  100),
}


def load_mimic(mimic_path: str, max_stays: int = 5000) -> pd.DataFrame:
    """
    Loads MIMIC-IV and returns a DataFrame matching the EDI feature schema.

    Parameters
    ----------
    mimic_path : str
        Root folder of MIMIC-IV (contains hosp/ and icu/ subdirectories)
    max_stays : int
        Cap on ICU stays to load (chartevents is huge — use this for dev)
    """
    print(f"[MIMIC] Loading from {mimic_path}")

    # ── Patients ─────────────────────────────────────────────────
    patients = pd.read_csv(
        os.path.join(mimic_path, "hosp", "patients.csv"),
        usecols=["subject_id", "anchor_age", "dod"],
    )
    patients = patients.rename(columns={"anchor_age": "age"})
    patients["died"] = patients["dod"].notna().astype(int)

    # ── ICU stays ────────────────────────────────────────────────
    icustays = pd.read_csv(
        os.path.join(mimic_path, "icu", "icustays.csv"),
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
    )
    icustays["intime"]  = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

    if max_stays:
        icustays = icustays.head(max_stays)

    stay_ids = icustays["stay_id"].tolist()
    print(f"[MIMIC] Processing {len(stay_ids)} ICU stays")

    # ── Chartevents (vitals) ──────────────────────────────────────
    all_itemids = [i for ids in VITALS_ITEMIDS.values() for i in ids]

    print("[MIMIC] Reading chartevents (this may take a minute on the full dataset)...")
    chartevents = pd.read_csv(
        os.path.join(mimic_path, "icu", "chartevents.csv"),
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        dtype={"stay_id": "Int64", "itemid": int, "valuenum": float},
    )
    chartevents = chartevents[
        chartevents["stay_id"].isin(stay_ids) &
        chartevents["itemid"].isin(all_itemids) &
        chartevents["valuenum"].notna()
    ]

    # ── Aggregate vitals per stay (median of first 24h readings) ─
    chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
    chartevents = chartevents.merge(
        icustays[["stay_id", "intime"]], on="stay_id", how="left"
    )
    chartevents["hours_in"] = (
        chartevents["charttime"] - chartevents["intime"]
    ).dt.total_seconds() / 3600
    chartevents = chartevents[chartevents["hours_in"].between(0, 24)]

    vitals_agg = {}

    for feat, itemids in VITALS_ITEMIDS.items():
        subset = chartevents[chartevents["itemid"].isin(itemids)].copy()

        if feat == "temp":
            # Convert Fahrenheit (itemid 223761) to Celsius
            f_mask = subset["itemid"] == 223761
            subset.loc[f_mask, "valuenum"] = (subset.loc[f_mask, "valuenum"] - 32) / 1.8

        if feat == "gcs":
            # Sum eye + verbal + motor per stay per charttime
            gcs = (
                subset.groupby(["stay_id", "charttime"])["valuenum"]
                .sum()
                .reset_index()
            )
            vitals_agg["gcs"] = gcs.groupby("stay_id")["valuenum"].median()
        else:
            lo, hi = VALID_RANGES.get(feat, (-np.inf, np.inf))
            subset = subset[subset["valuenum"].between(lo, hi)]
            vitals_agg[feat] = subset.groupby("stay_id")["valuenum"].median()

    vitals_df = pd.DataFrame(vitals_agg)

    # ── GCS → LOC (0=alert if GCS>=14, 1=altered if GCS<14) ─────
    if "gcs" in vitals_df.columns:
        vitals_df["loc"] = (vitals_df["gcs"] < 14).astype(int)
        vitals_df = vitals_df.drop(columns=["gcs"])

    # ── Merge with stay + patient info ───────────────────────────
    df = (
        icustays[["stay_id", "subject_id"]]
        .merge(vitals_df.reset_index(), on="stay_id", how="inner")
        .merge(patients[["subject_id", "age", "died"]], on="subject_id", how="left")
    )

    # ── Deterioration label: died in hospital OR ICU LOS > 7 days ─
    df = df.merge(icustays[["stay_id", "los"]], on="stay_id", how="left")
    df["deteriorated"] = ((df["died"] == 1) | (df["los"] > 7)).astype(int)

    # ── Final clean ──────────────────────────────────────────────
    df = df.rename(columns={"stay_id": "patient_id"})
    feature_cols = ["patient_id", "age", "hr", "rr", "sbp", "temp", "spo2", "loc", "deteriorated"]
    df = df[[c for c in feature_cols if c in df.columns]].dropna()

    # Clip ranges
    for feat, (lo, hi) in VALID_RANGES.items():
        if feat in df.columns:
            df[feat] = df[feat].clip(lo, hi)

    df["age"]  = df["age"].clip(18, 95).round(0).astype(int)
    df["loc"]  = df["loc"].astype(int)
    df["hr"]   = df["hr"].round(0).astype(int)
    df["rr"]   = df["rr"].round(0).astype(int)
    df["sbp"]  = df["sbp"].round(0).astype(int)
    df["temp"] = df["temp"].round(1)
    df["spo2"] = df["spo2"].round(1)

    prev = df["deteriorated"].mean() * 100
    print(f"[MIMIC] {len(df)} stays loaded | deteriorated: {df['deteriorated'].sum()} ({prev:.1f}%)")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python mimic_loader.py /path/to/mimic-iv-2.2")
        sys.exit(1)
    df = load_mimic(sys.argv[1])
    print(df.describe().round(2))
