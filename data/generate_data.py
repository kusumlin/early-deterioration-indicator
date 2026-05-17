"""
Generates synthetic ICU patient vitals dataset with ground-truth deterioration labels.
Features mirror clinical EDI inputs: HR, RR, SBP, Temp, SpO2, LOC, Age.
Deterioration (~15% prevalence) is seeded by clinically abnormal vitals.
"""

import numpy as np
import pandas as pd


def generate_dataset(n_patients: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Stable vital distributions (mean, std) from literature
    hr_stable   = rng.normal(80, 12, n_patients)
    rr_stable   = rng.normal(16, 3, n_patients)
    sbp_stable  = rng.normal(120, 15, n_patients)
    temp_stable = rng.normal(37.0, 0.5, n_patients)
    spo2_stable = rng.normal(97, 1.5, n_patients)
    loc_stable  = rng.choice([0, 1], size=n_patients, p=[0.92, 0.08])  # 0=alert, 1=altered
    age         = rng.normal(62, 15, n_patients).clip(18, 95)

    # Deteriorating patient perturbations (applied to ~15%)
    deteriorated = rng.random(n_patients) < 0.15
    hr   = np.where(deteriorated, hr_stable  + rng.normal(25, 8, n_patients), hr_stable)
    rr   = np.where(deteriorated, rr_stable  + rng.normal(8,  3, n_patients), rr_stable)
    sbp  = np.where(deteriorated, sbp_stable - rng.normal(25, 10, n_patients), sbp_stable)
    temp = np.where(deteriorated, temp_stable + rng.choice([-1, 1], n_patients) * rng.normal(1.2, 0.4, n_patients), temp_stable)
    spo2 = np.where(deteriorated, spo2_stable - rng.normal(6, 2, n_patients), spo2_stable)
    loc  = np.where(deteriorated, rng.choice([0, 1], n_patients, p=[0.4, 0.6]), loc_stable)

    # Clip to physiologically plausible ranges
    hr   = hr.clip(30, 220)
    rr   = rr.clip(4, 50)
    sbp  = sbp.clip(50, 220)
    temp = temp.clip(34.0, 42.0)
    spo2 = spo2.clip(60, 100)

    df = pd.DataFrame({
        "patient_id":  np.arange(1, n_patients + 1),
        "age":         age.round(0).astype(int),
        "hr":          hr.round(0).astype(int),
        "rr":          rr.round(0).astype(int),
        "sbp":         sbp.round(0).astype(int),
        "temp":        temp.round(1),
        "spo2":        spo2.round(1),
        "loc":         loc.astype(int),
        "deteriorated": deteriorated.astype(int),
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("patients.csv", index=False)
    print(f"Generated {len(df)} patients | deteriorated: {df['deteriorated'].sum()} ({df['deteriorated'].mean()*100:.1f}%)")
    print(df.describe().round(2))
