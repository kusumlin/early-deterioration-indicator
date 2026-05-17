"""
Early Deterioration Indicator (EDI) — Main Demo
================================================
Pipeline:
  1. Generate synthetic ICU patient dataset
  2. Train EDI model (Naive Bayes risk curves + logistic regression)
  3. Evaluate on held-out test set (AUROC, classification report)
  4. Produce all visualizations
  5. Score example individual patients
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from data.generate_data import generate_dataset
from edi import EDIScorer
from edi.risk_curves import FEATURES
from visualize import (
    plot_risk_curves,
    plot_edi_distribution,
    plot_roc_curve,
    plot_patient_dashboard,
)


def print_banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print_banner("EDI — Early Deterioration Indicator for ICU Patients")

    # ── 1. Data ──────────────────────────────────────────────────
    print("\n[1/5] Generating synthetic ICU dataset...")
    df = generate_dataset(n_patients=1200, seed=42)
    print(f"      {len(df)} patients | deteriorated: {df['deteriorated'].sum()} "
          f"({df['deteriorated'].mean()*100:.1f}%)")

    X = df[FEATURES]
    y = df["deteriorated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ── 2. Train ─────────────────────────────────────────────────
    print("\n[2/5] Training EDI model...")
    scorer = EDIScorer()
    scorer.fit(X_train, y_train)
    print("      Naive Bayes risk curves + Logistic Regression trained.")

    # ── 3. Evaluate ──────────────────────────────────────────────
    print("\n[3/5] Evaluating on test set...")
    results = scorer.score_patients(X_test)
    results["deteriorated"] = y_test.values

    auroc = roc_auc_score(y_test, results["edi_probability"])
    print(f"\n      AUROC: {auroc:.4f}")
    print("\n      Classification Report (threshold = 0.5):")
    preds = (results["edi_probability"] >= 0.5).astype(int)
    print(classification_report(y_test, preds, target_names=["Stable", "Deteriorating"]))

    risk_dist = results["risk_level"].value_counts()
    print("      Risk Level Distribution:")
    for level, count in risk_dist.items():
        print(f"        {level:<10}: {count}")

    # ── 4. Visualizations ────────────────────────────────────────
    print("\n[4/5] Generating visualizations → output/")
    os.makedirs("output", exist_ok=True)
    plot_risk_curves(scorer.risk_model)
    plot_edi_distribution(results)
    plot_roc_curve(results)

    # ── 5. Single-patient scoring ─────────────────────────────────
    print("\n[5/5] Single-patient scoring examples:")

    example_patients = [
        {
            "name": "Patient A — critically ill",
            "vitals": {"hr": 132, "rr": 28, "sbp": 80, "temp": 38.9, "spo2": 87.0, "loc": 1, "age": 74},
        },
        {
            "name": "Patient B — borderline",
            "vitals": {"hr": 105, "rr": 22, "sbp": 100, "temp": 37.8, "spo2": 93.0, "loc": 0, "age": 58},
        },
        {
            "name": "Patient C — stable",
            "vitals": {"hr": 75, "rr": 14, "sbp": 122, "temp": 37.0, "spo2": 98.0, "loc": 0, "age": 45},
        },
    ]

    for i, patient in enumerate(example_patients, start=1):
        result = scorer.score_single(patient["vitals"])
        print(f"\n  {patient['name']}")
        print(f"    EDI Probability : {result['edi_probability']:.4f}")
        print(f"    Risk Level      : {result['risk_level']}")
        top_contrib = sorted(result["feature_contributions"].items(),
                             key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f"    Top drivers     : " +
              ", ".join(f"{k}({v:+.3f})" for k, v in top_contrib))
        plot_patient_dashboard(result, patient_id=i,
                               save_path=f"output/patient_{i}_dashboard.png")

    print_banner("Done — all outputs saved to output/")


if __name__ == "__main__":
    main()
