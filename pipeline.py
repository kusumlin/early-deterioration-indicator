"""
EDI End-to-End Pipeline
------------------------
Runs the full DE/DS workflow:
  1. Ingest  — load raw patient data
  2. Clean   — validate and standardise
  3. Train   — fit EDI model on 75% of data
  4. Score   — score all patients
  5. Store   — write to SQLite (edi.db)
  6. Report  — generate business markdown report
"""

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

from pipeline import ingest, clean, store_to_db
from edi import EDIScorer
from edi.risk_curves import FEATURES
from scripts.report import generate_report


def run_pipeline():
    print("=" * 60)
    print("  EDI Pipeline Starting")
    print("=" * 60)

    # 1. Ingest
    raw = ingest(source="auto")

    # 2. Clean
    cleaned = clean(raw)

    # 3. Train/test split — train on 75%, score all
    X = cleaned[FEATURES]
    y = cleaned["deteriorated"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("[Train] Fitting EDI model...")
    scorer = EDIScorer()
    scorer.fit(X_train, y_train)
    print("[Train] Done.")

    # 4. Score all patients
    print("[Score] Scoring all patients...")
    scores = scorer.score_patients(X)
    scores.insert(0, "patient_id", cleaned["patient_id"].values)

    auroc = roc_auc_score(y, scores["edi_probability"])
    print(f"[Score] AUROC on full dataset: {auroc:.4f}")
    print(f"[Score] Risk distribution:\n{scores['risk_level'].value_counts().to_string()}")

    # 5. Store
    store_to_db(cleaned, scores)

    # 6. Report
    generate_report()

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("  - Database : edi.db")
    print("  - Report   : output/EDI_Report.md")
    print("  - Dashboard: streamlit run dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
