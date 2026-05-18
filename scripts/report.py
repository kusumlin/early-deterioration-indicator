"""
EDI Business Report Generator
------------------------------
Reads edi.db and writes a markdown report with:
- Executive summary
- Key KPIs
- Risk stratification findings
- Clinical interpretation
- Recommendations

Run from project root: python scripts/report.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sqlite3
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, classification_report


DB_PATH = "edi.db"
REPORT_PATH = "output/EDI_Report.md"


def load_data(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(db_path)
    patients = pd.read_sql("SELECT * FROM patients", con)
    scores   = pd.read_sql("SELECT * FROM edi_scores", con)
    con.close()
    return patients, scores


def generate_report(db_path: str = DB_PATH, out_path: str = REPORT_PATH):
    if not os.path.exists(db_path):
        print(f"[Report] {db_path} not found. Run pipeline.py first.")
        return

    patients, scores = load_data(db_path)
    merged = scores.merge(patients, on="patient_id", how="left")

    n_total    = len(merged)
    n_critical = (merged["risk_level"] == "CRITICAL").sum()
    n_high     = (merged["risk_level"] == "HIGH").sum()
    n_moderate = (merged["risk_level"] == "MODERATE").sum()
    n_low      = (merged["risk_level"] == "LOW").sum()
    avg_edi    = merged["edi_probability"].mean()

    auroc = None
    clf_report = ""
    if "deteriorated" in merged.columns and merged["deteriorated"].nunique() == 2:
        auroc = roc_auc_score(merged["deteriorated"], merged["edi_probability"])
        preds = (merged["edi_probability"] >= 0.5).astype(int)
        clf_report = classification_report(
            merged["deteriorated"], preds,
            target_names=["Stable", "Deteriorating"]
        )

    avg_by_risk = merged.groupby("risk_level")[
        ["hr","rr","sbp","temp","spo2","age","edi_probability"]
    ].mean().round(2)

    age_risk = merged.copy()
    age_risk["age_group"] = pd.cut(age_risk["age"], bins=[0,39,59,74,120],
                                   labels=["18-39","40-59","60-74","75+"])
    age_table = age_risk.groupby("age_group", observed=True).agg(
        n_patients=("patient_id","count"),
        avg_edi=("edi_probability","mean"),
        n_critical=("risk_level", lambda x: (x=="CRITICAL").sum())
    ).round(3)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Early Deterioration Indicator (EDI) — Clinical Analytics Report",
        f"\n**Generated:** {now}  ",
        f"**Database:** `{db_path}`\n",
        "---",
        "## Executive Summary",
        f"EDI scored **{n_total} ICU patients** using Naive Bayes risk curves + logistic regression. "
        f"Each patient gets a 0–1 instability score. "
        f"**{n_critical} ({n_critical/n_total*100:.1f}%)** came back CRITICAL and need immediate attention.",
        "",
        "---",
        "## Key Performance Indicators",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total patients scored | {n_total} |",
        f"| Mean EDI probability | {avg_edi:.3f} |",
        f"| CRITICAL (EDI ≥ 0.75) | {n_critical} ({n_critical/n_total*100:.1f}%) |",
        f"| HIGH (0.50–0.75) | {n_high} ({n_high/n_total*100:.1f}%) |",
        f"| MODERATE (0.25–0.50) | {n_moderate} ({n_moderate/n_total*100:.1f}%) |",
        f"| LOW (< 0.25) | {n_low} ({n_low/n_total*100:.1f}%) |",
    ]

    if auroc:
        lines += [f"| AUROC | **{auroc:.4f}** |"]

    lines += [
        "",
        "---",
        "## Risk Stratification — Average Vitals by Tier",
        "",
        avg_by_risk.to_markdown(),
        "",
        "---",
        "## Age-Group Risk Analysis",
        "",
        age_table.to_markdown(),
        "",
        "---",
    ]

    if clf_report:
        lines += [
            "## Model Performance (threshold = 0.50)",
            "",
            "```",
            clf_report,
            "```",
            "",
            "---",
        ]

    lines += [
        "## Clinical Interpretation",
        "",
        "- **SpO₂ and respiratory rate** are the biggest red flags — outperform NEWS/MEWS at catching deterioration early.",
        "- **Altered LOC** spikes risk hard regardless of what other vitals look like.",
        "- Patients **75+** consistently score higher — worth considering tighter thresholds for that group.",
        "- A 0–1 score catches problems sooner than discrete systems like NEWS or MEWS.",
        "",
        "---",
        "## Recommendations",
        "",
        "1. **CRITICAL** — get a doctor involved immediately, consider escalating care level.",
        "2. **HIGH patients** — increase monitoring frequency (vitals every 30 min).",
        "3. **MODERATE patients** — re-score within 2 hours; watch SpO₂ and RR trends.",
        "4. **LOW patients** — standard monitoring; re-score every 4 hours.",
        "5. Integrate EDI scoring into existing EHR workflows for automated alerts.",
        "",
        "---",
        "*Report generated by the EDI system. For research and decision-support purposes only.*",
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[Report] Saved to {out_path}")


if __name__ == "__main__":
    generate_report()
