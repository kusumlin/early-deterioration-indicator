"""
Store layer — writes cleaned + scored data into SQLite (acts as our data warehouse).
Creates three tables: patients (raw), edi_scores (model output), audit_log (run metadata).
"""

import sqlite3
import pandas as pd
from datetime import datetime


DB_PATH = "edi.db"


def store_to_db(patients: pd.DataFrame, scores: pd.DataFrame, db_path: str = DB_PATH):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Schema
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id  INTEGER PRIMARY KEY,
            age         INTEGER,
            hr          INTEGER,
            rr          INTEGER,
            sbp         INTEGER,
            temp        REAL,
            spo2        REAL,
            loc         INTEGER,
            deteriorated INTEGER
        );

        CREATE TABLE IF NOT EXISTS edi_scores (
            patient_id      INTEGER PRIMARY KEY,
            edi_probability REAL,
            composite_score REAL,
            risk_level      TEXT,
            lr_hr           REAL,
            lr_rr           REAL,
            lr_sbp          REAL,
            lr_temp         REAL,
            lr_spo2         REAL,
            lr_loc          REAL,
            lr_age          REAL,
            scored_at       TEXT
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at      TEXT,
            n_patients  INTEGER,
            n_critical  INTEGER,
            n_high      INTEGER,
            n_moderate  INTEGER,
            n_low       INTEGER
        );

        CREATE TABLE IF NOT EXISTS vitals_log (
            log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      INTEGER,
            hr              INTEGER,
            rr              INTEGER,
            sbp             INTEGER,
            temp            REAL,
            spo2            REAL,
            loc             INTEGER,
            edi_probability REAL,
            risk_level      TEXT,
            recorded_at     TEXT,
            recorded_by     TEXT
        );

        CREATE TABLE IF NOT EXISTS alerts (
            alert_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      INTEGER,
            old_risk_level  TEXT,
            new_risk_level  TEXT,
            edi_probability REAL,
            triggered_at    TEXT,
            acknowledged    INTEGER DEFAULT 0,
            acknowledged_by TEXT,
            acknowledged_at TEXT
        );
    """)

    # Upsert patients
    patients_cols = ["patient_id", "age", "hr", "rr", "sbp", "temp", "spo2", "loc", "deteriorated"]
    missing = [c for c in patients_cols if c not in patients.columns]
    write_cols = [c for c in patients_cols if c not in missing]
    patients[write_cols].to_sql("patients", con, if_exists="replace", index=False)

    # Upsert scores
    now = datetime.utcnow().isoformat()
    scores = scores.copy()
    scores["scored_at"] = now
    score_cols = ["patient_id", "edi_probability", "composite_score", "risk_level",
                  "lr_hr", "lr_rr", "lr_sbp", "lr_temp", "lr_spo2", "lr_loc", "lr_age", "scored_at"]
    available = [c for c in score_cols if c in scores.columns]
    scores[available].to_sql("edi_scores", con, if_exists="replace", index=False)

    # Audit log
    risk_counts = scores["risk_level"].value_counts().to_dict()
    cur.execute("""
        INSERT INTO audit_log (run_at, n_patients, n_critical, n_high, n_moderate, n_low)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (now, len(scores),
          risk_counts.get("CRITICAL", 0),
          risk_counts.get("HIGH", 0),
          risk_counts.get("MODERATE", 0),
          risk_counts.get("LOW", 0)))

    con.commit()
    con.close()
    print(f"[Store] Written to {db_path} → patients, edi_scores, audit_log")
