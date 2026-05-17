"""
Alert Engine
------------
Called every time a nurse submits updated vitals for a patient.
  1. Rescores the patient using the EDI model
  2. Compares new risk level to previous risk level
  3. If risk escalated → writes an alert to the DB
  4. Logs vitals + score to vitals_log (time-series)
"""

import sqlite3
from datetime import datetime

import pandas as pd

from .risk_curves import FEATURES

DB_PATH = "edi.db"

RISK_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


def update_vitals_and_alert(
    scorer,
    patient_id: int,
    vitals: dict,
    nurse_name: str = "Nurse",
    db_path: str = DB_PATH,
) -> dict:
    """
    Main entry point called from the Nurse Portal.

    Returns a dict with:
      - edi_probability
      - risk_level
      - alert_triggered (bool)
      - old_risk_level
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Ensure tables exist
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS vitals_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER, hr INTEGER, rr INTEGER, sbp INTEGER,
            temp REAL, spo2 REAL, loc INTEGER,
            edi_probability REAL, risk_level TEXT,
            recorded_at TEXT, recorded_by TEXT
        );
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER, old_risk_level TEXT, new_risk_level TEXT,
            edi_probability REAL, triggered_at TEXT,
            acknowledged INTEGER DEFAULT 0,
            acknowledged_by TEXT, acknowledged_at TEXT
        );
    """)

    # Get previous risk level
    row = cur.execute(
        "SELECT risk_level FROM edi_scores WHERE patient_id = ?", (patient_id,)
    ).fetchone()
    old_risk = row[0] if row else "LOW"

    # Rescore
    result = scorer.score_single(vitals)
    new_risk = result["risk_level"]
    edi_prob = result["edi_probability"]
    now = datetime.utcnow().isoformat()

    # Log vitals
    cur.execute("""
        INSERT INTO vitals_log
            (patient_id, hr, rr, sbp, temp, spo2, loc, edi_probability, risk_level, recorded_at, recorded_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, vitals["hr"], vitals["rr"], vitals["sbp"],
          vitals["temp"], vitals["spo2"], vitals["loc"],
          edi_prob, new_risk, now, nurse_name))

    # Update current score in edi_scores
    cur.execute("""
        INSERT INTO edi_scores (patient_id, edi_probability, risk_level, scored_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(patient_id) DO UPDATE SET
            edi_probability = excluded.edi_probability,
            risk_level      = excluded.risk_level,
            scored_at       = excluded.scored_at
    """, (patient_id, edi_prob, new_risk, now))

    # Update current vitals in patients table
    cur.execute("""
        UPDATE patients SET hr=?, rr=?, sbp=?, temp=?, spo2=?, loc=?
        WHERE patient_id=?
    """, (vitals["hr"], vitals["rr"], vitals["sbp"],
          vitals["temp"], vitals["spo2"], vitals["loc"], patient_id))

    # Fire alert if risk escalated
    alert_triggered = RISK_RANK.get(new_risk, 0) > RISK_RANK.get(old_risk, 0)
    if alert_triggered:
        cur.execute("""
            INSERT INTO alerts (patient_id, old_risk_level, new_risk_level, edi_probability, triggered_at)
            VALUES (?, ?, ?, ?, ?)
        """, (patient_id, old_risk, new_risk, edi_prob, now))

    con.commit()
    con.close()

    return {
        "edi_probability": edi_prob,
        "risk_level": new_risk,
        "old_risk_level": old_risk,
        "alert_triggered": alert_triggered,
        "feature_contributions": result["feature_contributions"],
    }


def get_active_alerts(db_path: str = DB_PATH) -> pd.DataFrame:
    """Returns all unacknowledged alerts joined with patient info, newest first."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT
            a.alert_id,
            a.patient_id,
            p.age,
            p.hr, p.rr, p.sbp, p.spo2,
            a.old_risk_level,
            a.new_risk_level,
            ROUND(a.edi_probability, 3) AS edi_probability,
            a.triggered_at
        FROM alerts a
        JOIN patients p USING (patient_id)
        WHERE a.acknowledged = 0
        ORDER BY a.edi_probability DESC, a.triggered_at DESC
    """, con)
    con.close()
    return df


def acknowledge_alert(alert_id: int, doctor_name: str, db_path: str = DB_PATH):
    con = sqlite3.connect(db_path)
    con.execute("""
        UPDATE alerts SET acknowledged=1, acknowledged_by=?, acknowledged_at=?
        WHERE alert_id=?
    """, (doctor_name, datetime.utcnow().isoformat(), alert_id))
    con.commit()
    con.close()


def get_vitals_history(patient_id: int, db_path: str = DB_PATH, limit: int = 20) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT recorded_at, hr, rr, sbp, temp, spo2, loc, edi_probability, risk_level, recorded_by
        FROM vitals_log
        WHERE patient_id = ?
        ORDER BY recorded_at DESC
        LIMIT ?
    """, con, params=(patient_id, limit))
    con.close()
    return df
