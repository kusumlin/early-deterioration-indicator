-- EDI Data Warehouse Schema
-- Three tables: raw vitals, model scores, pipeline audit log

CREATE TABLE IF NOT EXISTS patients (
    patient_id   INTEGER PRIMARY KEY,
    age          INTEGER,
    hr           INTEGER,
    rr           INTEGER,
    sbp          INTEGER,
    temp         REAL,
    spo2         REAL,
    loc          INTEGER,       -- 0=alert, 1=altered
    deteriorated INTEGER        -- ground truth label
);

CREATE TABLE IF NOT EXISTS edi_scores (
    patient_id       INTEGER PRIMARY KEY,
    edi_probability  REAL,      -- instability probability 0-1
    composite_score  REAL,      -- raw Naive Bayes log-ratio sum
    risk_level       TEXT,      -- LOW / MODERATE / HIGH / CRITICAL
    lr_hr            REAL,      -- per-feature log-ratio contributions
    lr_rr            REAL,
    lr_sbp           REAL,
    lr_temp          REAL,
    lr_spo2          REAL,
    lr_loc           REAL,
    lr_age           REAL,
    scored_at        TEXT       -- ISO timestamp
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
