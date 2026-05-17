-- ============================================================
-- EDI Analytical Queries  (SQLite compatible)
-- Run these against edi.db after pipeline.py has executed
-- ============================================================


-- 1. Overall risk level breakdown
SELECT
    risk_level,
    COUNT(*)                            AS n_patients,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM edi_scores
GROUP BY risk_level
ORDER BY CASE risk_level
    WHEN 'CRITICAL' THEN 1
    WHEN 'HIGH'     THEN 2
    WHEN 'MODERATE' THEN 3
    ELSE 4 END;


-- 2. Average vitals by risk tier — useful for clinical benchmarking
SELECT
    s.risk_level,
    ROUND(AVG(p.hr),   1) AS avg_hr,
    ROUND(AVG(p.rr),   1) AS avg_rr,
    ROUND(AVG(p.sbp),  1) AS avg_sbp,
    ROUND(AVG(p.temp), 2) AS avg_temp,
    ROUND(AVG(p.spo2), 1) AS avg_spo2,
    ROUND(AVG(p.age),  1) AS avg_age,
    ROUND(AVG(s.edi_probability), 3) AS avg_edi
FROM edi_scores s
JOIN patients p USING (patient_id)
GROUP BY s.risk_level
ORDER BY avg_edi DESC;


-- 3. CRITICAL patients with their top destabilising vital (max absolute log-ratio)
SELECT
    s.patient_id,
    p.age,
    p.hr, p.rr, p.sbp, p.spo2,
    ROUND(s.edi_probability, 3)  AS edi_prob,
    CASE
        WHEN ABS(s.lr_spo2) >= ABS(s.lr_hr)
         AND ABS(s.lr_spo2) >= ABS(s.lr_rr)
         AND ABS(s.lr_spo2) >= ABS(s.lr_sbp)
         AND ABS(s.lr_spo2) >= ABS(s.lr_loc)  THEN 'SpO2'
        WHEN ABS(s.lr_loc)  >= ABS(s.lr_hr)
         AND ABS(s.lr_loc)  >= ABS(s.lr_rr)
         AND ABS(s.lr_loc)  >= ABS(s.lr_sbp)  THEN 'LOC'
        WHEN ABS(s.lr_rr)   >= ABS(s.lr_hr)
         AND ABS(s.lr_rr)   >= ABS(s.lr_sbp)  THEN 'RR'
        WHEN ABS(s.lr_sbp)  >= ABS(s.lr_hr)   THEN 'SBP'
        ELSE 'HR'
    END AS top_driver
FROM edi_scores s
JOIN patients p USING (patient_id)
WHERE s.risk_level = 'CRITICAL'
ORDER BY s.edi_probability DESC
LIMIT 20;


-- 4. Model performance — predicted vs actual deterioration
SELECT
    s.risk_level,
    COUNT(*)                                         AS n_total,
    SUM(p.deteriorated)                              AS n_true_deteriorated,
    ROUND(SUM(p.deteriorated) * 100.0 / COUNT(*), 1) AS pct_true_deteriorated
FROM edi_scores s
JOIN patients p USING (patient_id)
GROUP BY s.risk_level
ORDER BY pct_true_deteriorated DESC;


-- 5. Age-group risk stratification
SELECT
    CASE
        WHEN p.age < 40  THEN '18-39'
        WHEN p.age < 60  THEN '40-59'
        WHEN p.age < 75  THEN '60-74'
        ELSE '75+'
    END AS age_group,
    COUNT(*)                                              AS n_patients,
    ROUND(AVG(s.edi_probability), 3)                      AS avg_edi,
    SUM(CASE WHEN s.risk_level = 'CRITICAL' THEN 1 ELSE 0 END) AS n_critical
FROM edi_scores s
JOIN patients p USING (patient_id)
GROUP BY age_group
ORDER BY avg_edi DESC;


-- 6. Audit log — pipeline run history
SELECT
    run_id,
    run_at,
    n_patients,
    n_critical,
    n_high,
    n_moderate,
    n_low
FROM audit_log
ORDER BY run_id DESC;
