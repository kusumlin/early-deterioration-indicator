# Early Deterioration Indicator (EDI) for ICU Patients

Flags at-risk ICU patients early using Naive Bayes risk curves and logistic regression. Nurses update vitals, the model rescores in real time, and doctors get an alert queue sorted by risk.

## How It Works

1. **Nurses** enter updated patient vitals (HR, RR, BP, Temp, SpO₂, LOC, Age)
2. The **EDI model** rescores the patient instantly (probability 0–1.0)
3. If risk escalates, an **alert fires automatically** to the doctor queue
4. **Doctors** see a prioritized list of patients sorted by EDI score and acknowledge alerts once reviewed

## Data Note

> **No real patient data is included here** — hospital data is subject to HIPAA and can't be shared publicly. Synthetic vitals are used instead, generated to match real ICU distributions from published research. For real data, check out [MIMIC-IV on PhysioNet](https://physionet.org/content/mimiciv/) (free access, requires ethics training).

## Stack

| Layer | Tools |
|-------|-------|
| ML Model | Naive Bayes (scikit-learn), Logistic Regression |
| Data Engineering | Python ETL pipeline, SQLite |
| Analytics | pandas, NumPy, SQL queries |
| Visualization | matplotlib, Streamlit dashboard |

## Quick Start

```bash
# 1. Set up environment
bash setup.sh
source .venv/bin/activate

# 2. Run ETL pipeline (generates data, trains model, populates DB, writes report)
python pipeline.py

# 3. Launch dashboard
streamlit run dashboard.py
```

## Project Structure

```
EDI/
├── edi/                      # Core ML modules
│   ├── risk_curves.py        # Naive Bayes per-feature risk curves
│   ├── edi_scorer.py         # Logistic regression → EDI probability
│   └── alert_engine.py       # Vitals update + alert generation
├── pipeline/                 # ETL: ingest → clean → store
├── scripts/                  # Helper scripts
│   ├── main.py               # ML demo + visualizations
│   ├── visualize.py          # Plot generation
│   └── report.py             # Business report generator
├── sql/                      # Schema + analytical queries
├── dashboard.py              # Streamlit dashboard (Nurse + Doctor views)
└── pipeline.py               # End-to-end pipeline runner
```

## Model Performance (synthetic data)

| Metric | Value |
|--------|-------|
| AUROC | 0.999 |
| Accuracy | 99% |
| Precision (Deteriorating) | 97% |
| Recall (Deteriorating) | 96% |

*Note: Performance on real ICU data (e.g. MIMIC-IV) would realistically be 0.75–0.85 AUROC, consistent with published EDI literature.*

## References

- Johnson et al. (2020). MIMIC-IV. PhysioNet.
- Churpek et al. (2017). Multicenter Comparison of Early Warning Scores. *Critical Care Medicine.*
- Mao et al. (2018). Continual Prediction of Deterioration in General Wards. *Resuscitation.*
