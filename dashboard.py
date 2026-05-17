"""
EDI Streamlit Dashboard
-----------------------
Run with:  streamlit run dashboard.py

Four tabs:
  1. Doctor Alerts  — live alert queue, sorted by EDI, acknowledge button
  2. Nurse Portal   — update patient vitals → triggers EDI rescore + alert
  3. Overview       — population KPIs and charts
  4. SQL Explorer   — run live SQL against edi.db
"""

import sqlite3
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

DB_PATH = "edi.db"
FEATURES = ["hr", "rr", "sbp", "temp", "spo2", "loc", "age"]

RISK_COLOR = {
    "LOW":      "#2ecc71",
    "MODERATE": "#f1c40f",
    "HIGH":     "#e67e22",
    "CRITICAL": "#e74c3c",
}
RISK_EMOJI = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}

st.set_page_config(
    page_title="EDI — ICU Early Deterioration Indicator",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("ICU Early Deterioration Indicator (EDI)")

# ── Shared model (cached so it only trains once per session) ─────────────────

@st.cache_resource
def load_model():
    from data.generate_data import generate_dataset
    from edi import EDIScorer
    from sklearn.model_selection import train_test_split

    df = generate_dataset(n_patients=1200, seed=42)
    X  = df[FEATURES]
    y  = df["deteriorated"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scorer = EDIScorer()
    scorer.fit(X_train, y_train)
    return scorer


def db_ok():
    return os.path.exists(DB_PATH)


def read_sql(query: str, params=()) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, con, params=params)
    con.close()
    return df


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_doctor, tab_nurse, tab_overview, tab_sql = st.tabs(
    ["🔴 Doctor Alerts", "🩺 Nurse Portal", "📊 Overview", "🗄️ SQL Explorer"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DOCTOR ALERTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_doctor:
    st.subheader("Active Patient Alerts")
    st.caption("Sorted by EDI probability — highest risk first. Auto-refreshes every 15 s.")

    doctor_name = st.text_input("Your name", value="Dr. Smith", key="doc_name")

    if not db_ok():
        st.warning("Run `python pipeline.py` first to populate the database.")
    else:
        @st.fragment(run_every="15s")
        def doctor_alerts_panel():
            from edi.alert_engine import get_active_alerts, acknowledge_alert, get_vitals_history

            alerts = get_active_alerts()

            if alerts.empty:
                st.success("No active alerts. All patients are stable.")
            else:
                st.error(f"**{len(alerts)} unacknowledged alert(s)**")

                for _, row in alerts.iterrows():
                    risk     = row["new_risk_level"]
                    color    = RISK_COLOR.get(risk, "gray")
                    emoji    = RISK_EMOJI.get(risk, "")
                    old_risk = row["old_risk_level"]

                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns([2, 3, 3, 2])

                        c1.markdown(
                            f"### {emoji} Patient {int(row['patient_id'])}\n"
                            f"**EDI: {row['edi_probability']:.3f}**"
                        )
                        c2.markdown(
                            f"**Risk:** `{old_risk}` → **:{color.replace('#','')}[{risk}]**\n\n"
                            f"Age: {int(row['age'])} | LOC: {'Altered' if row.get('spo2', 97) < 90 else 'Alert'}"
                        )
                        c3.markdown(
                            f"HR: **{int(row['hr'])}** bpm | "
                            f"RR: **{int(row['rr'])}** br/min\n\n"
                            f"SBP: **{int(row['sbp'])}** mmHg | "
                            f"SpO₂: **{row['spo2']}%**"
                        )
                        if c4.button("✅ Acknowledge", key=f"ack_{row['alert_id']}"):
                            acknowledge_alert(int(row["alert_id"]), doctor_name)
                            st.rerun()

                        with st.expander(f"Vitals history — Patient {int(row['patient_id'])}"):
                            history = get_vitals_history(int(row["patient_id"]))
                            if history.empty:
                                st.info("No vitals history yet.")
                            else:
                                history["recorded_at"] = pd.to_datetime(history["recorded_at"])
                                fig, axes = plt.subplots(1, 3, figsize=(12, 2.5))
                                for ax, (col_name, label, color_line) in zip(axes, [
                                    ("edi_probability", "EDI Probability", "#e74c3c"),
                                    ("hr",              "Heart Rate",      "#3498db"),
                                    ("spo2",            "SpO₂ (%)",        "#2ecc71"),
                                ]):
                                    ax.plot(history["recorded_at"], history[col_name],
                                            marker="o", color=color_line, linewidth=2)
                                    ax.set_title(label, fontsize=9)
                                    ax.tick_params(labelsize=7)
                                    ax.grid(True, alpha=0.3)
                                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
                                plt.tight_layout()
                                st.pyplot(fig, use_container_width=True)

            # All patients ranked by EDI
            st.divider()
            st.subheader("All ICU Patients — EDI Priority Queue")
            all_scores = read_sql("""
                SELECT s.patient_id, p.age, p.hr, p.rr, p.sbp, p.spo2,
                       ROUND(s.edi_probability,3) AS edi_probability, s.risk_level, s.scored_at
                FROM edi_scores s JOIN patients p USING (patient_id)
                ORDER BY s.edi_probability DESC
            """)

            def highlight_risk(row):
                c = RISK_COLOR.get(row["risk_level"], "#ffffff")
                return [f"background-color: {c}22" if col == "risk_level" else "" for col in row.index]

            st.dataframe(
                all_scores.style.apply(highlight_risk, axis=1),
                use_container_width=True,
                height=400,
            )

        doctor_alerts_panel()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — NURSE PORTAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_nurse:
    st.subheader("Update Patient Vitals")
    st.caption("Enter the latest vitals reading. EDI will be recalculated and an alert fired if risk escalates.")

    if not db_ok():
        st.warning("Run `python pipeline.py` first.")
    else:
        scorer = load_model()

        # Patient selector
        patient_list = read_sql(
            "SELECT patient_id, age, hr, rr, sbp, temp, spo2, loc FROM patients ORDER BY patient_id"
        )
        scores_now = read_sql("SELECT patient_id, risk_level, edi_probability FROM edi_scores")
        patient_list = patient_list.merge(scores_now, on="patient_id", how="left")

        col_sel, col_nurse = st.columns([3, 1])
        patient_options = {
            f"Patient {row.patient_id}  |  {RISK_EMOJI.get(row.risk_level,'⬜')} {row.risk_level}  |  EDI {row.edi_probability:.3f}": row.patient_id
            for _, row in patient_list.iterrows()
        }
        selected_label = col_sel.selectbox("Select Patient", list(patient_options.keys()))
        nurse_name = col_nurse.text_input("Nurse name", value="Nurse Johnson", key="nurse_name")
        patient_id = patient_options[selected_label]

        # Pre-fill with current vitals
        current = patient_list[patient_list["patient_id"] == patient_id].iloc[0]

        st.markdown("#### Enter New Vitals")
        c1, c2, c3, c4 = st.columns(4)
        hr   = c1.number_input("Heart Rate (bpm)",     20,  250, int(current.hr))
        rr   = c2.number_input("Resp. Rate (br/min)",  2,   60,  int(current.rr))
        sbp  = c3.number_input("Systolic BP (mmHg)",   40,  260, int(current.sbp))
        temp = c4.number_input("Temperature (°C)",     32.0, 44.0, float(current.temp), step=0.1)

        c5, c6, c7 = st.columns(3)
        spo2 = c5.number_input("SpO₂ (%)",             50.0, 100.0, float(current.spo2), step=0.5)
        loc  = c6.selectbox("Level of Consciousness",  [0, 1],
                             index=int(current["loc"]),
                             format_func=lambda x: "Alert (0)" if x == 0 else "Altered (1)")
        age  = c7.number_input("Age (years)",           0,    120,  int(current.age))

        if st.button("Submit Vitals", type="primary", use_container_width=True):
            from edi.alert_engine import update_vitals_and_alert

            vitals = {"hr": hr, "rr": rr, "sbp": sbp, "temp": temp,
                      "spo2": spo2, "loc": loc, "age": age}
            result = update_vitals_and_alert(scorer, patient_id, vitals, nurse_name)

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("EDI Probability", f"{result['edi_probability']:.4f}")
            m2.metric("Risk Level", result["risk_level"],
                      delta=f"from {result['old_risk_level']}" if result["old_risk_level"] != result["risk_level"] else None,
                      delta_color="inverse")

            if result["alert_triggered"]:
                m3.error(f"⚠️ ALERT FIRED\n{result['old_risk_level']} → {result['risk_level']}")
                st.error(
                    f"**Alert sent to Doctor queue!** "
                    f"Patient {patient_id} escalated from "
                    f"**{result['old_risk_level']}** to **{result['risk_level']}**."
                )
            else:
                m3.success("No escalation")

            # Feature contributions bar
            st.subheader("Which vitals are driving the score?")
            contribs = result["feature_contributions"]
            feat_labels = {"hr": "Heart Rate", "rr": "Resp Rate", "sbp": "Systolic BP",
                           "temp": "Temperature", "spo2": "SpO₂", "loc": "LOC", "age": "Age"}
            fig, ax = plt.subplots(figsize=(8, 3))
            keys = list(contribs.keys())
            vals = list(contribs.values())
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]
            bars = ax.barh([feat_labels[k] for k in keys], vals, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Log-Ratio  (positive = destabilising)")
            for bar, val in zip(bars, vals):
                ax.text(val + (0.1 if val >= 0 else -0.1),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.2f}", va="center",
                        ha="left" if val >= 0 else "right", fontsize=8)
            st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("ICU Population Overview")

    if not db_ok():
        st.warning("Run `python pipeline.py` first.")
    else:
        scores   = read_sql("SELECT * FROM edi_scores")
        patients = read_sql("SELECT * FROM patients")
        merged   = scores.merge(patients[["patient_id","deteriorated"]], on="patient_id", how="left")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Patients", len(scores))
        c2.metric("🔴 CRITICAL", int((scores["risk_level"] == "CRITICAL").sum()))
        c3.metric("🟠 HIGH",     int((scores["risk_level"] == "HIGH").sum()))
        c4.metric("🟡 MODERATE", int((scores["risk_level"] == "MODERATE").sum()))
        c5.metric("🟢 LOW",      int((scores["risk_level"] == "LOW").sum()))

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Risk Level Distribution**")
            counts = scores["risk_level"].value_counts().reindex(
                ["CRITICAL", "HIGH", "MODERATE", "LOW"], fill_value=0)
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(counts.index, counts.values,
                          color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"])
            for b in bars:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                        str(int(b.get_height())), ha="center", fontsize=9)
            ax.set_ylabel("Patients")
            ax.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig, use_container_width=True)

        with col_b:
            st.markdown("**EDI Probability Distribution**")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            if "deteriorated" in merged.columns:
                ax2.hist(merged[merged["deteriorated"]==0]["edi_probability"],
                         bins=25, alpha=0.65, color="#2ecc71", label="Stable", density=True)
                ax2.hist(merged[merged["deteriorated"]==1]["edi_probability"],
                         bins=25, alpha=0.65, color="#e74c3c", label="Deteriorating", density=True)
                ax2.legend(fontsize=8)
            else:
                ax2.hist(scores["edi_probability"], bins=25, color="#2c3e50")
            ax2.set_xlabel("EDI Probability")
            ax2.set_ylabel("Density")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2, use_container_width=True)

        st.divider()
        st.markdown("**Average Vitals by Risk Tier**")
        avg = read_sql("""
            SELECT s.risk_level,
                   ROUND(AVG(p.hr),1) AS avg_hr, ROUND(AVG(p.rr),1) AS avg_rr,
                   ROUND(AVG(p.sbp),1) AS avg_sbp, ROUND(AVG(p.spo2),1) AS avg_spo2,
                   ROUND(AVG(p.age),1) AS avg_age,
                   ROUND(AVG(s.edi_probability),3) AS avg_edi
            FROM edi_scores s JOIN patients p USING (patient_id)
            GROUP BY s.risk_level ORDER BY avg_edi DESC
        """)
        st.dataframe(avg, use_container_width=True)

        st.divider()
        n_alerts = 0
        try:
            n_alerts = read_sql("SELECT COUNT(*) AS n FROM alerts WHERE acknowledged=0").iloc[0]["n"]
        except Exception:
            pass
        alert_hist = pd.DataFrame()
        try:
            alert_hist = read_sql("""
                SELECT triggered_at, new_risk_level, COUNT(*) AS n
                FROM alerts GROUP BY DATE(triggered_at), new_risk_level
            """)
        except Exception:
            pass

        st.markdown(f"**Alert Summary** — {int(n_alerts)} unacknowledged alerts")
        if not alert_hist.empty:
            st.dataframe(alert_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SQL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_sql:
    st.subheader("Live SQL — edi.db")

    default_q = """SELECT s.risk_level,
       COUNT(*) AS n_patients,
       ROUND(AVG(s.edi_probability),3) AS avg_edi,
       ROUND(AVG(p.age),1) AS avg_age,
       ROUND(AVG(p.spo2),1) AS avg_spo2
FROM edi_scores s
JOIN patients p USING (patient_id)
GROUP BY s.risk_level
ORDER BY avg_edi DESC;"""

    query = st.text_area("SQL Query", value=default_q, height=160)
    if st.button("Run Query", type="primary"):
        if not db_ok():
            st.error("edi.db not found — run `python pipeline.py` first.")
        else:
            try:
                result = read_sql(query)
                st.success(f"{len(result)} rows")
                st.dataframe(result, use_container_width=True)
            except Exception as e:
                st.error(f"Query error: {e}")

    with st.expander("Pre-built queries (sql/analysis_queries.sql)"):
        if os.path.exists("sql/analysis_queries.sql"):
            with open("sql/analysis_queries.sql") as f:
                st.code(f.read(), language="sql")
