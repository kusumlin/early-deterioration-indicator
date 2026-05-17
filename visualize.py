"""
EDI Visualization Module

Produces four plot types:
  1. Per-feature risk curves (P(deterioration) vs feature value)
  2. EDI probability distribution (stable vs deteriorating cohorts)
  3. ROC curve with AUROC
  4. Patient dashboard — waterfall of feature contributions for a single patient
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc

from edi.risk_curves import RiskCurveModel, FEATURES, FEATURE_WEIGHTS

# Feature display metadata: (label, physiological range, unit)
FEATURE_META = {
    "hr":   ("Heart Rate",        (30, 180),   "bpm"),
    "rr":   ("Resp. Rate",        (4,  40),    "br/min"),
    "sbp":  ("Systolic BP",       (60, 200),   "mmHg"),
    "temp": ("Temperature",       (34, 42),    "°C"),
    "spo2": ("SpO₂",             (70, 100),   "%"),
    "loc":  ("Level of Consc.",   (0,  1),     "0=alert"),
    "age":  ("Age",               (18, 95),    "years"),
}

ZONE_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
ZONE_LABELS = ["Low (0–0.25)", "Moderate (0.25–0.5)", "High (0.5–0.75)", "Critical (0.75–1.0)"]


def plot_risk_curves(risk_model: RiskCurveModel, save_path: str = "output/risk_curves.png"):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle("EDI — Per-Feature Risk Curves (Naive Bayes)", fontsize=15, fontweight="bold")

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        label, (lo, hi), unit = FEATURE_META[feat]

        x = np.linspace(lo, hi, 300)
        risk = risk_model.feature_risk(feat, x)

        # Shade risk zones
        thresholds = [lo, lo + (hi - lo) * 0.25, lo + (hi - lo) * 0.5, lo + (hi - lo) * 0.75, hi]
        for j, color in enumerate(ZONE_COLORS):
            ax.axvspan(thresholds[j], thresholds[j + 1], alpha=0.08, color=color)

        ax.plot(x, risk, color="#2c3e50", linewidth=2.2, label="P(deterioration)")
        ax.axhline(0.5, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.6, label="Decision boundary")

        ax.set_title(f"{label}", fontsize=10, fontweight="bold")
        ax.set_xlabel(f"{unit}", fontsize=8)
        ax.set_ylabel("P(deterioration)", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_edi_distribution(results: pd.DataFrame, save_path: str = "output/edi_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("EDI Probability Distribution", fontsize=14, fontweight="bold")

    # Histogram by true label
    ax = axes[0]
    stable = results[results["deteriorated"] == 0]["edi_probability"]
    deteri = results[results["deteriorated"] == 1]["edi_probability"]
    ax.hist(stable, bins=30, alpha=0.65, color="#2ecc71", label=f"Stable (n={len(stable)})", density=True)
    ax.hist(deteri, bins=30, alpha=0.65, color="#e74c3c", label=f"Deteriorating (n={len(deteri)})", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold 0.5")
    ax.set_xlabel("EDI Probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Score Separation by Class", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Risk level pie chart
    ax2 = axes[1]
    counts = results["risk_level"].value_counts()
    pie_colors = {"LOW": "#2ecc71", "MODERATE": "#f1c40f", "HIGH": "#e67e22", "CRITICAL": "#e74c3c"}
    colors = [pie_colors.get(k, "gray") for k in counts.index]
    ax2.pie(counts.values, labels=counts.index, colors=colors, autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 10})
    ax2.set_title("Risk Level Distribution", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curve(results: pd.DataFrame, save_path: str = "output/roc_curve.png"):
    fpr, tpr, _ = roc_curve(results["deteriorated"], results["edi_probability"])
    auroc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2c3e50", linewidth=2.5, label=f"EDI (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2c3e50")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — EDI Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}  |  AUROC = {auroc:.3f}")
    return auroc


def plot_patient_dashboard(patient_result: dict, patient_id: int = 1,
                           save_path: str = "output/patient_dashboard.png"):
    """Waterfall chart showing each feature's contribution to the EDI score."""
    contributions = patient_result["feature_contributions"]
    feats = list(contributions.keys())
    vals  = list(contributions.values())

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[2, 1, 1])

    # Waterfall
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.barh(feats, vals, color=colors, edgecolor="white", height=0.6)
    ax1.axvline(0, color="black", linewidth=1)
    for bar, val in zip(bars, vals):
        ax1.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height() / 2,
                 f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
    ax1.set_xlabel("Log-Ratio Contribution (+ = destabilizing)", fontsize=10)
    ax1.set_title("Feature Contributions to EDI Score", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="x", alpha=0.3)

    # EDI gauge
    ax2 = fig.add_subplot(gs[1])
    prob = patient_result["edi_probability"]
    level = patient_result["risk_level"]
    color_map = {"LOW": "#2ecc71", "MODERATE": "#f1c40f", "HIGH": "#e67e22", "CRITICAL": "#e74c3c"}
    gauge_color = color_map[level]

    ax2.barh(["EDI Score"], [prob], color=gauge_color, height=0.4)
    ax2.barh(["EDI Score"], [1 - prob], left=[prob], color="#ecf0f1", height=0.4)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Instability Probability", fontsize=10)
    ax2.set_title(f"Patient #{patient_id}\nEDI = {prob:.3f}", fontsize=11, fontweight="bold")
    ax2.text(0.5, 0, f"Risk: {level}", ha="center", va="bottom", fontsize=13,
             fontweight="bold", color=gauge_color, transform=ax2.transAxes)
    ax2.grid(True, axis="x", alpha=0.3)

    # Legend
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(ZONE_COLORS, ZONE_LABELS)]
    ax3.legend(handles=patches, loc="center", fontsize=10, title="Risk Zones", title_fontsize=11,
               framealpha=0.9)

    fig.suptitle("EDI Patient Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
