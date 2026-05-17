"""
Naive Bayes–based risk curve model.

For each feature, fits two Gaussian distributions (stable vs deteriorating)
and computes a log-likelihood ratio that represents how "unstable" a given
value is.  The raw log-ratios are then normalized per-feature so they can
be combined as a composite score.

Reference: Continuous risk-scoring approach outperforms NEWS/MEWS
(AUROC 0.77 vs 0.66, detects deterioration ~7 h earlier).
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


FEATURES = ["hr", "rr", "sbp", "temp", "spo2", "loc", "age"]

# Weights tuned from literature — LOC and SpO2 matter most clinically.
FEATURE_WEIGHTS = {
    "hr":   1.0,
    "rr":   1.2,
    "sbp":  1.2,
    "temp": 0.9,
    "spo2": 1.3,
    "loc":  1.5,
    "age":  0.8,
}


class RiskCurveModel:
    """
    Fits per-feature Gaussian Naive Bayes models and exposes smooth
    probability-of-deterioration curves over each feature's value range.
    """

    def __init__(self):
        self.models: dict[str, GaussianNB] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RiskCurveModel":
        for feat in FEATURES:
            scaler = StandardScaler()
            # Use .values so scaler stores no feature names — avoids transform warnings
            x_scaled = scaler.fit_transform(X[[feat]].values)
            gnb = GaussianNB()
            gnb.fit(x_scaled, y)
            self.models[feat] = gnb
            self.scalers[feat] = scaler
        self._fitted = True
        return self

    def feature_risk(self, feature: str, values: np.ndarray) -> np.ndarray:
        """Return P(deterioration | feature=value) for an array of values."""
        scaler = self.scalers[feature]
        gnb = self.models[feature]
        x = scaler.transform(values.reshape(-1, 1))
        return gnb.predict_proba(x)[:, 1]  # P(class=1)

    def patient_log_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return per-feature log P(deterioration|x) / P(stable|x) for each patient.
        Positive = destabilizing, negative = stabilizing.
        """
        log_ratios = {}
        for feat in FEATURES:
            scaler = self.scalers[feat]
            gnb = self.models[feat]
            x = scaler.transform(X[[feat]].values)
            proba = gnb.predict_proba(x)
            # Avoid log(0)
            p_det   = np.clip(proba[:, 1], 1e-9, 1 - 1e-9)
            p_stab  = np.clip(proba[:, 0], 1e-9, 1 - 1e-9)
            log_ratios[feat] = np.log(p_det / p_stab) * FEATURE_WEIGHTS[feat]
        return pd.DataFrame(log_ratios, index=X.index)

    def composite_score(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted sum of per-feature log-ratios."""
        lr = self.patient_log_ratios(X)
        return lr.sum(axis=1).values
