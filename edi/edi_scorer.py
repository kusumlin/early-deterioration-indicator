"""
EDI Scorer: Naive Bayes log-ratios → logistic regression → 0-1 probability → risk tier.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from .risk_curves import RiskCurveModel, FEATURES


RISK_LEVELS = {
    (0.00, 0.25): ("LOW",      "green"),
    (0.25, 0.50): ("MODERATE", "yellow"),
    (0.50, 0.75): ("HIGH",     "orange"),
    (0.75, 1.00): ("CRITICAL", "red"),
}


def _risk_label(prob: float) -> tuple[str, str]:
    for (lo, hi), (label, color) in RISK_LEVELS.items():
        if lo <= prob < hi:
            return label, color
    return "CRITICAL", "red"


class EDIScorer:
    """
    End-to-end pipeline:
      1. RiskCurveModel (Naive Bayes per feature) → composite log-ratio score
      2. LogisticRegression → instability probability in [0, 1]
      3. EDI label + risk tier
    """

    def __init__(self):
        self.risk_model = RiskCurveModel()
        self.lr = LogisticRegression(max_iter=1000)
        self.score_scaler = MinMaxScaler()
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EDIScorer":
        self.risk_model.fit(X, y)
        composite = self.risk_model.composite_score(X).reshape(-1, 1)
        self.score_scaler.fit(composite)
        self.lr.fit(composite, y)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        composite = self.risk_model.composite_score(X).reshape(-1, 1)
        return self.lr.predict_proba(composite)[:, 1]

    def score_patients(self, X: pd.DataFrame) -> pd.DataFrame:
        probs = self.predict_proba(X)
        log_ratios = self.risk_model.patient_log_ratios(X)

        results = X.copy()
        results["edi_probability"] = probs.round(4)
        results["composite_score"] = self.risk_model.composite_score(X).round(4)

        labels, colors = zip(*[_risk_label(p) for p in probs])
        results["risk_level"] = labels
        results["risk_color"] = colors

        # Per-feature contributions (normalized log-ratios)
        for feat in FEATURES:
            results[f"lr_{feat}"] = log_ratios[feat].round(4)

        return results

    def score_single(self, vitals: dict) -> dict:
        """Score a single patient given a dict of feature values."""
        X = pd.DataFrame([vitals])[FEATURES]
        prob = float(self.predict_proba(X)[0])
        label, color = _risk_label(prob)
        lr = self.risk_model.patient_log_ratios(X).iloc[0].to_dict()
        return {
            "edi_probability": round(prob, 4),
            "risk_level": label,
            "risk_color": color,
            "feature_contributions": {k: round(v, 4) for k, v in lr.items()},
        }
