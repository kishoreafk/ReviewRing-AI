"""Post-hoc probability calibration (Platt scaling) on a separate partition."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Logistic regression on clipped logits, fit on the calibration split.

    Never fit on the test set (spec 11.6). ``score_kind`` records which
    population the calibration describes so the UI cannot present a synthetic
    calibrated score as real-world fraud probability.
    """

    def __init__(self, score_kind: str = "raw"):
        self.clf: LogisticRegression | None = None
        self.score_kind = score_kind
        self.fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        labels = np.asarray(labels)
        logits = np.asarray(logits, dtype=np.float64)
        logits = np.clip(logits, -30.0, 30.0).reshape(-1, 1)
        if len(np.unique(labels)) < 2:
            # cannot fit a two-class calibrator; keep raw scores
            self.fitted = False
            return self
        self.clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.clf.fit(logits, labels)
        self.fitted = True
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.clip(np.asarray(logits, dtype=np.float64), -30.0, 30.0).reshape(-1, 1)
        if not self.fitted or self.clf is None:
            from scipy.special import expit

            return expit(logits.reshape(-1))
        return self.clf.predict_proba(logits)[:, 1]

    def to_dict(self) -> dict:
        out = {"fitted": self.fitted, "score_kind": self.score_kind}
        if self.fitted and self.clf is not None:
            out["coef"] = float(self.clf.coef_[0, 0])
            out["intercept"] = float(self.clf.intercept_[0])
        return out

    @classmethod
    def from_dict(cls, d: dict) -> PlattCalibrator:
        cal = cls(score_kind=d.get("score_kind", "raw"))
        if d.get("fitted"):
            cal.clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            import numpy as np

            cal.clf.coef_ = np.array([[d["coef"]]])
            cal.clf.intercept_ = np.array([d["intercept"]])
            cal.clf.classes_ = np.array([0, 1])
            cal.fitted = True
        return cal
