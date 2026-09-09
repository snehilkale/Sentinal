"""
IsolationForest-based anomaly detector.

Trained ONLY on NORMAL-phase rolling-feature vectors, so it learns the
shape of "business as usual" and flags departures from it — regardless
of whether those departures turn out to be SUSPICIOUS/DANGER/EMERGENCY.
That's the isolation forest's job: pure statistical outlier detection,
with no notion of phase ordering (that's what the LSTM phase classifier
is for, downstream).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import build_feature_matrix
from generators import CHANNEL_NAMES


class IsolationForestDetector:
    def __init__(self, window: int = 15, channels: list[str] | None = None, **if_kwargs):
        self.window = window
        self.channels = channels or CHANNEL_NAMES
        self.scaler = StandardScaler()
        default_kwargs = dict(n_estimators=200, contamination="auto", random_state=0)
        default_kwargs.update(if_kwargs)
        self.model = IsolationForest(**default_kwargs)
        self._fitted = False

    def fit(self, normal_scenarios: list[pd.DataFrame]) -> "IsolationForestDetector":
        """Fit on rolling features computed from NORMAL-phase rows only."""
        X, meta = build_feature_matrix(normal_scenarios, window=self.window, channels=self.channels)
        mask = (meta["phase"] == "NORMAL").values
        X_normal = X[mask]
        if len(X_normal) < 20:
            raise ValueError(f"Not enough NORMAL rows to fit ({len(X_normal)}); need more/longer scenarios.")
        Xs = self.scaler.fit_transform(X_normal)
        self.model.fit(Xs)
        self._fitted = True
        return self

    def score_features(self, X: np.ndarray) -> np.ndarray:
        """
        Score a raw feature matrix (already rolling-features, aligned
        columns). Returns anomaly_score in [-1, 1]-ish sklearn convention
        via decision_function: higher = more normal, lower/negative = more
        anomalous. We flip sign so higher = MORE anomalous, consistent
        with the rest of SENTINEL's "higher score = more danger" convention.
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted.")
        Xs = self.scaler.transform(X)
        raw = self.model.decision_function(Xs)  # high = normal, low = anomalous
        return -raw  # flip: high = anomalous

    def score_scenario(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a single scenario DataFrame. Returns a DataFrame indexed
        like the valid (non-NaN-window) rows, with columns: step,
        anomaly_score_raw (flipped decision_function), anomaly_score
        (min-max normalized to [0,1] using a fixed reference range so
        scores are comparable across scenarios), is_outlier (sklearn
        predict == -1).
        """
        X, meta = build_feature_matrix([df], window=self.window, channels=self.channels)
        Xs = self.scaler.transform(X)
        raw = -self.model.decision_function(Xs)
        pred = self.model.predict(Xs)  # -1 outlier, 1 inlier
        out = meta.copy()
        out["anomaly_score_raw"] = raw
        # Normalize using a stable logistic squashing around 0 (decision
        # function is roughly centered there) rather than per-scenario
        # min-max, so scores are comparable across different scenarios.
        out["anomaly_score"] = 1.0 / (1.0 + np.exp(-8.0 * raw))
        out["is_outlier"] = pred == -1
        return out
