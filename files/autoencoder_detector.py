"""
Autoencoder-based anomaly detector, trained on NORMAL-phase rolling
features only (mirrors IsolationForestDetector's contract).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import build_feature_matrix
from numpy_nn import NumpyAutoencoder
from generators import CHANNEL_NAMES


class AutoencoderDetector:
    def __init__(self, window: int = 15, channels: list[str] | None = None, seed: int = 0):
        self.window = window
        self.channels = channels or CHANNEL_NAMES
        self.seed = seed
        self.ae: NumpyAutoencoder | None = None

    def fit(
        self,
        normal_scenarios: list[pd.DataFrame],
        epochs: int = 300,
        batch_size: int = 64,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> "AutoencoderDetector":
        X, meta = build_feature_matrix(normal_scenarios, window=self.window, channels=self.channels)
        mask = (meta["phase"] == "NORMAL").values
        X_normal = X[mask]
        if len(X_normal) < 50:
            raise ValueError(f"Not enough NORMAL rows to fit ({len(X_normal)}).")
        self.ae = NumpyAutoencoder(input_dim=X_normal.shape[1], seed=self.seed)
        self.ae.fit(X_normal, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        return self

    def score_scenario(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.ae is None:
            raise RuntimeError("Detector not fitted.")
        X, meta = build_feature_matrix([df], window=self.window, channels=self.channels)
        out = meta.copy()
        out["recon_error"] = self.ae.reconstruction_error(X)
        out["anomaly_score"] = self.ae.score(X)
        out["is_outlier"] = out["recon_error"] > self.ae.threshold_
        return out
