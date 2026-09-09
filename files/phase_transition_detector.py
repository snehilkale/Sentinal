"""
Phase transition detector: wraps NumpyGRUClassifier to produce, per
scenario, a stream of (phase_probs, predicted_phase, confidence,
transition_velocity) — the last being the rate of change of the phase
probability distribution over time (large jumps = fast escalation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sequences import build_sequences
from numpy_gru import NumpyGRUClassifier
from generators import CHANNEL_NAMES, PHASES


class PhaseTransitionDetector:
    def __init__(self, seq_len: int = 50, hidden_dim: int = 32, channels: list[str] | None = None, seed: int = 0):
        self.seq_len = seq_len
        self.channels = channels or CHANNEL_NAMES
        self.model = NumpyGRUClassifier(
            input_dim=len(self.channels), hidden_dim=hidden_dim, n_classes=4, seed=seed
        )

    def fit(
        self,
        scenarios: list[pd.DataFrame],
        stride: int = 4,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 2e-3,
        dropout_p: float = 0.3,
        verbose: bool = False,
    ) -> "PhaseTransitionDetector":
        X, y, risk, groups = build_sequences(scenarios, seq_len=self.seq_len, stride=stride, channels=self.channels)
        self.model.fit(
            X, y, groups=groups, epochs=epochs, batch_size=batch_size, lr=lr,
            dropout_p=dropout_p, class_weight=True, verbose=verbose,
        )
        return self

    def score_scenario(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every step of a scenario for which a full seq_len window of
        history is available (steps 0..seq_len-2 have no prediction).
        Returns a DataFrame with columns: step, phase probs (one column
        per phase), predicted_phase, confidence (max prob), and
        transition_velocity (L2 norm of the change in the prob vector
        vs. the previous scored step — a fast-moving distribution signals
        rapid escalation even before a hard phase-label flip happens).
        """
        X, y, risk, _ = build_sequences([df], seq_len=self.seq_len, stride=1, channels=self.channels)
        probs = self.model.predict_proba(X)
        steps = df["step"].values[self.seq_len - 1:]

        out = pd.DataFrame(probs, columns=[f"prob_{p}" for p in PHASES])
        out.insert(0, "step", steps)
        out["predicted_phase"] = [PHASES[i] for i in probs.argmax(axis=1)]
        out["confidence"] = probs.max(axis=1)
        out["true_phase_id"] = y
        out["risk_score"] = risk

        deltas = np.diff(probs, axis=0, prepend=probs[:1])
        out["transition_velocity"] = np.linalg.norm(deltas, axis=1)
        return out
