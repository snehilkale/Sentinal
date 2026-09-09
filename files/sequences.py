"""
Build fixed-length (default 50) sliding-window sequences of raw channel
values for the recurrent phase classifier, labeled with the phase at the
LAST step of each window (i.e. "given the last 50 steps, what phase are
we in right now").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from generators import CHANNEL_NAMES, PHASES


def build_sequences(
    scenarios: list[pd.DataFrame],
    seq_len: int = 50,
    stride: int = 3,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : (n_samples, seq_len, n_channels) raw channel values
    y : (n_samples,) int phase_id labels (of the LAST step in the window)
    risk : (n_samples,) float risk_score (of the LAST step in the window)
    scenario_idx : (n_samples,) which scenario (index into `scenarios`) each
        window came from — used to split train/val BY SCENARIO, not by
        window, since overlapping windows (stride << seq_len) from the same
        scenario are near-duplicates and leak badly across a random split.
    """
    channels = channels or CHANNEL_NAMES
    X_list, y_list, risk_list, scen_list = [], [], [], []
    for i, df in enumerate(scenarios):
        vals = df[channels].values
        phase_id = df["phase_id"].values
        risk = df["risk_score"].values
        n = len(df)
        for end in range(seq_len - 1, n, stride):
            start = end - seq_len + 1
            X_list.append(vals[start:end + 1])
            y_list.append(phase_id[end])
            risk_list.append(risk[end])
            scen_list.append(i)
    X = np.stack(X_list).astype(np.float64)
    y = np.array(y_list, dtype=np.int64)
    risk = np.array(risk_list, dtype=np.float64)
    scenario_idx = np.array(scen_list, dtype=np.int64)
    return X, y, risk, scenario_idx


def compute_norm_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over all samples & timesteps, for normalization."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    return mean, std
