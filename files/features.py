"""
Feature extraction shared across detectors: rolling statistics per
channel over a sliding window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew

from generators import CHANNEL_NAMES


def rolling_features(df: pd.DataFrame, window: int = 15, channels: list[str] | None = None) -> pd.DataFrame:
    """
    Compute rolling mean/std/min/max/skew per channel over `window` steps.

    First `window-1` rows will contain NaN (not enough history) and should
    be dropped by the caller before fitting/scoring.
    """
    channels = channels or CHANNEL_NAMES
    feats = {}
    for ch in channels:
        s = df[ch]
        roll = s.rolling(window=window, min_periods=window)
        feats[f"{ch}_mean"] = roll.mean()
        feats[f"{ch}_std"] = roll.std()
        feats[f"{ch}_min"] = roll.min()
        feats[f"{ch}_max"] = roll.max()
        feats[f"{ch}_skew"] = roll.apply(lambda x: skew(x) if np.std(x) > 1e-9 else 0.0, raw=True)
    out = pd.DataFrame(feats, index=df.index)
    return out


def build_feature_matrix(
    scenarios: list[pd.DataFrame], window: int = 15, channels: list[str] | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute rolling features for a list of scenarios and stack them,
    keeping alignment with a combined metadata frame (phase, risk_score,
    is_incident, scenario_id, step) for the surviving (non-NaN) rows.
    """
    channels = channels or CHANNEL_NAMES
    feat_frames = []
    meta_frames = []
    for df in scenarios:
        f = rolling_features(df, window=window, channels=channels)
        valid = f.notna().all(axis=1)
        feat_frames.append(f[valid])
        meta_cols = ["step", "phase", "phase_id", "risk_score", "is_incident"]
        if "scenario_id" in df.columns:
            meta_cols.append("scenario_id")
        meta_frames.append(df.loc[valid, meta_cols])
    X = pd.concat(feat_frames, axis=0, ignore_index=True)
    meta = pd.concat(meta_frames, axis=0, ignore_index=True)
    return X.values, meta
