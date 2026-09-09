"""
Component 5: Multimodal Risk Fusion.

Combines the three learned detectors (IsolationForest anomaly score,
LSTM/GRU phase-transition probabilities, fused entropy signal) with a
simple, transparent rules engine, into one scalar risk score with
component-level attribution (so an operator can see *why* SENTINEL is
alarmed, not just that it is).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from generators import PHASES, CHANNEL_NAMES

PHASE_RISK_MAP = {"NORMAL": 0.1, "SUSPICIOUS": 0.4, "DANGER": 0.7, "EMERGENCY": 0.95}

DEFAULT_WEIGHTS = {
    "anomaly": 0.20,
    "phase_transition": 0.30,
    "entropy": 0.30,
    "rules": 0.20,
}


# ------------------------------------------------------------- rules ----
def rules_score(
    df: pd.DataFrame,
    hard_limits: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Simple, explainable threshold rules as a sanity backstop for the
    learned models: fraction of channels currently outside a generous
    hard operating envelope (mean +/- 4 baseline-std, computed from the
    channel's own NORMAL-like early-window statistics as a proxy baseline
    when explicit limits aren't supplied). This deliberately does NOT try
    to be clever — it's the "a human wrote an obvious safety rule" layer,
    so its disagreement with the learned components is itself informative.
    """
    channels = [c for c in CHANNEL_NAMES if c in df.columns]
    if hard_limits is None:
        # derive generous limits from the first 30 steps of THIS scenario
        # as a stand-in for a facility's documented operating envelope
        ref = df[channels].iloc[: min(30, len(df))]
        mean, std = ref.mean(), ref.std().replace(0, 1e-6)
        hard_limits = {c: (mean[c] - 4 * std[c], mean[c] + 4 * std[c]) for c in channels}

    breaches = pd.DataFrame(index=df.index)
    for c in channels:
        lo, hi = hard_limits[c]
        breaches[c] = (df[c] < lo) | (df[c] > hi)
    return breaches.mean(axis=1).values  # fraction of channels breaching


# ------------------------------------------------------------- fusion ---
def compute_risk(
    anomaly_score: float,
    phase_probs: dict[str, float],
    entropy_signal: float,
    rules_score_val: float,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Weighted fusion of the four risk components, per spec.

    Parameters
    ----------
    anomaly_score : float in [0,1], from IsolationForestDetector /
        AutoencoderDetector (average or max of the two — caller's choice).
    phase_probs : dict phase_name -> probability, from PhaseTransitionDetector.
    entropy_signal : float in [0,1], fCER from EntropyMonitor.
    rules_score_val : float in [0,1], fraction of channels breaching hard limits.
    weights : optional override of DEFAULT_WEIGHTS.

    Returns
    -------
    risk : float in [0,1], the fused SENTINEL risk score.
    contributions : dict component_name -> its weighted contribution to `risk`,
        so the sum of values equals `risk` (full attribution/explainability).
    """
    weights = weights or DEFAULT_WEIGHTS
    phase_risk = sum(phase_probs.get(p, 0.0) * PHASE_RISK_MAP[p] for p in PHASES)

    contributions = {
        "anomaly": weights["anomaly"] * float(np.clip(anomaly_score, 0, 1)),
        "phase_transition": weights["phase_transition"] * float(np.clip(phase_risk, 0, 1)),
        "entropy": weights["entropy"] * float(np.clip(entropy_signal, 0, 1)),
        "rules": weights["rules"] * float(np.clip(rules_score_val, 0, 1)),
    }
    risk = float(sum(contributions.values()))
    return risk, contributions


class RiskFuser:
    """
    Ties together IsolationForestDetector, AutoencoderDetector,
    PhaseTransitionDetector and EntropyMonitor scoring outputs (already
    computed via each detector's `score_scenario`) into one aligned,
    per-step fused risk timeline.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def fuse(
        self,
        raw_df: pd.DataFrame,
        if_scored: pd.DataFrame,
        ae_scored: pd.DataFrame,
        phase_scored: pd.DataFrame,
        entropy_scored: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Each `*_scored` frame is indexed by `step` but may start at a
        different offset (rolling-window / sequence-length warm-up differs
        per detector). We inner-join on `step` so every output row has a
        value from every component — the warm-up period (roughly the first
        max(window sizes) steps) simply has no fused score, same as any
        real system needs to "warm up" its baselines first.
        """
        rules = rules_score(raw_df)
        rules_df = pd.DataFrame({"step": raw_df["step"].values, "rules_score": rules})

        merged = (
            if_scored[["step", "anomaly_score"]].rename(columns={"anomaly_score": "if_anomaly"})
            .merge(ae_scored[["step", "anomaly_score"]].rename(columns={"anomaly_score": "ae_anomaly"}), on="step")
            .merge(phase_scored[["step"] + [f"prob_{p}" for p in PHASES] + ["predicted_phase", "confidence", "transition_velocity"]], on="step")
            .merge(entropy_scored[["step", "fCER", "persistent_alert",
                                    "C1_entropy", "C2_structural", "C3_residual_z", "CB_corr_break"]], on="step")
            .merge(rules_df, on="step")
        )

        merged["anomaly_combined"] = merged[["if_anomaly", "ae_anomaly"]].mean(axis=1)

        risks, contribs = [], {"anomaly": [], "phase_transition": [], "entropy": [], "rules": []}
        for _, row in merged.iterrows():
            phase_probs = {p: row[f"prob_{p}"] for p in PHASES}
            risk, contribution = compute_risk(
                anomaly_score=row["anomaly_combined"],
                phase_probs=phase_probs,
                entropy_signal=row["fCER"],
                rules_score_val=row["rules_score"],
                weights=self.weights,
            )
            risks.append(risk)
            for k in contribs:
                contribs[k].append(contribution[k])

        merged["sentinel_risk"] = risks
        for k in contribs:
            merged[f"contrib_{k}"] = contribs[k]

        # ground truth, for evaluation convenience
        gt = raw_df.set_index("step")[["phase", "risk_score", "is_incident"]]
        merged = merged.merge(gt, left_on="step", right_index=True, how="left", suffixes=("", "_true"))
        return merged
