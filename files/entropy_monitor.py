"""
4-channel Causal Entropy Divergence framework ("KA framework").

C1 - Causal Entropic Response: rolling permutation entropy (per channel,
     averaged across channels) and its rate of increase.
C2 - Structural Coupling Covariance: rolling cross-covariance/correlation
     matrix across all channels, compared to a NORMAL-phase baseline
     correlation matrix via Frobenius norm.
C3 - Residual Z-score: simple AR(p) one-step-ahead predictor per channel
     (fit by least squares on NORMAL data), residual z-scored against the
     baseline residual distribution, averaged across channels.
CB - Correlation Break: rolling pairwise correlation matrix vs baseline;
     fraction of channel pairs whose correlation has moved >2 baseline-std
     away from its baseline value.

All four are squashed to roughly [0, 1] and fused into a single scalar
fCER (fused Causal Entropy Response), plus a persistent-alert flag that
requires fCER to stay elevated for several consecutive steps (a single-
step spike is noise; a sustained rise is a genuine regime change).
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pandas as pd

from generators import CHANNEL_NAMES


# ---------------------------------------------------------------- C1 ----
def _permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Standard ordinal-pattern permutation entropy of a 1D array, normalized to [0,1]."""
    n = len(x)
    m = n - (order - 1) * delay
    if m <= 1:
        return 0.0
    patterns = {}
    for i in range(m):
        window = x[i:i + order * delay:delay]
        rank = tuple(np.argsort(window))
        patterns[rank] = patterns.get(rank, 0) + 1
    counts = np.array(list(patterns.values()), dtype=float)
    probs = counts / counts.sum()
    ent = -np.sum(probs * np.log(probs))
    max_ent = np.log(math.factorial(order))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def _rolling_permutation_entropy(values: np.ndarray, window: int, order: int = 3) -> np.ndarray:
    """Per-step rolling permutation entropy (averaged over channels already, if 1D input)."""
    n = len(values)
    out = np.full(n, np.nan)
    for end in range(window - 1, n):
        out[end] = _permutation_entropy(values[end - window + 1:end + 1], order=order)
    return out


# ---------------------------------------------------------------- C3 ----
def _fit_ar(values: np.ndarray, order: int) -> np.ndarray:
    """Fit AR(p) coefficients (+ intercept) by least squares: x_t ~ c + sum_i a_i x_{t-i}."""
    n = len(values)
    if n <= order + 5:
        return np.zeros(order + 1)
    Y = values[order:]
    X = np.column_stack(
        [np.ones(n - order)] + [values[order - i:n - i] for i in range(1, order + 1)]
    )
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return coef  # [intercept, a_1, ..., a_p]


def _ar_predict(values: np.ndarray, coef: np.ndarray, order: int) -> np.ndarray:
    """One-step-ahead AR predictions for indices >= order (NaN before that)."""
    n = len(values)
    pred = np.full(n, np.nan)
    for t in range(order, n):
        lags = values[t - order:t][::-1]  # x_{t-1}, x_{t-2}, ..., x_{t-p}
        pred[t] = coef[0] + np.dot(coef[1:], lags)
    return pred


class EntropyMonitor:
    def __init__(
        self,
        channels: list[str] | None = None,
        c1_window: int = 20,
        c2_window: int = 30,
        cb_window: int = 50,
        ar_order: int = 5,
        pe_order: int = 3,
        persistence_steps: int = 5,
        alert_threshold: float = 0.28,
    ):
        self.channels = channels or CHANNEL_NAMES
        self.c1_window = c1_window
        self.c2_window = c2_window
        self.cb_window = cb_window
        self.ar_order = ar_order
        self.pe_order = pe_order
        self.persistence_steps = persistence_steps
        self.alert_threshold = alert_threshold

        self.baseline_corr_: np.ndarray | None = None
        self.ar_coefs_: dict[str, np.ndarray] = {}
        self.ar_resid_mean_: dict[str, float] = {}
        self.ar_resid_std_: dict[str, float] = {}
        self.baseline_pe_: float | None = None
        self.baseline_pairwise_corr_std_: np.ndarray | None = None

    # ------------------------------------------------------------ fit --
    def fit(self, normal_scenarios: list[pd.DataFrame]) -> "EntropyMonitor":
        normal_concat = {
            ch: np.concatenate([df.loc[df.phase == "NORMAL", ch].values for df in normal_scenarios])
            for ch in self.channels
        }

        # C2/CB baseline: correlation matrix across channels under NORMAL.
        mat = np.column_stack([normal_concat[ch] for ch in self.channels])
        self.baseline_corr_ = np.corrcoef(mat, rowvar=False)

        # Baseline variability of rolling pairwise correlations (for CB's
        # ">2 std from baseline" test) — estimated via block bootstrap over
        # NORMAL windows within each scenario.
        pairwise_samples = []
        for df in normal_scenarios:
            sub = df.loc[df.phase == "NORMAL", self.channels]
            if len(sub) < self.cb_window * 2:
                continue
            for start in range(0, len(sub) - self.cb_window, max(self.cb_window // 2, 1)):
                block = sub.iloc[start:start + self.cb_window].values
                if block.std(axis=0).min() < 1e-9:
                    continue
                c = np.corrcoef(block, rowvar=False)
                pairwise_samples.append(c)
        if pairwise_samples:
            stacked = np.stack(pairwise_samples)
            self.baseline_pairwise_corr_std_ = stacked.std(axis=0)
        else:
            self.baseline_pairwise_corr_std_ = np.full_like(self.baseline_corr_, 0.1)

        # C3: per-channel AR(p) fit + baseline residual distribution.
        for ch in self.channels:
            vals = normal_concat[ch]
            coef = _fit_ar(vals, self.ar_order)
            self.ar_coefs_[ch] = coef
            pred = _ar_predict(vals, coef, self.ar_order)
            resid = vals - pred
            resid = resid[~np.isnan(resid)]
            self.ar_resid_mean_[ch] = float(np.mean(resid)) if len(resid) else 0.0
            self.ar_resid_std_[ch] = float(np.std(resid) + 1e-8) if len(resid) else 1.0

        # C1: baseline permutation entropy level (per-channel, averaged).
        pe_vals = []
        for ch in self.channels:
            vals = normal_concat[ch]
            if len(vals) >= self.c1_window:
                pe_vals.append(_permutation_entropy(vals[-self.c1_window * 4:], order=self.pe_order))
        self.baseline_pe_ = float(np.mean(pe_vals)) if pe_vals else 0.5

        return self

    # --------------------------------------------------------- score ---
    def score_scenario(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        vals = {ch: df[ch].values for ch in self.channels}

        # ---- C1: mean rolling permutation entropy across channels, + its rate of change
        pe_per_channel = np.stack(
            [_rolling_permutation_entropy(vals[ch], self.c1_window, self.pe_order) for ch in self.channels]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            c1_entropy = np.nanmean(pe_per_channel, axis=0)  # (n,) — first c1_window-1 entries are NaN by design
        c1_rate = np.gradient(np.nan_to_num(c1_entropy, nan=self.baseline_pe_))
        # Squash relative to the *gap to max possible entropy* (baseline is
        # often already near-maximal for noisy real-valued windows, so an
        # absolute-difference squash saturates uselessly) — use a gentler
        # linear-ish scaling calibrated on the observed deviation range.
        c1_gap = np.clip(self.baseline_pe_ - c1_entropy, 0, None)  # entropy *drop* is unusual too
        c1_raw = np.abs(c1_entropy - self.baseline_pe_) * 3.0 + np.clip(c1_rate, 0, None) * 15.0
        c1_signal = np.clip(c1_raw, 0, 1)

        # ---- C2: rolling correlation matrix vs baseline, Frobenius norm
        c2_signal = np.full(n, np.nan)
        mat_all = np.column_stack([vals[ch] for ch in self.channels])
        for end in range(self.c2_window - 1, n):
            block = mat_all[end - self.c2_window + 1:end + 1]
            if block.std(axis=0).min() < 1e-9:
                c2_signal[end] = 0.0
                continue
            corr = np.corrcoef(block, rowvar=False)
            fro = np.linalg.norm(corr - self.baseline_corr_, ord="fro")
            c2_signal[end] = fro
        max_possible_fro = np.linalg.norm(np.ones_like(self.baseline_corr_) * 2, ord="fro")
        c2_norm = np.clip(c2_signal / (max_possible_fro * 0.25), 0, 1)  # 0.25x-of-max already very anomalous

        # ---- C3: AR residual z-score, averaged |z| across channels
        z_per_channel = []
        for ch in self.channels:
            pred = _ar_predict(vals[ch], self.ar_coefs_[ch], self.ar_order)
            resid = vals[ch] - pred
            z = (resid - self.ar_resid_mean_[ch]) / self.ar_resid_std_[ch]
            z_per_channel.append(np.abs(z))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            c3_abs_z = np.nanmean(np.stack(z_per_channel), axis=0)
        c3_norm = 1.0 / (1.0 + np.exp(-1.2 * (c3_abs_z - 2.5)))  # centered around "2.5 sigma"

        # ---- CB: rolling pairwise correlation break fraction
        cb_signal = np.full(n, np.nan)
        n_pairs = len(self.channels) * (len(self.channels) - 1) // 2
        for end in range(self.cb_window - 1, n):
            block = mat_all[end - self.cb_window + 1:end + 1]
            if block.std(axis=0).min() < 1e-9:
                cb_signal[end] = 0.0
                continue
            corr = np.corrcoef(block, rowvar=False)
            dev = np.abs(corr - self.baseline_corr_)
            broken = dev > 2.0 * np.maximum(self.baseline_pairwise_corr_std_, 0.05)
            iu = np.triu_indices_from(broken, k=1)
            cb_signal[end] = broken[iu].sum() / max(n_pairs, 1)
        cb_norm = np.clip(cb_signal, 0, 1)

        out = pd.DataFrame({
            "step": df["step"].values,
            "C1_entropy": c1_signal,
            "C2_structural": np.nan_to_num(c2_norm, nan=0.0),
            "C3_residual_z": c3_norm,
            "CB_corr_break": np.nan_to_num(cb_norm, nan=0.0),
        })

        # fuse (equal weights across the 4 channels, per spec's "4-channel" framing)
        out["fCER"] = out[["C1_entropy", "C2_structural", "C3_residual_z", "CB_corr_break"]].mean(axis=1)

        # persistent alert: fCER above threshold for `persistence_steps` in a row
        above = (out["fCER"] >= self.alert_threshold).astype(int)
        run_length = above.groupby((above != above.shift()).cumsum()).cumsum()
        out["persistent_alert"] = (above == 1) & (run_length >= self.persistence_steps)

        out["phase"] = df["phase"].values
        out["risk_score"] = df["risk_score"].values
        return out
