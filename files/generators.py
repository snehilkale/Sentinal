"""
SENTINEL — Synthetic scenario generation.

Generates multimodal time series (behavioral, digital, operational,
environmental) that progress through four ground-truth phases:

    NORMAL -> SUSPICIOUS -> DANGER -> EMERGENCY

Each scenario carries per-step ground truth: phase (str), phase_id (int),
risk_score (float 0-1), is_incident (bool, True for DANGER/EMERGENCY).

Design notes
------------
- Phase transitions are smoothed with a logistic blend so the detectors
  are not trained/evaluated against unrealistic step discontinuities.
- One "lead" signal category drives each scenario's escalation; the other
  three categories receive a smaller *coupling* deviation, which is what
  gives the entropy/covariance channels (C2, CB) something real to find —
  a pure single-channel spike is trivial, cross-channel coupling is not.
- Variance (not just mean) increases with phase severity, which is what
  makes entropy-based detection (permutation entropy, residual z-score)
  meaningfully different from a plain mean-threshold rule.
- A configurable fraction of scenarios are "control" runs that stay in
  NORMAL (or wobble into SUSPICIOUS and recover) the whole time, so
  detectors trained downstream see real negatives, not just escalations.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
import pandas as pd

PHASES = ["NORMAL", "SUSPICIOUS", "DANGER", "EMERGENCY"]
PHASE_TO_ID = {p: i for i, p in enumerate(PHASES)}
PHASE_RISK_RANGE = {
    "NORMAL": (0.0, 0.15),
    "SUSPICIOUS": (0.15, 0.40),
    "DANGER": (0.40, 0.75),
    "EMERGENCY": (0.75, 1.00),
}
# Fractional deviation from baseline (mid-phase target), per spec.
PHASE_DEVIATION_RANGE = {
    "NORMAL": (0.0, 0.03),
    "SUSPICIOUS": (0.05, 0.15),
    "DANGER": (0.30, 0.50),
    "EMERGENCY": (0.50, 0.90),
}
# Variance inflation multiplier by phase (entropy-relevant).
PHASE_VARIANCE_MULT = {
    "NORMAL": 1.0,
    "SUSPICIOUS": 1.3,
    "DANGER": 2.0,
    "EMERGENCY": 3.2,
}

CATEGORIES = ["behavioral", "digital", "operational", "environmental"]


@dataclasses.dataclass
class ChannelSpec:
    """One measured signal channel."""

    name: str
    category: str
    baseline: float
    noise_std: float          # 1-sigma noise at NORMAL phase
    direction: int            # +1 if incidents push value up, -1 if down
    min_value: float | None = None   # physical clamp (e.g. can't go < 0)
    max_value: float | None = None
    # For some channels (auth failures, gas spikes) deviations are bursty
    # rather than a smooth mean-shift — modeled as a Poisson-ish spike rate.
    bursty: bool = False


CHANNELS: list[ChannelSpec] = [
    # --- behavioral ---
    ChannelSpec("movement_speed", "behavioral", baseline=1.40, noise_std=0.15,
                direction=-1, min_value=0.0),
    ChannelSpec("keystroke_interval_ms", "behavioral", baseline=220.0, noise_std=30.0,
                direction=+1, min_value=0.0),
    ChannelSpec("gaze_fixation_ms", "behavioral", baseline=350.0, noise_std=60.0,
                direction=-1, min_value=0.0),
    # --- digital ---
    ChannelSpec("auth_failures_per_min", "digital", baseline=0.2, noise_std=0.25,
                direction=+1, min_value=0.0, bursty=True),
    ChannelSpec("network_bytes_kbps", "digital", baseline=500.0, noise_std=80.0,
                direction=+1, min_value=0.0),
    ChannelSpec("system_event_rate", "digital", baseline=12.0, noise_std=3.0,
                direction=+1, min_value=0.0),
    # --- operational ---
    ChannelSpec("vibration_mms", "operational", baseline=2.0, noise_std=0.3,
                direction=+1, min_value=0.0),
    ChannelSpec("temperature_c", "operational", baseline=45.0, noise_std=2.0,
                direction=+1),
    ChannelSpec("pressure_kpa", "operational", baseline=101.0, noise_std=1.5,
                direction=-1, min_value=0.0),
    # --- environmental ---
    ChannelSpec("gas_conc_ppm", "environmental", baseline=5.0, noise_std=1.0,
                direction=+1, min_value=0.0, bursty=True),
    ChannelSpec("humidity_pct", "environmental", baseline=45.0, noise_std=5.0,
                direction=+1, min_value=0.0, max_value=100.0),
    ChannelSpec("noise_level_db", "environmental", baseline=55.0, noise_std=4.0,
                direction=+1, min_value=0.0),
]
CHANNEL_NAMES = [c.name for c in CHANNELS]


def _logistic(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """Smooth 0->1 step centered at `center`, transition width `width`."""
    return 1.0 / (1.0 + np.exp(-(x - center) / max(width, 1e-6)))


def _phase_schedule(
    n_steps: int,
    rng: np.random.Generator,
    control: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a per-step (phase_progress, phase_id) schedule.

    phase_progress: continuous value in [0, 3] interpolating across
    NORMAL(0) -> SUSPICIOUS(1) -> DANGER(2) -> EMERGENCY(3), smoothed.
    phase_id: nearest discrete phase for ground-truth labeling.

    If `control` is True, the scenario stays near NORMAL, with an optional
    brief wobble into SUSPICIOUS that recovers (false-alarm exposure).
    """
    t = np.arange(n_steps, dtype=float)

    if control:
        progress = np.zeros(n_steps)
        if rng.random() < 0.5:
            # brief SUSPICIOUS excursion that recovers — teaches detectors
            # that not every deviation is an escalation.
            center = rng.uniform(0.3, 0.7) * n_steps
            width = rng.uniform(8, 20)
            bump_height = rng.uniform(0.6, 1.1)  # stays sub-DANGER
            span = rng.uniform(15, 40)
            progress = bump_height * np.exp(-0.5 * ((t - center) / span) ** 2)
        progress += rng.normal(0, 0.03, n_steps)
        progress = np.clip(progress, 0, 1.4)
        phase_id = np.clip(np.round(progress), 0, 3).astype(int)
        return progress, phase_id

    # Escalating scenario: choose 3 monotonically increasing transition
    # points (NORMAL->SUSP, SUSP->DANGER, DANGER->EMERGENCY) as fractions
    # of total duration, each with its own smoothing width.
    fracs = np.sort(rng.uniform(0.35, 0.95, 3))
    fracs[0] = max(fracs[0], 0.30)
    centers = fracs * n_steps
    widths = rng.uniform(4, 12, 3)

    progress = (
        _logistic(t, centers[0], widths[0])
        + _logistic(t, centers[1], widths[1])
        + _logistic(t, centers[2], widths[2])
    )
    phase_id = np.clip(np.round(progress), 0, 3).astype(int)
    return progress, phase_id


def _risk_score(progress: np.ndarray, phase_id: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map continuous phase progress to a risk score, with within-phase noise."""
    frac = np.clip(progress - phase_id, -1, 1)  # position within/around phase
    risk = np.zeros_like(progress)
    for i, p in enumerate(PHASES):
        lo, hi = PHASE_RISK_RANGE[p]
        mask = phase_id == i
        if not np.any(mask):
            continue
        local = np.clip(0.5 + frac[mask], 0, 1)  # smooth interpolation inside band
        risk[mask] = lo + local * (hi - lo)
    risk += rng.normal(0, 0.015, len(risk))
    return np.clip(risk, 0.0, 1.0)


def _deviation_fraction(progress: np.ndarray, phase_id: np.ndarray) -> np.ndarray:
    """Per-step target deviation fraction from baseline, interpolated within phase."""
    frac = np.clip(progress - phase_id, 0, 1)
    dev = np.zeros_like(progress)
    for i, p in enumerate(PHASES):
        lo, hi = PHASE_DEVIATION_RANGE[p]
        mask = phase_id == i
        dev[mask] = lo + frac[mask] * (hi - lo)
    return dev


def _variance_mult(phase_id: np.ndarray, progress: np.ndarray) -> np.ndarray:
    frac = np.clip(progress - phase_id, 0, 1)
    mult = np.zeros_like(progress)
    for i, p in enumerate(PHASES):
        mask = phase_id == i
        lo = PHASE_VARIANCE_MULT[PHASES[max(i - 1, 0)]]
        hi = PHASE_VARIANCE_MULT[p]
        mult[mask] = lo + frac[mask] * (hi - lo)
    return mult


def generate_scenario(
    n_steps: int | None = None,
    lead_category: str | None = None,
    control: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a single scenario.

    Parameters
    ----------
    n_steps : length of the series (200-500 if not given)
    lead_category : which category ('behavioral'|'digital'|'operational'
        |'environmental') drives the escalation; random if None. Ignored
        when control=True (no escalation to lead).
    control : if True, generate a non-incident (or false-alarm) control run.
    seed : RNG seed for reproducibility.

    Returns
    -------
    DataFrame with one row per step: `step`, one column per channel,
    `phase`, `phase_id`, `risk_score`, `is_incident`, `lead_category`.
    """
    rng = np.random.default_rng(seed)
    if n_steps is None:
        n_steps = int(rng.integers(200, 501))
    if lead_category is None:
        lead_category = rng.choice(CATEGORIES)

    progress, phase_id = _phase_schedule(n_steps, rng, control=control)
    risk = _risk_score(progress, phase_id, rng)
    dev_frac = _deviation_fraction(progress, phase_id)
    var_mult = _variance_mult(phase_id, progress)

    data = {}
    for ch in CHANNELS:
        is_lead = ch.category == lead_category
        # Lead category gets full deviation; others get coupled "echo"
        # deviation at reduced magnitude (simulates cascading effects
        # across subsystems, which is what CB/C2 channels are built to
        # detect downstream).
        coupling = 1.0 if is_lead else rng.uniform(0.15, 0.35)
        eff_dev = dev_frac * coupling

        noise_std = ch.noise_std * np.sqrt(np.maximum(var_mult, 1e-6))
        noise_std = noise_std * (1.0 if is_lead else (0.6 + 0.4 * coupling))
        noise = rng.normal(0, 1, n_steps) * noise_std

        mean_shift = ch.direction * eff_dev * ch.baseline

        if ch.bursty and is_lead:
            # Additional Poisson-ish burst component layered on top,
            # scaling with deviation fraction (captures things like
            # auth-failure spikes or gas leak puffs rather than smooth drift).
            burst_rate = eff_dev * 3.0
            bursts = rng.poisson(np.maximum(burst_rate, 0)) * ch.baseline * 0.8
            mean_shift = mean_shift + bursts

        series = ch.baseline + mean_shift + noise
        if ch.min_value is not None:
            series = np.maximum(series, ch.min_value)
        if ch.max_value is not None:
            series = np.minimum(series, ch.max_value)
        data[ch.name] = series

    df = pd.DataFrame(data)
    df.insert(0, "step", np.arange(n_steps))
    df["phase"] = [PHASES[i] for i in phase_id]
    df["phase_id"] = phase_id
    df["risk_score"] = risk
    df["is_incident"] = np.isin(df["phase"], ["DANGER", "EMERGENCY"])
    df["lead_category"] = lead_category if not control else "none"
    df["scenario_seed"] = seed if seed is not None else -1
    return df


def generate_dataset(
    n_scenarios: int = 40,
    control_fraction: float = 0.25,
    seed: int = 0,
) -> list[pd.DataFrame]:
    """Generate a list of scenario DataFrames, each with a unique scenario_id."""
    rng = np.random.default_rng(seed)
    scenarios = []
    n_control = int(round(n_scenarios * control_fraction))
    n_escalating = n_scenarios - n_control
    plan = [True] * n_control + [False] * n_escalating
    rng.shuffle(plan)

    for i, is_control in enumerate(plan):
        s = int(rng.integers(0, 2**31 - 1))
        df = generate_scenario(control=is_control, seed=s)
        df["scenario_id"] = i
        scenarios.append(df)
    return scenarios


if __name__ == "__main__":
    df = generate_scenario(seed=42)
    print(df.head())
    print("\nphase counts:\n", df["phase"].value_counts())
    print("\nlead category:", df["lead_category"].iloc[0])
    print("shape:", df.shape)
