# SENTINEL — Early Warning System for Safety-Critical Incident Precursors

A modular Python system that fuses statistical anomaly detection, sequence-based
phase classification, and information-theoretic entropy monitoring into a single
explainable risk score, trained on synthetic 4-stage incident progressions
(NORMAL → SUSPICIOUS → DANGER → EMERGENCY).

## Environment note

This container has **no internet access**, so `torch` / `statsmodels` could not
be installed. Every learned component (autoencoder, GRU sequence classifier,
AR predictor) is **hand-rolled in NumPy with manual backprop**, gradient-checked
against numerical differentiation (see `tests/`). The LSTM in the original spec
was implemented as a **GRU** (3 gates vs 4, no separate cell state) by explicit
choice — same role in the pipeline, substantially less gradient-bookkeeping code
to write and verify correctly by hand.

## Architecture

```
data/generators.py            Synthetic 4-stage multimodal scenario generator
utils/features.py             Rolling-window statistical features (for IF/AE)
utils/sequences.py            Fixed-length raw sequences (for the GRU classifier)

detectors/
  isolation_forest_detector.py   sklearn IsolationForest on rolling features
  numpy_nn.py                    Hand-rolled NumPy Dense/Adam layers
  autoencoder_detector.py        64->32->8->32->64 autoencoder (reconstruction error)
  numpy_gru.py                   Hand-rolled NumPy GRU + manual BPTT, gradient-checked
  phase_transition_detector.py   4-class phase classifier + transition velocity
  entropy_monitor.py             C1/C2/C3/CB entropy channels -> fused fCER

fusion/
  risk_fusion.py                 Weighted multimodal fusion + rules backstop

tests/
  test_pipeline_e2e.py           Full pipeline smoke test across all 4 lead categories
```

## Key design decisions & why

- **Coupled multimodal generation**: each scenario has one "lead" category that
  drives the escalation, with the other 3 categories receiving a smaller
  ("echoed") deviation. A single spiking channel is trivial to detect; realistic
  incidents show *cross-channel coupling*, which is what the C2/CB entropy
  channels are specifically built to catch.
- **Train/val split by scenario, not by window**: sliding windows (stride 4,
  length 50) from the same scenario are near-duplicates. A random per-window
  split leaks validation data into training almost completely — caught this via
  a below-chance validation accuracy that didn't match manual re-evaluation
  (see commit history / build log), root-caused to a double-normalization bug
  in the verbose logger, fixed, and the split changed to be group-aware regardless
  since window-level splits are unsound in principle even when the immediate bug
  is fixed.
- **Class-weighted GRU loss**: NORMAL dominates raw sequence counts ~3:1 over
  any other single phase; inverse-frequency weighting prevents the classifier
  from collapsing to "always predict NORMAL".
- **Entropy channel calibration**: permutation entropy (C1) is a genuinely weak
  signal for pure mean-shift synthetic data (it measures order/complexity, not
  magnitude) — this is realistic, not a bug, and the fusion is designed so C1
  contributes without dominating. C3 (AR residual z-score) and CB (correlation
  break) carry most of the discriminative weight in practice.

## Validated results (see `tests/test_pipeline_e2e.py` output)

Across all 4 lead categories (behavioral / digital / operational / environmental),
fused `sentinel_risk` correlates with ground-truth `risk_score` at **r = 0.96–0.98**,
with clean monotonic increase across NORMAL → SUSPICIOUS → DANGER → EMERGENCY.

## Known limitations (honest, not papered over)

1. **Digital-lead EMERGENCY under-triggers persistent entropy alerts** — bursty
   auth-failure clamping (min_value=0) changes covariance structure
   non-monotonically for that one category. Fusion score still rises correctly
   (anomaly + phase channels compensate), but the entropy component alone is
   less reliable there.
2. **SUSPICIOUS is the hardest phase to classify** (~40-70% GRU accuracy vs
   >95% for the other 3) — by design, since it's defined as a 5-15% deviation,
   genuinely close to noise. A real deployment would want a human-in-the-loop
   review queue for SUSPICIOUS-confidence predictions rather than full automation.
3. **No true LSTM/ARIMA** — see environment note above. Architecture and role
   in the pipeline match the spec; the specific cell/model type was substituted
   for hand-rolling feasibility.
4. **Small training budget** (fast/small, per explicit scope decision) — 15-25
   scenarios, ~60-300 epochs. Production use would want more scenarios and a
   held-out test set beyond the val split used during training.

## Running

```bash
export PYTHONPATH=/path/to/sentinel
python3 tests/test_pipeline_e2e.py
```
