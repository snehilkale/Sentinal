"""Quick end-to-end smoke test of the full SENTINEL pipeline."""
import time
import numpy as np
import pandas as pd

from generators import generate_dataset, generate_scenario
from isolation_forest_detector import IsolationForestDetector
from autoencoder_detector import AutoencoderDetector
from phase_transition_detector import PhaseTransitionDetector
from entropy_monitor import EntropyMonitor
from risk_fusion import RiskFuser

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

t0 = time.time()
print("Generating training data...")
train = generate_dataset(n_scenarios=25, seed=10)

print("Training IsolationForestDetector...")
if_det = IsolationForestDetector(window=15).fit(train)

print("Training AutoencoderDetector...")
ae_det = AutoencoderDetector(window=15, seed=1).fit(train, epochs=250, verbose=False)

print("Training PhaseTransitionDetector...")
phase_det = PhaseTransitionDetector(seq_len=50, hidden_dim=32, seed=2).fit(train, epochs=60, verbose=False)

print("Fitting EntropyMonitor...")
entropy_mon = EntropyMonitor().fit(train)

print(f"All components trained in {time.time()-t0:.1f}s\n")

fuser = RiskFuser()

for lead in ["operational", "digital", "behavioral", "environmental"]:
    test = generate_scenario(lead_category=lead, seed=4242, n_steps=350)
    if_scored = if_det.score_scenario(test)
    ae_scored = ae_det.score_scenario(test)
    phase_scored = phase_det.score_scenario(test)
    entropy_scored = entropy_mon.score_scenario(test)

    fused = fuser.fuse(test, if_scored, ae_scored, phase_scored, entropy_scored)

    print(f"=== lead category: {lead} ===")
    summary = fused.groupby("phase")[["sentinel_risk", "risk_score", "contrib_anomaly",
                                       "contrib_phase_transition", "contrib_entropy", "contrib_rules"]].mean()
    print(summary.round(3))
    corr = np.corrcoef(fused["sentinel_risk"], fused["risk_score"])[0, 1]
    print(f"correlation(sentinel_risk, ground_truth risk_score) = {corr:.3f}")
    print()

print(f"Total pipeline time: {time.time()-t0:.1f}s")
