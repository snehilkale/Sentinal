# SENTINEL — Early Warning System for Multi-Modal Risk Detection

A comprehensive anomaly detection and risk fusion pipeline that combines multiple detection strategies (Isolation Forest, Autoencoders, Phase Transitions, and Entropy Analysis) to identify escalating system threats across behavioral, digital, operational, and environmental dimensions.

---

## 🎯 Overview

SENTINEL is a **multimodal risk detection framework** designed to:
- Detect anomalies in real-time streaming data
- Identify phase transitions (NORMAL → SUSPICIOUS → DANGER → EMERGENCY)
- Fuse multiple detector outputs into a single, interpretable risk score
- Provide component-level attribution (why is the system alarmed?)

The system trains on normal-phase baseline data and flags deviations, combining:
1. **Statistical outlier detection** (Isolation Forest)
2. **Reconstruction-based anomalies** (Autoencoders)
3. **Phase progression signals** (GRU-based classifier)
4. **Entropy divergence** (causal entropic response framework)

---

## 📁 Project Structure

```
guru/
├── files/                          # Python backend (detectors & pipeline)
│   ├── generators.py               # Scenario generation + baseline data
│   ├── features.py                 # Rolling feature extraction
│   ├── sequences.py                # Time-series windowing for RNN
│   │
│   ├── isolation_forest_detector.py # Component 1: Statistical anomaly
│   ├── autoencoder_detector.py      # Component 2: Reconstruction anomaly
│   ├── phase_transition_detector.py # Component 3: Phase progression
│   ├── entropy_monitor.py           # Component 4: Entropy divergence (C1-C4)
│   │
│   ├── numpy_nn.py                 # Pure-NumPy Autoencoder impl.
│   ├── numpy_gru.py                # Pure-NumPy GRU classifier
│   │
│   ├── risk_fusion.py              # Component 5: Multimodal fusion + rules
│   ├── test_pipeline_e2e.py        # End-to-end smoke test
│   └── README.md
│
├── sentinel-console/               # React dashboard (web frontend)
│   ├── src/
│   │   ├── SentinelConsole.jsx      # Main interactive dashboard
│   │   ├── App.jsx                  # App wrapper
│   │   ├── main.jsx                 # React entry point
│   │   ├── index.css                # Minimal reset
│   │   └── assets/
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── README.md
│
└── README.md (you are here)
```

---

## 🚀 Quick Start

### Python Pipeline (End-to-End Test)

```bash
cd /Users/snehil/Desktop/guru/files

# Run the full pipeline on 4 scenarios (operational, digital, behavioral, environmental)
python3 test_pipeline_e2e.py
```

**Expected output:**
- Training time: ~55 seconds
- Per-scenario correlation with ground-truth risk: 0.96–0.98
- Risk attribution breakdown for each phase

### React Dashboard (Interactive Console)

```bash
cd /Users/snehil/Desktop/guru/sentinel-console

# Start dev server (Vite)
npm run dev

# Open in browser
http://localhost:5175/
```

**Dashboard features:**
- Play/pause timeline animation
- Scrubber to jump to any step
- Lead category selector (behavioral, digital, operational, environmental)
- "New run" to regenerate scenario with new seed
- Real-time risk dial, entropy signals, attribution breakdown
- Alert log (persistent fCER alerts)

---

## 🧠 The 5-Component Pipeline

### 1. **Isolation Forest Detector** (`isolation_forest_detector.py`)
- Trained on NORMAL-phase rolling features only
- Flags statistical outliers regardless of phase
- Output: `ifScore` ∈ [0, 1]
- Pure anomaly detection (no phase awareness)

### 2. **Autoencoder Detector** (`autoencoder_detector.py`)
- Reconstruction-based anomaly (variational AE)
- Trained on NORMAL-phase 15-step rolling windows
- Outputs reconstruction error: `aeScore` ∈ [0, 1]
- Captures complex normal patterns; flags when input doesn't fit

### 3. **Phase Transition Detector** (`phase_transition_detector.py`)
- GRU classifier predicting phase from 50-step window of raw channels
- Output: phase probabilities for [NORMAL, SUSPICIOUS, DANGER, EMERGENCY]
- Also outputs "transition velocity" (rate of phase probability change)
- Learns phase ordering and escalation patterns

### 4. **Entropy Monitor** (`entropy_monitor.py`)
- **4-channel Causal Entropy Divergence framework:**
  - **C1**: Causal entropic response (permutation entropy rate)
  - **C2**: Structural coupling covariance (rolling correlation divergence)
  - **C3**: Residual Z-score (AR model residuals)
  - **CB**: Correlation break (pairwise correlation shift)
- Fused into **fCER** (fused causal entropy response)
- Persistent alert: fCER > threshold for 5 consecutive steps

### 5. **Risk Fuser** (`risk_fusion.py`)
- Combines all 4 detectors via multimodal weighting:
  - Anomaly (IF+AE): 20%
  - Phase transition: 30%
  - Entropy (fCER): 30%
  - Rules backstop: 20%
- Per-phase risk mapping: NORMAL=0.1, SUSPICIOUS=0.4, DANGER=0.7, EMERGENCY=0.95
- Output: `sentinel_risk` ∈ [0, 1] with component attribution

---

## 📊 Scenario Generation

All 4 lead categories share the same phase structure but with different anomaly magnitudes:

```
Phase progression: NORMAL → SUSPICIOUS → DANGER → EMERGENCY

Each phase has a range of:
- Baseline channel deviation
- Noise scaling
- Variance multiplier

Lead category coupling:
- Lead channels: full effect (coupling = 1.0)
- Non-lead channels: 15–35% coupling (weaker signal)
- Deterministic RNG: same seed reproduces exact scenario
```

---

## 🎮 Dashboard Walkthrough

### Header
- **SENTINEL** logo + theme indicator
- **Lead category** dropdown: behavioral, digital, operational, environmental
- **New run** button: increment seed to generate new scenario

### Main Panels
1. **Risk Dial** (left)
   - Animated ring showing `sentinel_risk` as percentage
   - Color: green (NORMAL) → yellow (SUSPICIOUS) → orange (DANGER) → red (EMERGENCY)
   - Current phase badge

2. **Metrics Grid** (right)
   - STEP: current position in scenario
   - GROUND TRUTH RISK: simulated true risk
   - FUSED RISK: SENTINEL's output
   - fCER, IF/AE anomaly scores
   - Play/Pause + Timeline scrubber

3. **Risk Timeline** (below)
   - X-axis: scenario steps (0–319)
   - Y-axis: risk [0, 1]
   - Background: phase bands (NORMAL green → EMERGENCY red)
   - Lines: ground truth (gray) vs. sentinel_risk (red)
   - Interactive: click to jump to step

4. **Entropy Channels** (lower left)
   - C1, C2, C3, CB: individual entropy components
   - fCER: fused entropy signal
   - Color: orange (SUSPICIOUS) → red (EMERGENCY) if persistent alert active

5. **Risk Attribution** (lower right)
   - Stacked bar: anomaly, phase, entropy, rules contributions
   - Percentages of total `sentinel_risk`
   - Shows why alert is raised

6. **Alert Log** (bottom)
   - List of persistent fCER alerts
   - Step number, phase, clickable (jumps timeline)

---

## 🔬 How to Interpret Results

### Correlation with Ground Truth
End-to-end test shows **0.96–0.98 correlation** between `sentinel_risk` and ground-truth `risk_score`:
- **High correlation**: SENTINEL accurately tracks true risk progression
- **Slight variance**: Entropy signals add jitter; persistent alert lag creates phase delays

### Component Attribution
Each detector contributes to final risk:
- **Anomaly** (IF+AE): Catches sudden deviations from baseline
- **Phase**: Weights risk by phase progression (EMERGENCY is highest)
- **Entropy**: Detects gradual coupling/coherence breakdown
- **Rules**: Fallback mechanism for extreme parameter combinations

### Lead Category Effects
- **Operational anomalies**: Vibration, temperature, pressure deviations
- **Digital anomalies**: Auth failures, network spikes, system events
- **Behavioral anomalies**: Mouse speed, keystroke timing, gaze fixation
- **Environmental anomalies**: Gas concentration, humidity, noise

Lead anomalies have full coupling (strong signal); non-lead channels have 15–35% coupling (weaker cross-contamination).

---

## 📈 Performance Metrics

From `test_pipeline_e2e.py`:
- **Training time**: ~55 seconds (25 scenarios, 4 detectors)
- **Correlation (operational)**: 0.963
- **Correlation (digital)**: 0.980
- **Correlation (behavioral)**: 0.983
- **Correlation (environmental)**: 0.980

Mean absolute errors and per-phase attribution breakdowns printed per category.

---

## 🛠️ Technologies

### Backend (Python)
- **NumPy** (core numerical ops, custom NN/GRU)
- **Pandas** (data manipulation)
- **Scikit-learn** (Isolation Forest)
- **Pure-NumPy NN/GRU** (no TensorFlow/PyTorch required)

### Frontend (React)
- **React 19** (UI framework)
- **Recharts** (interactive charts)
- **Vite** (dev server + bundling)
- **CSS-in-JS** (inline styles for theming)

---

## 🎨 Theming

Dashboard supports **light** and **dark** modes (toggle in header):
- **Light**: high-contrast, professional (white/purple accents)
- **Dark**: low-light-friendly, terminal-style (dark navy/neon accents)

Both themes respect WCAG contrast standards.

---

## 🔧 Development Workflow

### Make changes to Python pipeline:
```bash
cd files/
# Edit generator, detector, or fusion logic
# Re-run test
python3 test_pipeline_e2e.py
```

### Make changes to React dashboard:
```bash
cd sentinel-console/
# Edit SentinelConsole.jsx or App.jsx
# Vite hot-reload will refresh automatically
# Changes visible instantly in browser
```

### Rebuild for production:
```bash
cd sentinel-console/
npm run build
# Output in dist/ folder (static HTML/JS)
```

---

## 📝 Files Guide

| File | Purpose |
|------|---------|
| `generators.py` | Scenario creation, baseline data, phase definitions |
| `features.py` | Rolling mean/std/skew/min/max per channel |
| `sequences.py` | Fixed-length windowing for RNN input |
| `numpy_nn.py` | Pure-NumPy Autoencoder (encoder→decoder) |
| `numpy_gru.py` | Pure-NumPy GRU cell (3-gate RNN) |
| `isolation_forest_detector.py` | Scikit-learn Isolation Forest wrapper |
| `autoencoder_detector.py` | AE-based anomaly detection |
| `phase_transition_detector.py` | GRU phase classifier |
| `entropy_monitor.py` | Causal entropy divergence (4 channels + fusion) |
| `risk_fusion.py` | Multimodal risk aggregation + attribution |
| `test_pipeline_e2e.py` | End-to-end integration test (4 scenarios) |
| `SentinelConsole.jsx` | Interactive React dashboard (main component) |
| `App.jsx` | React app wrapper |
| `main.jsx` | React entry point |

---

## 🚨 Troubleshooting

### Python: "ModuleNotFoundError: No module named 'pandas'"
```bash
pip install numpy pandas scikit-learn
```

### React: "localhost refused to connect"
```bash
cd sentinel-console/
npm run dev  # Ensure dev server is running on 5175
```

### React: "Component error: Cannot read properties of undefined"
Usually means a child component didn't receive required props (e.g., `C` for colors). Check that parent passes all expected props.

---

## 📚 Key Papers & References

- **Isolation Forest**: Liu et al., "Isolation Forest" (IEEE Trans. Knowledge Data Eng., 2012)
- **Autoencoders for Anomaly**: Goodfellow et al., "Deep Learning" (MIT Press, 2016)
- **Temporal Phase Detection**: Hochreiter & Schmidhuber, "LSTM" (Neural Computation, 1997)
- **Entropy Divergence**: Cover & Thomas, "Elements of Information Theory" (Wiley, 2006)

---

## 📄 License

SENTINEL is provided as-is for research and educational purposes.

---

## 🎯 Next Steps

1. **Tune detector thresholds** in `risk_fusion.py` (PHASE_RISK_MAP, weights)
2. **Add new channels** in `generators.py` (CHANNELS list)
3. **Implement real data adapter** (replace scenario generation with live stream)
4. **Deploy dashboard** (build React, host on web server)
5. **Add persistence** (save alerts to database)

---

**Built by GitHub Copilot** | SENTINEL Early Warning Console | 2026
