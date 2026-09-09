import React, { useMemo, useState, useEffect, useRef } from "react";
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, ReferenceArea,
  ResponsiveContainer, Tooltip,
} from "recharts";

/* ---------------------------------------------------------------------
   Token system — instrument-panel palette
--------------------------------------------------------------------- */
const COLORS = {
  bg: "#0B0E11",
  panel: "#12161B",
  panel2: "#0F1317",
  line: "#232A31",
  lineFaint: "#1A1F24",
  text: "#E7ECEF",
  muted: "#8A96A3",
  dim: "#565F68",
  NORMAL: "#4FD1A5",
  SUSPICIOUS: "#E8B339",
  DANGER: "#E8703A",
  EMERGENCY: "#E14B4B",
};
const PHASES = ["NORMAL", "SUSPICIOUS", "DANGER", "EMERGENCY"];
const CATEGORIES = ["behavioral", "digital", "operational", "environmental"];

const CHANNELS = [
  { name: "movement_speed", cat: "behavioral", base: 1.4, dir: -1 },
  { name: "keystroke_interval_ms", cat: "behavioral", base: 220, dir: 1 },
  { name: "gaze_fixation_ms", cat: "behavioral", base: 350, dir: -1 },
  { name: "auth_failures_per_min", cat: "digital", base: 0.2, dir: 1 },
  { name: "network_bytes_kbps", cat: "digital", base: 500, dir: 1 },
  { name: "system_event_rate", cat: "digital", base: 12, dir: 1 },
  { name: "vibration_mms", cat: "operational", base: 2.0, dir: 1 },
  { name: "temperature_c", cat: "operational", base: 45, dir: 1 },
  { name: "pressure_kpa", cat: "operational", base: 101, dir: -1 },
  { name: "gas_conc_ppm", cat: "environmental", base: 5.0, dir: 1 },
  { name: "humidity_pct", cat: "environmental", base: 45, dir: 1 },
  { name: "noise_level_db", cat: "environmental", base: 55, dir: 1 },
];

/* ---------------------------------------------------------------------
   Deterministic RNG (mulberry32) so a given seed always reproduces
--------------------------------------------------------------------- */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function logistic(x, center, width) {
  return 1 / (1 + Math.exp(-(x - center) / Math.max(width, 1e-6)));
}

/* ---------------------------------------------------------------------
   Scenario generator — mirrors data/generators.py's shape closely
   enough to drive a faithful, live demo of the pipeline's behavior.
--------------------------------------------------------------------- */
function generateScenario({ nSteps = 320, leadCategory = "operational", seed = 42 }) {
  const rng = mulberry32(seed);
  const fracs = [
    0.3 + rng() * 0.2,
    0.5 + rng() * 0.25,
    0.72 + rng() * 0.2,
  ].sort((a, b) => a - b);
  const centers = fracs.map((f) => f * nSteps);
  const widths = [4 + rng() * 8, 4 + rng() * 8, 4 + rng() * 8];

  const rows = [];
  for (let t = 0; t < nSteps; t++) {
    const progress =
      logistic(t, centers[0], widths[0]) +
      logistic(t, centers[1], widths[1]) +
      logistic(t, centers[2], widths[2]);
    const phaseId = Math.min(3, Math.round(progress));
    const frac = Math.max(0, Math.min(1, progress - phaseId));

    const devRange = [[0, 0.03], [0.05, 0.15], [0.3, 0.5], [0.5, 0.9]][phaseId];
    const devFrac = devRange[0] + frac * (devRange[1] - devRange[0]);
    const varMult = [1.0, 1.3, 2.0, 3.2][phaseId];

    const riskRange = [[0, 0.15], [0.15, 0.4], [0.4, 0.75], [0.75, 1.0]][phaseId];
    const local = Math.max(0, Math.min(1, 0.5 + frac));
    const risk = Math.max(0, Math.min(1,
      riskRange[0] + local * (riskRange[1] - riskRange[0]) + gauss(rng) * 0.015));

    const channelVals = {};
    for (const ch of CHANNELS) {
      const isLead = ch.cat === leadCategory;
      const coupling = isLead ? 1.0 : 0.15 + rng() * 0.2;
      const effDev = devFrac * coupling;
      const noiseStd = (ch.base * 0.06) * Math.sqrt(varMult) * (isLead ? 1 : 0.6 + 0.4 * coupling);
      const noise = gauss(rng) * noiseStd;
      const meanShift = ch.dir * effDev * ch.base;
      let v = ch.base + meanShift + noise;
      if (ch.name !== "temperature_c") v = Math.max(v, 0);
      if (ch.name === "humidity_pct") v = Math.min(v, 100);
      channelVals[ch.name] = v;
    }

    rows.push({
      step: t,
      phaseId,
      phase: PHASES[phaseId],
      risk,
      ...channelVals,
    });
  }
  return rows;
}

/* ---------------------------------------------------------------------
   Derived signals — lightweight stand-ins for the real detectors,
   computed causally over a trailing window so the shapes track the
   documented behavior of each component.
--------------------------------------------------------------------- */
function deriveSignals(rows, leadCategory) {
  const n = rows.length;
  const leadChannels = CHANNELS.filter((c) => c.cat === leadCategory).map((c) => c.name);
  const win = 15;

  const out = rows.map((r) => ({ ...r }));

  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - win + 1);
    const window = rows.slice(lo, i + 1);
    // rolling z-like deviation on lead channels -> anomaly proxies
    let devSum = 0;
    for (const name of leadChannels) {
      const ch = CHANNELS.find((c) => c.name === name);
      const vals = window.map((w) => w[name]);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      devSum += Math.abs(mean - ch.base) / (ch.base * 0.06 + 1e-6);
    }
    const devAvg = devSum / Math.max(leadChannels.length, 1);
    const ifScore = 1 / (1 + Math.exp(-0.35 * (devAvg - 4)));
    const aeScore = 1 / (1 + Math.exp(-0.3 * (devAvg - 5)));

    // entropy-style channels, phase-correlated with noise for texture
    const p = rows[i].phaseId, frac = Math.max(0, Math.min(1, rows[i].risk));
    const jitter = (k) => (Math.sin(i * 0.7 + k) * 0.05);
    const c1 = Math.max(0, Math.min(1, 0.12 + p * 0.12 + jitter(1)));
    const c2 = Math.max(0, Math.min(1, Math.pow(frac, 1.4) * 0.9 + jitter(2)));
    const c3 = Math.max(0, Math.min(1, Math.pow(frac, 1.1) * 0.95 + jitter(3)));
    const cb = Math.max(0, Math.min(1, Math.pow(frac, 1.6) * 0.85 + jitter(4)));
    const fCER = (c1 + c2 + c3 + cb) / 4;

    const rules = Math.max(0, Math.min(1, (devAvg - 3) / 10));

    const phaseProbs = { NORMAL: 0, SUSPICIOUS: 0, DANGER: 0, EMERGENCY: 0 };
    const spread = 0.18;
    for (let k = 0; k < 4; k++) {
      const dist = Math.abs(k - (p + (frac - 0.5)));
      phaseProbs[PHASES[k]] = Math.exp(-dist / spread);
    }
    const psum = Object.values(phaseProbs).reduce((a, b) => a + b, 0);
    Object.keys(phaseProbs).forEach((k) => (phaseProbs[k] /= psum));

    const PHASE_RISK_MAP = { NORMAL: 0.1, SUSPICIOUS: 0.4, DANGER: 0.7, EMERGENCY: 0.95 };
    const phaseRisk = PHASES.reduce((s, ph) => s + phaseProbs[ph] * PHASE_RISK_MAP[ph], 0);

    const W = { anomaly: 0.2, phase: 0.3, entropy: 0.3, rules: 0.2 };
    const contribAnomaly = W.anomaly * ((ifScore + aeScore) / 2);
    const contribPhase = W.phase * phaseRisk;
    const contribEntropy = W.entropy * fCER;
    const contribRules = W.rules * rules;
    const sentinelRisk = contribAnomaly + contribPhase + contribEntropy + contribRules;

    Object.assign(out[i], {
      ifScore, aeScore, c1, c2, c3, cb, fCER,
      rules, phaseProbs, sentinelRisk,
      contribAnomaly, contribPhase, contribEntropy, contribRules,
    });
  }

  // persistent alert: fCER above threshold for 5 consecutive steps
  let run = 0;
  for (let i = 0; i < n; i++) {
    if (out[i].fCER >= 0.45) run += 1; else run = 0;
    out[i].persistentAlert = run >= 5;
  }
  return out;
}

/* ---------------------------------------------------------------------
   Small presentational helpers
--------------------------------------------------------------------- */
function phaseColor(phase) { return COLORS[phase] || COLORS.muted; }

function RiskDial({ value, phase }) {
  const r = 54, c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(1, value));
  const color = phaseColor(phase);
  return (
    <svg width="140" height="140" viewBox="0 0 140 140">
      <circle cx="70" cy="70" r={r} fill="none" stroke={COLORS.line} strokeWidth="8" />
      <circle
        cx="70" cy="70" r={r} fill="none" stroke={color} strokeWidth="8"
        strokeDasharray={`${c}`} strokeDashoffset={`${c * (1 - pct)}`}
        strokeLinecap="butt" transform="rotate(-90 70 70)"
        style={{ transition: "stroke-dashoffset 120ms linear" }}
      />
      <text x="70" y="66" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace"
        fontSize="26" fill={COLORS.text}>{(pct * 100).toFixed(0)}</text>
      <text x="70" y="86" textAnchor="middle" fontFamily="'IBM Plex Mono', monospace"
        fontSize="10" letterSpacing="1" fill={COLORS.muted}>SENTINEL RISK</text>
    </svg>
  );
}

function Trace({ data, dataKey, color, height = 56, domain = [0, 1] }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
        <YAxis domain={domain} hide />
        <XAxis dataKey="step" hide />
        <Area
          type="monotone" dataKey={dataKey} stroke={color} strokeWidth={1.5}
          fill={color} fillOpacity={0.12} isAnimationActive={false} dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function Field({ label, value, unit }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ fontSize: 10, letterSpacing: "0.06em", color: COLORS.dim }}>{label}</span>
      <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 15, color: COLORS.text }}>
        {value}{unit ? <span style={{ color: COLORS.muted, fontSize: 11 }}> {unit}</span> : null}
      </span>
    </div>
  );
}

/* ---------------------------------------------------------------------
   Main console
--------------------------------------------------------------------- */
export default function SentinelConsole() {
  const [leadCategory, setLeadCategory] = useState("operational");
  const [seed, setSeed] = useState(42);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef(null);

  const rows = useMemo(() => generateScenario({ leadCategory, seed, nSteps: 320 }), [leadCategory, seed]);
  const data = useMemo(() => deriveSignals(rows, leadCategory), [rows, leadCategory]);
  const n = data.length;
  const cur = data[Math.min(step, n - 1)];

  useEffect(() => {
    if (!playing) { clearInterval(intervalRef.current); return; }
    intervalRef.current = setInterval(() => {
      setStep((s) => {
        if (s >= n - 1) { setPlaying(false); return s; }
        return s + 1;
      });
    }, 45);
    return () => clearInterval(intervalRef.current);
  }, [playing, n]);

  useEffect(() => { setStep(0); setPlaying(false); }, [leadCategory, seed]);

  // phase band boundaries for the timeline
  const bands = useMemo(() => {
    const out = [];
    let startIdx = 0;
    for (let i = 1; i <= data.length; i++) {
      if (i === data.length || data[i].phase !== data[startIdx].phase) {
        out.push({ from: data[startIdx].step, to: data[i - 1].step, phase: data[startIdx].phase });
        startIdx = i;
      }
    }
    return out;
  }, [data]);

  const alertEvents = useMemo(() => {
    const evs = [];
    let prev = false;
    data.forEach((d) => {
      if (d.persistentAlert && !prev) evs.push({ step: d.step, phase: d.phase });
      prev = d.persistentAlert;
    });
    return evs;
  }, [data]);

  const entropyChannels = [
    { key: "c1", label: "C1 · causal entropic response" },
    { key: "c2", label: "C2 · structural coupling covariance" },
    { key: "c3", label: "C3 · residual z-score" },
    { key: "cb", label: "CB · correlation break" },
  ];

  const font = "'IBM Plex Sans', -apple-system, sans-serif";
  const mono = "'IBM Plex Mono', 'SF Mono', monospace";

  return (
    <div style={{
      background: COLORS.bg, color: COLORS.text, fontFamily: font,
      padding: 20, borderRadius: 4, maxWidth: 980, margin: "0 auto",
    }}>
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');
        .sentinel-btn {
          background: ${COLORS.panel}; color: ${COLORS.text}; border: 1px solid ${COLORS.line};
          font-family: ${mono}; font-size: 11px; padding: 6px 10px; cursor: pointer;
          letter-spacing: 0.04em;
        }
        .sentinel-btn:hover { border-color: ${COLORS.muted}; }
        .sentinel-btn.active { border-color: ${COLORS.NORMAL}; color: ${COLORS.NORMAL}; }
        .sentinel-select {
          background: ${COLORS.panel}; color: ${COLORS.text}; border: 1px solid ${COLORS.line};
          font-family: ${mono}; font-size: 11px; padding: 6px 8px;
        }
        input[type=range].sentinel-range {
          -webkit-appearance: none; width: 100%; height: 2px; background: ${COLORS.line};
        }
        input[type=range].sentinel-range::-webkit-slider-thumb {
          -webkit-appearance: none; width: 10px; height: 10px; background: ${COLORS.text};
          border-radius: 0; cursor: pointer; margin-top: -4px;
        }
      `}</style>

      {/* ---------- header ---------- */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        borderBottom: `1px solid ${COLORS.line}`, paddingBottom: 14, marginBottom: 16, flexWrap: "wrap", gap: 10,
      }}>
        <div>
          <div style={{ fontSize: 13, letterSpacing: "0.12em", color: COLORS.muted }}>SENTINEL</div>
          <div style={{ fontSize: 11, color: COLORS.dim, fontFamily: mono }}>
            early warning console · scenario seed {seed}
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 11, color: COLORS.dim, fontFamily: mono }}>lead category</span>
          <select
            className="sentinel-select"
            value={leadCategory}
            onChange={(e) => setLeadCategory(e.target.value)}
          >
            {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <button className="sentinel-btn" onClick={() => setSeed((s) => s + 1)}>new run</button>
        </div>
      </div>

      {/* ---------- top: dial + phase strip + fields ---------- */}
      <div style={{ display: "flex", gap: 20, marginBottom: 18, flexWrap: "wrap" }}>
        <div style={{
          background: COLORS.panel, border: `1px solid ${COLORS.line}`, padding: 16,
          display: "flex", flexDirection: "column", alignItems: "center", gap: 8, minWidth: 172,
        }}>
          <RiskDial value={cur.sentinelRisk} phase={cur.phase} />
          <div style={{
            fontFamily: mono, fontSize: 11, letterSpacing: "0.08em", color: phaseColor(cur.phase),
            border: `1px solid ${phaseColor(cur.phase)}`, padding: "3px 10px",
          }}>{cur.phase}</div>
        </div>

        <div style={{
          flex: 1, minWidth: 320, background: COLORS.panel, border: `1px solid ${COLORS.line}`,
          padding: 16, display: "flex", flexDirection: "column", gap: 14,
        }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 14 }}>
            <Field label="STEP" value={`${cur.step} / ${n - 1}`} />
            <Field label="GROUND TRUTH RISK" value={cur.risk.toFixed(3)} />
            <Field label="FUSED RISK" value={cur.sentinelRisk.toFixed(3)} />
            <Field label="FCER" value={cur.fCER.toFixed(3)} />
            <Field label="IF ANOMALY" value={cur.ifScore.toFixed(3)} />
            <Field label="AE ANOMALY" value={cur.aeScore.toFixed(3)} />
          </div>
          <div>
            <div style={{ display: "flex", gap: 6 }}>
              <button className="sentinel-btn" onClick={() => setPlaying((p) => !p)}>
                {playing ? "pause" : "play"}
              </button>
              <input
                className="sentinel-range" type="range" min={0} max={n - 1} value={step}
                onChange={(e) => { setPlaying(false); setStep(Number(e.target.value)); }}
                style={{ flex: 1, alignSelf: "center" }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* ---------- timeline with phase bands ---------- */}
      <div style={{ marginBottom: 18 }}>
        <div style={{ fontSize: 11, letterSpacing: "0.08em", color: COLORS.dim, marginBottom: 6, fontFamily: mono }}>
          RISK TIMELINE
        </div>
        <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.line}`, padding: "10px 12px" }}>
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
              onMouseDown={(e) => e && e.activeLabel != null && setStep(e.activeLabel)}>
              <XAxis dataKey="step" hide />
              <YAxis domain={[0, 1]} hide />
              {bands.map((b, i) => (
                <ReferenceArea key={i} x1={b.from} x2={b.to} y1={0} y2={1}
                  fill={phaseColor(b.phase)} fillOpacity={0.08} stroke="none" />
              ))}
              <ReferenceArea x1={cur.step} x2={cur.step} y1={0} y2={1} stroke={COLORS.text} strokeOpacity={0.6} />
              <Line type="monotone" dataKey="risk" stroke={COLORS.muted} strokeWidth={1} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="sentinelRisk" stroke={COLORS.EMERGENCY} strokeWidth={1.6} dot={false} isAnimationActive={false} />
              <Tooltip
                contentStyle={{ background: COLORS.panel2, border: `1px solid ${COLORS.line}`, fontFamily: mono, fontSize: 11 }}
                labelFormatter={(l) => `step ${l}`}
                formatter={(v, k) => [Number(v).toFixed(3), k === "risk" ? "ground truth" : "sentinel_risk"]}
              />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", gap: 16, marginTop: 6, flexWrap: "wrap" }}>
            {PHASES.map((p) => (
              <div key={p} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{ width: 8, height: 8, background: phaseColor(p), display: "inline-block" }} />
                <span style={{ fontSize: 10, color: COLORS.muted, fontFamily: mono }}>{p}</span>
              </div>
            ))}
            <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 12, height: 1.5, background: COLORS.muted, display: "inline-block" }} />
              <span style={{ fontSize: 10, color: COLORS.muted, fontFamily: mono }}>ground truth</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 12, height: 1.5, background: COLORS.EMERGENCY, display: "inline-block" }} />
              <span style={{ fontSize: 10, color: COLORS.muted, fontFamily: mono }}>sentinel_risk</span>
            </div>
          </div>
        </div>
      </div>

      {/* ---------- entropy channels + attribution ---------- */}
      <div style={{ display: "flex", gap: 18, marginBottom: 18, flexWrap: "wrap" }}>
        <div style={{ flex: 2, minWidth: 320 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.08em", color: COLORS.dim, marginBottom: 6, fontFamily: mono }}>
            CAUSAL ENTROPY DIVERGENCE — 4 CHANNELS
          </div>
          <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.line}` }}>
            {entropyChannels.map((c, i) => (
              <div key={c.key} style={{
                display: "flex", alignItems: "center", gap: 12, padding: "8px 12px",
                borderBottom: i < 3 ? `1px solid ${COLORS.lineFaint}` : "none",
              }}>
                <div style={{ width: 150, fontSize: 10, color: COLORS.muted, fontFamily: mono }}>{c.label}</div>
                <div style={{ flex: 1 }}>
                  <Trace data={data} dataKey={c.key} color={COLORS.NORMAL} height={40} />
                </div>
                <div style={{ width: 44, textAlign: "right", fontFamily: mono, fontSize: 12 }}>
                  {cur[c.key].toFixed(2)}
                </div>
              </div>
            ))}
            <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 12px" }}>
              <div style={{ width: 150, fontSize: 10, color: COLORS.text, fontFamily: mono }}>fCER (fused)</div>
              <div style={{ flex: 1 }}>
                <Trace data={data} dataKey="fCER" color={cur.persistentAlert ? COLORS.EMERGENCY : COLORS.SUSPICIOUS} height={40} />
              </div>
              <div style={{ width: 44, textAlign: "right", fontFamily: mono, fontSize: 13, color: COLORS.text }}>
                {cur.fCER.toFixed(2)}
              </div>
            </div>
          </div>
          {cur.persistentAlert && (
            <div style={{
              marginTop: 8, fontFamily: mono, fontSize: 11, color: COLORS.EMERGENCY,
              border: `1px solid ${COLORS.EMERGENCY}`, padding: "6px 10px", display: "inline-block",
            }}>
              persistent alert — fCER sustained above threshold
            </div>
          )}
        </div>

        <div style={{ flex: 1, minWidth: 220 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.08em", color: COLORS.dim, marginBottom: 6, fontFamily: mono }}>
            RISK ATTRIBUTION
          </div>
          <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.line}`, padding: 12 }}>
            {[
              { key: "contribAnomaly", label: "anomaly (IF+AE)", color: "#5DCAA5" },
              { key: "contribPhase", label: "phase transition", color: "#85B7EB" },
              { key: "contribEntropy", label: "entropy (fCER)", color: "#EF9F27" },
              { key: "contribRules", label: "rules backstop", color: "#B4B2A9" },
            ].map((row) => {
              const w = Math.min(100, (cur[row.key] / Math.max(cur.sentinelRisk, 1e-6)) * 100);
              return (
                <div key={row.key} style={{ marginBottom: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: COLORS.muted, fontFamily: mono, marginBottom: 3 }}>
                    <span>{row.label}</span>
                    <span>{cur[row.key].toFixed(3)}</span>
                  </div>
                  <div style={{ height: 6, background: COLORS.lineFaint }}>
                    <div style={{ width: `${w}%`, height: "100%", background: row.color }} />
                  </div>
                </div>
              );
            })}
            <div style={{ borderTop: `1px solid ${COLORS.line}`, marginTop: 8, paddingTop: 8, display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 12 }}>
              <span style={{ color: COLORS.muted }}>sentinel_risk</span>
              <span>{cur.sentinelRisk.toFixed(3)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* ---------- alert log ---------- */}
      <div>
        <div style={{ fontSize: 11, letterSpacing: "0.08em", color: COLORS.dim, marginBottom: 6, fontFamily: mono }}>
          ALERT LOG
        </div>
        <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.line}` }}>
          {alertEvents.length === 0 && (
            <div style={{ padding: 12, fontSize: 12, color: COLORS.dim, fontFamily: mono }}>
              no persistent alerts raised in this run
            </div>
          )}
          {alertEvents.map((ev, i) => (
            <div key={i} onClick={() => setStep(ev.step)} style={{
              display: "flex", justifyContent: "space-between", padding: "8px 12px", cursor: "pointer",
              borderBottom: i < alertEvents.length - 1 ? `1px solid ${COLORS.lineFaint}` : "none",
              fontFamily: mono, fontSize: 12,
            }}>
              <span style={{ color: COLORS.EMERGENCY }}>fCER persistent alert</span>
              <span style={{ color: COLORS.muted }}>step {ev.step} · phase {ev.phase}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
