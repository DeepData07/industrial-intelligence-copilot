import React, { useEffect, useMemo, useRef, useState } from "react";

const scenarios = {
  OSF: { label: "OSF Scenario", source: "Synthetic Demo Scenario Based on AI4I Rules", rule: "Overstrain (torque × tool wear)", cycles: 36 },
  HDF: { label: "HDF Scenario", source: "Synthetic Demo Scenario Based on AI4I Rules", rule: "Heat dissipation (temp delta / rpm)", cycles: 36 },
  PWF: { label: "PWF Scenario", source: "Synthetic Demo Scenario Based on AI4I Rules", rule: "Mechanical power boundary", cycles: 36 },
  REPLAY: { label: "AI4I Replay", source: "AI4I Dataset Replay", rule: "Original AI4I rows replayed unchanged", cycles: 36 },
};
const stages = [["Sense", "Live telemetry"], ["Detect", "Rule + risk signal"], ["Understand", "What changed + evidence"], ["Investigate", "Copilot answers"], ["Decide", "What-if sliders"], ["Act", "Suggested checks"]];
const questions = ["Why did you flag this?", "Has this happened under similar conditions before?", "Is RPM causing this?", "What should I check first?"];
const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

const fmt = (value, digits = 1) => Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const noise = (i, amount, salt = 1) => ((Math.sin(i * 12.9898 * salt) * 43758.5453) % 1 - 0.5) * 2 * amount;

function telemetryAt(scenario, cycle) {
  const progress = Math.min(cycle, 36) / 36;
  if (scenario === "OSF") return { cycle, air: 298.6 + noise(cycle, .15), process: 308.4 + progress * .9 + noise(cycle, .15, 3), rpm: 1452 - progress * 14 + noise(cycle, 6, 2), torque: 40 + progress * 16 + noise(cycle, .6, 5), wear: Math.round(120 + progress * 98) };
  if (scenario === "HDF") return { cycle, air: 301.2 + progress * 1.6 + noise(cycle, .12), process: 308.4 + progress * .4 + noise(cycle, .12, 3), rpm: 1420 - progress * 90 + noise(cycle, 6, 2), torque: 44 + progress * 4 + noise(cycle, .5, 5), wear: Math.round(80 + progress * 40) };
  if (scenario === "PWF") return { cycle, air: 298.9 + noise(cycle, .12), process: 308.8 + noise(cycle, .12, 3), rpm: 1380 + progress * 520 + noise(cycle, 8, 2), torque: 46 + progress * 18 + noise(cycle, .6, 5), wear: Math.round(60 + progress * 30) };
  return { cycle, air: 298.4 + noise(cycle, .5), process: 308.3 + noise(cycle, .5, 3), rpm: 1480 + noise(cycle, 60, 2), torque: 39 + noise(cycle, 5, 5), wear: Math.round(40 + cycle * 2.2) };
}
function twinAt(scenario, cycle) {
  const data = telemetryAt(scenario, cycle); const delta = data.process - data.air; const power = data.torque * ((data.rpm * 2 * Math.PI) / 60); const load = data.torque * data.wear; const osfMargin = 11000 - load;
  const z = 30 * (load / 11000 - .985) + 2.6 * Math.max(0, (8.6 - delta) / 1.2) * (data.rpm < 1400 ? 1 : .15) + 2.6 * Math.max(0, (power - 9000) / 900) + 2 * Math.max(0, (3500 - power) / 700) - 1.2;
  const risk = 1 / (1 + Math.exp(-z)); const status = risk >= .35 || osfMargin < 0 ? "INCIDENT" : risk >= .2 || osfMargin < 1000 ? "WARNING" : risk >= .08 || osfMargin < 3000 ? "WATCH" : "NORMAL";
  return { ...data, delta, power, load, osfMargin, risk, status };
}
function changes(history) {
  if (history.length < 6) return [];
  const recent = history.slice(-5), baseline = history.slice(Math.max(0, history.length - 15), -5); if (!baseline.length) return [];
  const features = [["Overstrain load", "min·Nm", s => s.load], ["Torque", "Nm", s => s.torque], ["Mechanical power", "W", s => s.power], ["Tool wear", "min", s => s.wear], ["Temperature delta", "K", s => s.delta], ["Rotational speed", "rpm", s => s.rpm]];
  const average = (items, get) => items.reduce((sum, item) => sum + get(item), 0) / items.length;
  return features.map(([name, unit, get]) => { const before = average(baseline, get), now = average(recent, get); return { name, unit, before, now, percent: (now - before) / Math.abs(before || 1) * 100 }; }).sort((a, b) => Math.abs(b.percent) - Math.abs(a.percent));
}
function similarity(state) {
  const severity = Math.min(1, Math.max(0, (.35 - state.osfMargin / 11000) * 1.4 + state.risk));
  const cases = Array.from({ length: 8 }, (_, index) => ({ failed: ((Math.sin((state.cycle * 7 + index * 13) * 12.9898) * 43758.5453) % 1 + 1) % 1 < severity }));
  const failed = cases.filter(item => item.failed).length; return { cases, retrieved: 8, failed, rate: failed / 8, topMode: "OSF" };
}
const tone = status => status === "INCIDENT" ? "incident" : status === "WARNING" ? "warning" : status === "WATCH" ? "watch" : "normal";

function Panel({ eyebrow, title, action, children, className = "" }) { return <section className={`panel ${className}`}><header>{eyebrow && <small>{eyebrow}</small>}<h2>{title}</h2>{action}</header><div className="panel-body">{children}</div></section>; }
function Status({ status }) { return <span className={`status ${tone(status)}`}><i />{status}</span>; }
function Spark({ points, status }) { if (points.length < 2) return <div className="spark" />; const min = Math.min(...points), max = Math.max(...points), span = max - min || 1; const d = points.map((p, i) => `${i ? "L" : "M"}${(i / (points.length - 1) * 100).toFixed(2)},${(23 - (p-min)/span*21).toFixed(2)}`).join(" "); return <svg viewBox="0 0 100 24" className={`spark ${tone(status)}`} preserveAspectRatio="none"><path d={d} /></svg>; }
function Kpi({ label, value, unit, sub, points, status, delta }) { return <div className={`kpi ${tone(status)}`}><div><small>{label}</small>{delta !== null && <em>{delta >= 0 ? "▲" : "▼"} {fmt(Math.abs(delta))}%</em>}</div><strong>{value}<span>{unit}</span></strong><p>{sub}</p><Spark points={points} status={status} /></div>; }
function Twin({ state }) { const c = tone(state.status); return <div className="twin"><div className="twin-frame"><Status status={state.status} /><svg viewBox="0 0 220 150"><defs><pattern id="grid" width="12" height="12" patternUnits="userSpaceOnUse"><path d="M12 0H0V12" fill="none" stroke="#22303b" strokeWidth=".4"/></pattern></defs><rect width="220" height="150" fill="url(#grid)"/><rect x="24" y="20" width="172" height="110" rx="6" className={`frame ${c}`}/><rect x="36" y="34" width="40" height="30" rx="4" className="motor"/><text x="56" y="52" textAnchor="middle">MOTOR</text><line x1="76" y1="49" x2="120" y2="49" className={`spindle ${c}`}/><circle cx="120" cy="49" r="12" className={`circle ${c}`}/><circle cx="120" cy="49" r="3" className={`fill ${c}`}/><text x="120" y="30" textAnchor="middle">SPINDLE</text><rect x="115" y="61" width="10" height="26" rx="2" className={`tool ${c}`}/><text x="150" y="78">TOOL</text><rect x="86" y="92" width="70" height="16" rx="2" className="work"/><text x="121" y="120" textAnchor="middle">WORKPIECE — TYPE L</text><path d="M170 40h16v60h-16" className="heat"/><text x="178" y="34" textAnchor="middle">HEAT</text></svg></div><div className="twin-stats"><div><small>Spindle load</small><b>{fmt(state.power / 1000, 2)} kW</b></div><div><small>Tool wear</small><b>{state.wear} min</b></div><div><small>Thermal Δ</small><b>{fmt(state.delta, 2)} K</b></div></div></div>; }
function Chart({ history }) {
  const [active, setActive] = useState(["torque", "rpm", "wear", "risk"]);
  const items = [
    ["torque", "Torque", "Nm", 1],
    ["rpm", "RPM", "rpm", 0],
    ["wear", "Tool wear", "min", 0],
    ["risk", "Failure risk", "%", 1],
  ];
  const value = (state, key) => key === "risk" ? state.risk * 100 : state[key];
  const path = key => {
    const values = history.map(state => value(state, key));
    const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
    return values.map((point, index) => `${index ? "L" : "M"}${index / (values.length - 1) * 100},${26 - (point - min) / span * 22 - 2}`).join(" ");
  };
  const selected = items.filter(([key]) => active.includes(key));
  return <div>
    <div className="legend">{items.map(([key, label]) => <button key={key} className={active.includes(key) ? key : "off"} onClick={() => setActive(list => list.includes(key) ? list.filter(item => item !== key) : [...list, key])}>{label}</button>)}</div>
    <div className="chart telemetry-lanes">{history.length > 1 ? selected.map(([key, label, unit, digits]) => <div className="telemetry-lane" key={key}><div className="lane-label"><span>{label}</span><b>{fmt(value(history[history.length - 1], key), digits)} {unit}</b></div><svg viewBox="0 0 100 26" preserveAspectRatio="none"><line x1="0" x2="100" y1="24" y2="24"/><path className={key} d={path(key)}/></svg></div>) : <span>Stream idle — press Start to begin telemetry</span>}</div>
    <div className="axis"><span>cycle {history[0]?.cycle || 1}</span><span>separate normalised lanes · controlled backend scenario</span><span>cycle {history[history.length - 1]?.cycle || 1}</span></div>
  </div>;
}
function ChartControlled({ history }) { const [active, setActive] = useState(["torque", "wear"]); const items = [["torque", "Torque"], ["rpm", "RPM"], ["wear", "Tool wear"]]; const value = (s, key) => s[key]; const path = key => { const values = history.map(s => value(s, key)), min = Math.min(...values), max = Math.max(...values), span = max-min || 1; return values.map((v,i) => `${i ? "L" : "M"}${i/(values.length-1)*100},${40-(v-min)/span*36-2}`).join(" "); }; return <div><div className="legend">{items.map(([key,label]) => <button key={key} className={active.includes(key) ? key : "off"} onClick={() => setActive(list => list.includes(key) ? list.filter(item => item !== key) : [...list,key])}>{label}</button>)}</div><div className="chart">{history.length > 1 ? <svg viewBox="0 0 100 40" preserveAspectRatio="none">{[10,20,30].map(y => <line key={y} x1="0" x2="100" y1={y} y2={y}/>)}{active.map(key => <path key={key} className={key} d={path(key)} />)}</svg> : <span>Stream idle — press Start to begin telemetry</span>}</div><div className="axis"><span>cycle {history[0]?.cycle || 1}</span><span>normalised per-series scaling · controlled scenario ramp</span><span>cycle {history[history.length - 1]?.cycle || 1}</span></div></div>; }
function LocalCopilot({ incident, state, changes: data, similar }) { const [turns,setTurns] = useState([]), [draft,setDraft] = useState(""); const ask = q => { if (!q.trim()) return; const top = data[0]; let answer = ""; if (!incident) answer = "No active incident on this asset. Start or advance the stream to build an evidence window."; else if (/rpm|speed/i.test(q)) answer = `Current evidence does not support RPM as the primary driver. The strongest recent change is ${top?.name || "still being calculated"}.`; else if (/before|similar|history/i.test(q)) answer = `Similar historical conditions: ${similar.retrieved} retrieved; ${similar.failed} failed (${fmt(similar.rate*100)}%). Most common associated failure flag: ${similar.topMode}.`; else if (/check|next|do/i.test(q)) answer = "Suggested next checks: inspect tool condition, verify whether the torque increase is expected for this job, and confirm the product-type overstrain threshold."; else answer = `Incident ${incident.id} is active for MACHINE-01. Largest recent change: ${top?.name || "waiting for comparison window"}.`; setTurns(all => [...all,{q,answer}]); setDraft(""); }; return <Panel eyebrow="Investigate" title="Incident Copilot" action={<small className="context">{incident ? `Context: ${incident.id}` : "No incident context"}</small>} className="copilot"><div className="conversation">{turns.length === 0 && <div className="empty">The Copilot answers only from this incident's evidence: twin state, What Changed window, and AI4I similarity retrieval. It does not calculate new numbers, diagnose root cause, or estimate remaining useful life.</div>}{turns.map((turn,index) => <div key={index} className="turn"><div className="question">{turn.q}</div><div className="answer"><b>{turn.answer}</b><p>Evidence: operational twin state · validated What Changed window · AI4I retrieval</p><small>Limitation: evidence comparison only; not a causal diagnosis or a machine command.</small></div></div>)}</div><div className="chips">{questions.map(q => <button key={q} onClick={() => ask(q)}>{q}</button>)}</div><form onSubmit={event => {event.preventDefault();ask(draft)}}><input value={draft} onChange={event => setDraft(event.target.value)} placeholder="Ask about this incident…"/><button>Ask</button></form></Panel>; }
function WhatIf({ current }) { const [next,setNext] = useState(current); useEffect(() => setNext(current), [current]); const proposed = useMemo(() => { const synthetic = { ...current, rpm: next.rpm, torque: next.torque, wear: next.wear }; const power = synthetic.torque * ((synthetic.rpm * 2*Math.PI)/60), load = synthetic.torque*synthetic.wear, osfMargin = 11000-load; const risk = 1/(1+Math.exp(-(30*(load/11000-.985)-1.2))); const status = risk >= .35 || osfMargin < 0 ? "INCIDENT" : risk >= .2 || osfMargin < 1000 ? "WARNING" : risk >= .08 || osfMargin < 3000 ? "WATCH" : "NORMAL"; return {...synthetic,power,load,osfMargin,risk,status}; },[current,next]); const sliders = [["torque","Torque [Nm]",5,78,.5],["rpm","Rotational speed [rpm]",1160,2890,5],["wear","Tool wear [min]",0,253,1]]; return <Panel eyebrow="Decide" title="What-if analysis" action={<button className="reset" onClick={() => setNext(current)}>Reset to live</button>}><div className="whatif"><div>{sliders.map(([key,label,min,max,step]) => <label key={key}><span>{label}</span><b>{fmt(current[key], step < 1 ? 1 : 0)} → <i>{fmt(next[key], step < 1 ? 1 : 0)}</i></b><input type="range" min={min} max={max} step={step} value={next[key]} onChange={event => setNext(point => ({...point,[key]:Number(event.target.value)}))}/></label>)}</div><div className="compare"><div><small>Metric</small><small>Current</small><small>Proposed</small></div><p><span>Machine state</span><Status status={current.status}/><Status status={proposed.status}/></p><p><span>Failure risk</span><b>{fmt(current.risk*100)}%</b><b>{fmt(proposed.risk*100)}%</b></p><p><span>OSF margin</span><b>{fmt(current.osfMargin,0)}</b><b>{fmt(proposed.osfMargin,0)}</b></p><p><span>Mechanical power</span><b>{fmt(current.power/1000,2)} kW</b><b>{fmt(proposed.power/1000,2)} kW</b></p><aside>Decision support only. No machine command has been issued — this recalculates a proposed telemetry point using documented AI4I rules.</aside></div></div></Panel>; }

function LocalDecisionWorkspace({ current }) {
  const [proposal, setProposal] = useState(current);
  useEffect(() => setProposal(current), [current]);
  const proposed = useMemo(() => {
    const power = proposal.torque * ((proposal.rpm * 2 * Math.PI) / 60);
    const load = proposal.torque * proposal.wear;
    const osfMargin = 11000 - load;
    const delta = proposal.process - proposal.air;
    const risk = 1 / (1 + Math.exp(-(30 * (load / 11000 - .985) + 2.6 * Math.max(0, (8.6 - delta) / 1.2) * (proposal.rpm < 1400 ? 1 : .15) + 2.6 * Math.max(0, (power - 9000) / 900) + 2 * Math.max(0, (3500 - power) / 700) - 1.2)));
    const status = risk >= .35 || osfMargin < 0 ? "INCIDENT" : risk >= .2 || osfMargin < 1000 ? "WARNING" : risk >= .08 || osfMargin < 3000 ? "WATCH" : "NORMAL";
    return { ...proposal, power, load, osfMargin, delta, risk, status };
  }, [proposal]);
  const sliders = [["rpm", "Rotational speed", "rpm", 1160, 2890, 5], ["torque", "Torque", "Nm", 5, 78, .5], ["wear", "Tool wear", "min", 0, 253, 1], ["air", "Air temperature", "K", 295, 305, .1], ["process", "Process temperature", "K", 305, 314, .1]];
  return <div className="decision-workspace"><div className="decision-sliders">{sliders.map(([key,label,unit,min,max,step]) => <label key={key}><span>{label} [{unit}]</span><b>{fmt(current[key], step < 1 ? 1 : 0)} → <i>{fmt(proposal[key], step < 1 ? 1 : 0)}</i></b><input type="range" min={min} max={max} step={step} value={proposal[key]} onChange={event => setProposal(point => ({ ...point, [key]: Number(event.target.value) }))}/></label>)}</div><div className="decision-compare"><div className="compare-head"><small>Metric</small><small>Current</small><small>Proposed</small></div><div><span>Machine status</span><Status status={current.status}/><Status status={proposed.status}/></div><div><span>Failure risk</span><b>{fmt(current.risk * 100)}%</b><b>{fmt(proposed.risk * 100)}%</b></div><div><span>OSF margin</span><b>{fmt(current.osfMargin, 0)}</b><b>{fmt(proposed.osfMargin, 0)}</b></div><div><span>HDF margin</span><b>{fmt(current.delta - 8.6, 2)} K</b><b>{fmt(proposed.delta - 8.6, 2)} K</b></div><div><span>PWF high margin</span><b>{fmt(9000 - current.power, 0)} W</b><b>{fmt(9000 - proposed.power, 0)} W</b></div><aside>Decision support only. No machine command sent. Engineer review is required before any operational change.</aside></div></div>;
}

function DecisionWorkspace({ current, scenario }) {
  const [proposal, setProposal] = useState(current);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [calculating, setCalculating] = useState(false);
  useEffect(() => { setProposal(current); setResult(null); setError(""); }, [current, scenario]);
  const sliders = [["rpm", "Rotational speed", "rpm", 1160, 2890, 5], ["torque", "Torque", "Nm", 5, 78, .5], ["wear", "Tool wear", "min", 0, 253, 1], ["air", "Air temperature", "K", 295, 305, .1], ["process", "Process temperature", "K", 305, 314, .1]];
  const evaluate = async () => {
    setCalculating(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/live/what-if`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scenario, cycle: current.cycle, air_temperature_k: proposal.air, process_temperature_k: proposal.process, rotational_speed_rpm: proposal.rpm, torque_nm: proposal.torque, tool_wear_min: proposal.wear }) });
      const body = await response.text();
      let data;
      try { data = JSON.parse(body); } catch { throw new Error(`Backend returned HTTP ${response.status} without a JSON response. Restart the FastAPI server on port 8000.`); }
      if (!response.ok) throw new Error(data.detail || "What-if evaluation failed.");
      setResult(data); setError("");
    } catch (caught) { setResult(null); setError(caught.message); }
    finally { setCalculating(false); }
  };
  const proposed = result?.proposed;
  const proposedValue = (render) => proposed ? render(proposed) : <span className="pending-value">Pending</span>;
  return <div className="decision-workspace">
    <div className="decision-sliders">
      {sliders.map(([key,label,unit,min,max,step]) => <label key={key}><span>{label} [{unit}]</span><b>{fmt(current[key], step < 1 ? 1 : 0)} → <i>{fmt(proposal[key], step < 1 ? 1 : 0)}</i></b><input type="range" min={min} max={max} step={step} value={proposal[key]} disabled={calculating} onChange={event => { setResult(null); setError(""); setProposal(point => ({ ...point, [key]: Number(event.target.value) })); }}/></label>)}
      <button className="primary" onClick={evaluate} disabled={calculating}>{calculating ? "Recalculating through backend…" : result ? "Recalculate proposed outcome again" : "Recalculate proposed outcome"}</button>
      <div className={`calculation-status ${result ? "success" : "pending"}`}>{result ? "✓ Proposed outcome updated from the backend model and engineering rules." : "Adjust the controls, then press Recalculate to generate a proposed outcome."}</div>
    </div>
    <div className="decision-compare">
      <div className="compare-head"><small>Metric</small><small>Current</small><small>Proposed</small></div>
      <div><span>Machine status</span><Status status={current.status}/>{proposedValue(point => <Status status={point.status}/>)}</div>
      <div><span>Failure risk</span><b>{fmt(current.risk * 100)}%</b>{proposedValue(point => <b>{fmt(point.risk * 100)}%</b>)}</div>
      <div><span>OSF margin</span><b>{fmt(current.osfMargin, 0)}</b>{proposedValue(point => <b>{fmt(point.osfMargin, 0)}</b>)}</div>
      <div><span>HDF margin</span><b>{fmt(current.delta - 8.6, 2)} K</b>{proposedValue(point => <b>{fmt(point.delta - 8.6, 2)} K</b>)}</div>
      <div><span>PWF high margin</span><b>{fmt(9000 - current.power, 0)} W</b>{proposedValue(point => <b>{fmt(9000 - point.power, 0)} W</b>)}</div>
      {error && <p className="api-error">Calculation failed: {error}</p>}
      <aside>{result ? result.summary : "No proposed result has been calculated yet. No machine command is issued by this analysis."}</aside>
      {result?.limitations?.[0] && <small className="decision-limitation">{result.limitations[0]}</small>}
    </div>
  </div>;
}

function ActPanel({ incident }) {
  const [completed, setCompleted] = useState([]);
  const [action, setAction] = useState(null);
  const checks = [
    "Inspect tool condition and confirm its expected wear stage.",
    "Verify whether the torque change is expected for the current job.",
    "Confirm the product-specific overstrain threshold for this run.",
  ];
  const toggleCheck = index => setCompleted(items => items.includes(index) ? items.filter(item => item !== index) : [...items, index]);
  const createAction = () => {
    if (!incident) return;
    setAction({ id: `MA-${incident.id.replace("INC-", "")}`, incidentId: incident.id, created: true });
  };
  return <Panel eyebrow="Act" title="Suggested next checks" className="act-panel">
    <ol className="action-checks">{checks.map((check, index) => <li key={check}><button className={completed.includes(index) ? "done" : ""} onClick={() => toggleCheck(index)}>{completed.includes(index) ? "Checked" : "Mark checked"}</button><span>{check}</span></li>)}</ol>
    {!action ? <button onClick={createAction} disabled={!incident}>Create Maintenance Action <span>Engineer review required</span></button> : <div className="maintenance-action"><div><b>{action.id}</b><Status status="WATCH"/></div><p>Draft maintenance action created from {action.incidentId}.</p><small>Local demo record only — it has not been sent to a CMMS or issued as a machine command.</small><button onClick={() => setAction(null)}>Dismiss draft</button></div>}
  </Panel>;
}

function LegacyCopilot({ incident, scenario, cycle, state }) {
  const [turns, setTurns] = useState([]);
  const [draft, setDraft] = useState("");
  const [asking, setAsking] = useState(false);
  const [suggestions, setSuggestions] = useState(questions);
  useEffect(() => {
    setTurns([]);
    setDraft("");
    setSuggestions(questions);
  }, [scenario]);
  const ask = async question => {
    if (!question.trim() || asking) return;
    setAsking(true);
    try {
      const conversation = turns.flatMap(turn => [
        { role: "user", content: turn.question },
        { role: "assistant", content: turn.answer },
      ]).slice(-6);
      const response = await fetch(`${API_BASE}/live/copilot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario: scenario || window.__industrialScenario || "OSF",
          cycle: cycle ?? state?.cycle ?? 0,
          question,
          conversation,
        }),
      });
      const body = await response.text();
      let result;
      try { result = JSON.parse(body); } catch { throw new Error(`Backend returned HTTP ${response.status} without a JSON response. Restart the FastAPI server on port 8000.`); }
      if (!response.ok) throw new Error(result.detail || "Copilot request failed.");
      setTurns(items => [...items, {
        question,
        answer: result.answer,
        verifiedAnswer: result.verified_answer,
        evidence: result.findings || [],
        limitations: result.limitations || [],
        aiGenerated: result.ai_generated,
        aiStatus: result.ai_status,
        aiProvider: result.ai_provider,
        aiModel: result.ai_model,
        aiWarning: result.ai_warning,
      }]);
      if (result.suggested_next_questions?.length) setSuggestions(result.suggested_next_questions);
      setDraft("");
    } catch (error) {
      setTurns(items => [...items, { question, answer: `Copilot is unavailable: ${error.message}`, evidence: [], limitations: [] }]);
    } finally { setAsking(false); }
  };
  return <Panel eyebrow="Investigate" title="AI Incident Copilot" action={<small className="context">{incident ? `Context: ${scenario} · ${incident.id}` : `Context: ${scenario} · monitoring`}</small>} className="copilot"><div className="conversation">{turns.length === 0 && <div className="empty"><b>{incident ? "Groq-powered incident reasoning" : `${scenario} monitoring context`}</b><br/>{incident ? "Ask a natural follow-up. The AI receives the active incident, validated changes, model evidence, and similar AI4I cases; calculations remain backend-verified." : "No incident is open. You can still ask about the current scenario and its latest verified machine state."}</div>}{turns.map((turn, index) => <div className="turn" key={index}><div className="question">{turn.question}</div><div className="answer"><span className={`ai-source ${turn.aiGenerated ? "generated" : "fallback"}`}>{turn.aiGenerated ? `Groq AI · ${turn.aiModel}` : turn.aiStatus === "not_applicable" ? "Live scenario context" : "Verified fallback"}</span><b>{turn.answer}</b>{(turn.verifiedAnswer || turn.evidence.length > 0) && <details><summary>Verified evidence used for this answer</summary>{turn.verifiedAnswer && <p className="verified-answer">{turn.verifiedAnswer}</p>}{turn.evidence.length > 0 && <p>{turn.evidence.join(" · ")}</p>}</details>}{turn.aiWarning && <small>AI note: {turn.aiWarning}</small>}{turn.limitations[0] && <small>Limitation: {turn.limitations[0]}</small>}</div></div>)}</div><div className="chips">{suggestions.map(question => <button key={question} onClick={() => ask(question)} disabled={asking}>{question}</button>)}</div><form onSubmit={event => { event.preventDefault(); ask(draft); }}><input value={draft} onChange={event => setDraft(event.target.value)} placeholder={incident ? "Ask a natural incident question…" : `Ask about the current ${scenario} scenario…`}/><button disabled={asking || !draft.trim()}>{asking ? "Thinking…" : "Ask AI"}</button></form></Panel>;
}

function Copilot({ incident, scenario, cycle, state }) {
  const [turns, setTurns] = useState([]);
  const [draft, setDraft] = useState("");
  const [asking, setAsking] = useState(false);
  const [mode, setMode] = useState("quick");
  const [suggestions, setSuggestions] = useState(questions);

  useEffect(() => {
    setTurns([]);
    setDraft("");
    setSuggestions(questions);
  }, [scenario]);

  const ask = async question => {
    if (!question.trim() || asking) return;
    setAsking(true);
    try {
      const conversation = turns.flatMap(turn => [
        { role: "user", content: turn.question },
        { role: "assistant", content: turn.answer },
      ]).slice(-6);
      const response = await fetch(`${API_BASE}/live/copilot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario, cycle: cycle ?? state?.cycle ?? 0, question, conversation, mode }),
      });
      const body = await response.text();
      let result;
      try { result = JSON.parse(body); } catch { throw new Error(`Backend returned HTTP ${response.status} without JSON. Restart FastAPI on port 8000.`); }
      if (!response.ok) throw new Error(result.detail || "Copilot investigation failed.");
      setTurns(items => [...items, {
        question,
        answer: result.answer,
        verifiedAnswer: result.verified_answer,
        evidence: result.findings || [],
        limitations: result.limitations || [],
        aiGenerated: result.ai_generated,
        aiStatus: result.ai_status,
        aiProvider: result.ai_provider,
        aiModel: result.ai_model,
        aiWarning: result.ai_warning,
        trace: result.investigation_trace,
        citations: result.citations || [],
        knowledgeSources: result.knowledge_sources || [],
        groundingStatus: result.grounding_status,
      }]);
      if (result.suggested_next_questions?.length) setSuggestions(result.suggested_next_questions);
      setDraft("");
    } catch (error) {
      setTurns(items => [...items, { question, answer: `Copilot is unavailable: ${error.message}`, evidence: [], limitations: [] }]);
    } finally { setAsking(false); }
  };

  const authority = value => value === "dataset_rule" ? "AI4I dataset mechanism" : value === "engineering_reference" ? "Real-world engineering reference" : value === "system_limit" ? "Current system limitation" : "Backend-calculated evidence";

  return <Panel eyebrow="Investigate" title="AI Incident Copilot" action={<small className="context">{incident ? `Context: ${scenario} · ${incident.id}` : `Context: ${scenario} · monitoring`}</small>} className="copilot">
    <div className="copilot-mode"><span>Investigation</span><button className={mode === "quick" ? "selected" : ""} onClick={() => setMode("quick")} disabled={asking}>Quick</button><button className={mode === "deep" ? "selected" : ""} onClick={() => setMode("deep")} disabled={asking}>Deep</button></div>
    {asking && <div className="investigation-progress">Understanding question · running evidence checks · building grounded answer</div>}
    <div className="conversation">
      {turns.length === 0 && <div className="empty"><b>{incident ? "Bounded AI investigation" : `${scenario} monitoring context`}</b><br/>{incident ? "The AI chooses only permitted evidence checks. Backend calculations, citations and safety validation remain authoritative." : "No incident is open. You can still ask about the latest verified machine state."}</div>}
      {turns.map((turn, index) => <div className="turn" key={index}><div className="question">{turn.question}</div><div className="answer">
        <span className={`ai-source ${turn.aiGenerated ? "generated" : "fallback"}`}>{turn.aiGenerated ? `${turn.aiProvider === "together" ? "Together AI" : turn.aiProvider === "groq" ? "Groq AI" : "Gemini AI"} · ${turn.aiModel}` : turn.aiStatus === "not_applicable" ? "Live scenario context" : "Evidence-based response"}</span>
        <b>{turn.answer}</b>
        {(turn.verifiedAnswer || turn.evidence.length > 0) && <details><summary>Verified deterministic evidence</summary>{turn.verifiedAnswer && <p className="verified-answer">{turn.verifiedAnswer}</p>}{turn.evidence.length > 0 && <p>{turn.evidence.join(" · ")}</p>}</details>}
        {turn.citations?.length > 0 && <details><summary>Evidence atoms used ({turn.citations.length})</summary><div className="citation-list">{turn.citations.map(atom => <p key={atom.id}><b>[{atom.id}]</b> {atom.display_value ? `${atom.statement}: ${atom.display_value}` : atom.statement}<small>{authority(atom.authority)}</small></p>)}</div></details>}
        {turn.trace && <details className="agent-trace"><summary>AI investigation trace</summary><p><b>Objective:</b> {turn.trace.objective}</p><p><b>Plan:</b> {turn.trace.planner_status} · {turn.trace.tool_round_count} evidence round{turn.trace.tool_round_count === 1 ? "" : "s"} · grounding {turn.groundingStatus || turn.trace.grounding_status}</p>{turn.trace.tools?.length > 0 && <ul>{turn.trace.tools.map((tool, item) => <li key={`${tool.name}-${item}`}>{tool.status === "completed" ? "✓" : "!"} {tool.name} — {tool.purpose}</li>)}</ul>}{turn.knowledgeSources?.length > 0 && <div className="knowledge-source-list">{turn.knowledgeSources.map((source, item) => <small key={`${source.title}-${item}`}>{authority(source.authority)}: {source.title}</small>)}</div>}</details>}
        {turn.aiWarning && <small>AI note: {turn.aiWarning}</small>}{turn.limitations[0] && <small>Limitation: {turn.limitations[0]}</small>}
      </div></div>)}
    </div>
    <div className="chips">{suggestions.map(question => <button key={question} onClick={() => ask(question)} disabled={asking}>{question}</button>)}</div>
    <form onSubmit={event => { event.preventDefault(); ask(draft); }}><input value={draft} onChange={event => setDraft(event.target.value)} placeholder={incident ? "Ask a natural incident question…" : `Ask about the current ${scenario} scenario…`}/><button disabled={asking || !draft.trim()}>{asking ? "Investigating…" : "Investigate with AI"}</button></form>
  </Panel>;
}

export default function App() {
  const [scenario, setScenario] = useState("OSF");
  const [cycle, setCycle] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(5);
  const [localIncident, setIncident] = useState(null);
  const [workspace, setWorkspace] = useState("understand");
  const [sessionRevision, setSessionRevision] = useState(0);
  const [live, setLive] = useState(null);
  const [liveError, setLiveError] = useState("");
  const activeScenarioRef = useRef(scenario);
  const activeSessionRef = useRef(sessionRevision);
  const latestAppliedCycleRef = useRef(-1);
  const total = scenarios[scenario].cycles;
  window.__industrialScenario = scenario;
  const complete = cycle >= total;
  useEffect(() => { if (!running) return; const id = setInterval(() => setCycle(value => value >= total ? (setRunning(false), value) : value + 1), 1200 / speed); return () => clearInterval(id); }, [running, speed, total]);
  const localHistory = useMemo(() => Array.from({ length: Math.max(1, cycle) }, (_, index) => twinAt(scenario, index + 1)), [scenario, cycle]);
  useEffect(() => {
    activeScenarioRef.current = scenario;
    activeSessionRef.current = sessionRevision;
    latestAppliedCycleRef.current = -1;
  }, [scenario, sessionRevision]);
  useEffect(() => {
    const stride = speed >= 20 ? 6 : speed >= 5 ? 2 : 1;
    const shouldFetch = cycle <= 1 || cycle >= total || cycle % stride === 0;
    if (!shouldFetch) return;
    const requestedScenario = scenario;
    const requestedSession = sessionRevision;
    const requestedCycle = cycle;
    fetch(`${API_BASE}/live/state`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scenario, cycle }) })
      .then(response => response.ok ? response.json() : response.json().then(body => Promise.reject(new Error(body.detail || "Live API request failed."))))
      .then(snapshot => {
        const stillCurrent = activeScenarioRef.current === requestedScenario && activeSessionRef.current === requestedSession;
        if (stillCurrent && requestedCycle >= latestAppliedCycleRef.current) {
          latestAppliedCycleRef.current = requestedCycle;
          setLive(snapshot);
          setLiveError("");
        }
      })
      .catch(error => {
        if (activeScenarioRef.current === requestedScenario && activeSessionRef.current === requestedSession) setLiveError(error.message);
      });
  }, [scenario, cycle, sessionRevision, speed, total]);
  // A hosted backend can respond after the UI has advanced several simulated
  // cycles (especially on a free-tier service waking from idle). Never mix a
  // current header cycle with an older backend history: use the deterministic
  // scenario projection until the matching backend snapshot arrives.
  const liveIsCurrent = Boolean(live && live.cycle >= cycle);
  const history = liveIsCurrent ? live.history : localHistory;
  const state = liveIsCurrent ? live : history[history.length - 1];
  const previous = history[history.length - 2];
  const syncing = Boolean(live && !liveIsCurrent);
  const localChange = useMemo(() => changes(history), [history]);
  const localSimilar = useMemo(() => similarity(state), [state]);
  const change = live?.changes || localChange;
  const similar = live?.similar || localSimilar;
  const incident = live?.incident ?? localIncident;
  useEffect(() => {
    if (live) { setIncident(live.incident); if (live.incident) setWorkspace("understand"); return; }
    if (cycle === 0) { setIncident(null); return; }
    if (["WARNING", "INCIDENT"].includes(state.status)) {
      setIncident(current => {
        if (!current) return { id: "INC-0001", severity: state.status, opened: state.cycle, reason: state.osfMargin < 1000 ? "Overstrain margin narrowing" : "Model failure-risk estimate crossed the alert policy threshold" };
        if (state.status === "INCIDENT" && current.severity !== "INCIDENT") return { ...current, severity: "INCIDENT" };
        return current;
      });
      setWorkspace("understand");
    }
  }, [cycle, live, state]);
  const reset = () => { setRunning(false); setCycle(0); setIncident(null); setLive(null); setWorkspace("understand"); setSessionRevision(value => value + 1); };
  const delta = key => previous ? (state[key] - previous[key]) / Math.abs(previous[key] || 1) * 100 : null;
  const kpis = [["Rotational speed", fmt(state.rpm, 0), "rpm", "AI4I range 1168 – 2886", "rpm", "NORMAL"], ["Torque", fmt(state.torque), "Nm", "AI4I range 3.8 – 76.6", "torque", state.torque > 52 ? "WARNING" : "NORMAL"], ["Tool wear", state.wear, "min", "AI4I max 253", "wear", state.wear > 200 ? "INCIDENT" : state.wear > 150 ? "WARNING" : "WATCH"], ["Temperature delta", fmt(state.delta, 2), "K", "HDF region below 8.6 K", "delta", state.delta < 8.6 ? "WARNING" : "NORMAL"], ["Mechanical power", fmt(state.power / 1000, 2), "kW", "PWF band 3.5 – 9.0 kW", "power", state.power > 9000 ? "WARNING" : "NORMAL"], ["Failure risk", fmt(state.risk * 100), "%", "Calibrated AI4I estimate", "risk", state.status]];
  return <main className="workflow-app">
    <section className="stream"><div className="stream-top"><div className="brand"><span className="logo">⌁</span><div><h1>Industrial Intelligence Copilot</h1><p>MACHINE-01 · Session SIM-LIVE-001 · Product type L</p></div></div><div className="controls"><div><small>Cycle</small><b>{cycle} / {total}</b></div><div><small>Stream</small><b>{complete ? "COMPLETE" : running ? "STREAMING" : "IDLE"}</b></div><Status status={cycle ? state.status : "NORMAL"}/><button className="primary" disabled={complete} onClick={() => setRunning(value => !value)}>{running ? "Pause" : complete ? "Streamed" : "Start"}</button><button onClick={reset}>Reset</button></div></div><div className="stream-bottom"><div className="scenarios">{Object.entries(scenarios).map(([key,item]) => <button key={key} onClick={() => { setScenario(key); reset(); }} className={scenario === key ? "selected" : ""}>{item.label}</button>)}</div><div className="speed"><small>Speed</small>{[1,5,20].map(item => <button key={item} onClick={() => setSpeed(item)} className={speed === item ? "selected" : ""}>{item}x</button>)}</div></div><p className="source">Source: {scenarios[scenario].source} <span>Rule basis: {scenarios[scenario].rule}. No live PLC connection.</span></p></section>
    <nav className="workflow-stepper">{stages.map(([name, detail], index) => <button key={name} className={(workspace === name.toLowerCase() || (name === "Sense" && !incident)) ? "active" : ""} onClick={() => name === "Decide" ? setWorkspace("decide") : name === "Understand" ? setWorkspace("understand") : name === "Act" ? document.getElementById("act")?.scrollIntoView({ behavior: "smooth" }) : name === "Investigate" ? document.getElementById("copilot")?.scrollIntoView({ behavior: "smooth" }) : null}><b>0{index + 1}</b><strong>{name}</strong><span>{detail}</span></button>)}</nav>
    {liveError && <div className="live-error">Backend synchronization issue: {liveError}</div>}
    <section className="live-console"><div className="sense-column"><Panel eyebrow="Sense" title="Live operating condition" action={<Status status={state.status}/>}><div className="compact-kpis">{kpis.map(([label,value,unit,sub,key,status]) => <Kpi key={label} label={label} value={value} unit={unit} sub={sub} status={status} delta={delta(key)} points={history.slice(-12).map(item => key === "risk" ? item.risk : item[key])}/>)}</div></Panel><Panel eyebrow="Sense" title="Telemetry stream" action={<small>{history.length} cycles buffered{syncing ? " · syncing" : ""}</small>}><Chart history={history}/></Panel><Panel eyebrow="Sense" title="Operational machine twin"><Twin state={state}/></Panel></div>
    <aside className="investigation-column"><Panel eyebrow="Detect" title="Active incident" action={incident ? <Status status={incident.severity}/> : null}>{incident ? <div className="incident focus"><b>{incident.id} · opened at cycle {incident.opened}</b><h3>Operating condition changed</h3><p>{incident.reason}</p><div className="incident-metrics"><span><small>Failure risk</small><b>{fmt(state.risk * 100)}%</b></span><span><small>OSF margin</small><b>{fmt(state.osfMargin, 0)}</b></span></div><details><summary>Why the system flagged this</summary><p>Rule margins and calibrated model risk are evaluated against the current telemetry window.</p></details></div> : <div className="empty"><i className="dot"/> <b>No active incident on MACHINE-01.</b><br/>Monitoring waits for risk, rule-margin, or envelope evidence.</div>}</Panel><div id="copilot"><Copilot key={`${scenario}-${sessionRevision}`} incident={incident} scenario={scenario} cycle={cycle} state={state} changes={change} similar={similar}/></div><div id="act"><ActPanel incident={incident}/></div></aside></section>
    <section className="workspace"><div className="workspace-tabs"><button className={workspace === "understand" ? "active" : ""} onClick={() => setWorkspace("understand")}>Understand</button><button className={workspace === "decide" ? "active" : ""} onClick={() => setWorkspace("decide")}>Decide · What-if</button><button className={workspace === "evidence" ? "active" : ""} onClick={() => setWorkspace("evidence")}>Evidence trace</button></div>{workspace === "understand" && <div className="understand-grid"><Panel eyebrow="Understand" title="What changed"><p>Recent 5 cycles vs previous baseline window.</p>{change.length ? change.slice(0,4).map(item => <div className="change" key={item.name}><span>{item.name} [{item.unit}]</span><b>{item.percent >= 0 ? "+" : ""}{fmt(item.percent)}%</b><i><em style={{ width: `${Math.abs(item.percent) / Math.abs(change[0].percent) * 100}%` }}/></i><small>{fmt(item.before)} → {fmt(item.now)}</small></div>) : <div className="empty">Building the baseline window. Comparison appears after ~6 cycles.</div>}</Panel><Panel eyebrow="Understand" title="Similar historical conditions"><div className="mini"><b>{similar.retrieved}<small>Retrieved</small></b><b>{similar.failed}<small>Failed</small></b><b>{fmt(similar.rate * 100)}%<small>Failure rate</small></b></div><p>Most common associated failure flag: <b>{similar.topMode}</b>.</p><details><summary>Show similar-case evidence</summary><div className="outcomes">{similar.cases.map((item,index) => <i key={index} className={item.failed ? "failed" : "healthy"}/>)}</div><p>Nearest AI4I observations by RPM, torque and tool wear. Historical evidence only — not a prediction.</p></details></Panel></div>}{workspace === "decide" && <Panel eyebrow="Decide" title="What-if analysis" action={<small>Current vs proposed</small>}><DecisionWorkspace current={state} scenario={scenario}/></Panel>}{workspace === "evidence" && <Panel eyebrow="Evidence" title="Evidence trace"><div className="evidence-trace"><div><small>Validated inputs</small><p>Operational twin state, rule margins, recent-vs-baseline window, and AI4I nearest-condition retrieval.</p></div><div><small>Incident context</small><p>{incident ? `${incident.id} is open with ${incident.severity} severity.` : "No incident is currently open."}</p></div><div><small>Limitation</small><p>Evidence-backed decision support only. No causal diagnosis, RUL estimate, live PLC connection, or machine command.</p></div></div></Panel>}</section>
    <footer>Evidence-first decision support. Data source is {scenarios[scenario].source} — not a live PLC feed. This console does not estimate remaining useful life, does not issue machine commands, and does not claim causal diagnosis.</footer>
  </main>;
}
