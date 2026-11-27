"use client";

import { useState } from "react";
import Plot from "./Plot";
import { Btn, ErrorBox, fmt, KV, Learn, Section, showTok, TopKTable } from "./ui";
import { call, Tok, TopK } from "@/lib/api";

type JTok = { id: number; text: string; cosine: number; rank: number };
type JReadout = { method: string; plus: { top: JTok[]; targets: JTok[] }; minus: { top: JTok[]; targets: JTok[] }; lens: Record<string, unknown> };
type JShareResp = { j_share: number; k_used: number; all_coefficients_nonnegative: boolean; residual_norm_sq: number;
  directions: { id: number; text: string; coefficient: number; cosine_with_w: number }[];
  vectors: { w_unit: number[]; verbalizable: number[]; unexplained: number[] };
  random_baseline?: { j_shares: number[]; percentile_vs_random: number; random_p1: number; random_p99: number; random_median: number; n_random: number; class: string; label: string } };
type ResidJ = { position: number; tokens: Tok[]; model_prediction: TopK[]; layers: { state: string; top_k: TopK[]; targets: { id: number; text: string; rank: number; prob: number }[] }[] };
type ValResp = { dz_deps_context: number; dz_deps_corpus_jlens: number; slope_ratio_lens_over_context: number | null; residual_norm_at_injection: number;
  rows: { eps_rel: number; observed: number; context_jvp: number; corpus_jlens: number }[];
  relative_error: Record<string, number | null>; notes: Record<string, string> };

const CLASS_COLOR: Record<string, string> = { high: "var(--ok)", middle: "var(--warn)", low: "var(--bad)" };

export function JNeuronPanel({ backend, layer, neuron, targets }: { backend: string; layer: number; neuron: number; targets: string[] }) {
  const [ro, setRo] = useState<JReadout | null>(null);
  const [js, setJs] = useState<JShareResp | null>(null);
  const [sign, setSign] = useState<"+" | "-">("+");
  const [nRandom, setNRandom] = useState(256);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function readout() {
    setBusy("readout"); setErr(null);
    try { setRo(await call<JReadout>(backend, "/lens/jacobian", { kind: "neuron", layer, neuron, top_k: 25, target_tokens: targets })); }
    catch (e) { setErr((e as Error).message); }
    setBusy(null);
  }
  async function share() {
    setBusy("jshare"); setErr(null);
    try { setJs(await call<JShareResp>(backend, "/lens/j-share", { layer, neuron, sign, k: 25, n_random: nRandom })); }
    catch (e) { setErr((e as Error).message); }
    setBusy(null);
  }
  const col = (title: string, s: { top: JTok[]; targets: JTok[] }, c: string) => (
    <Section title={title}>
      <table className="w-full text-[12px] num"><tbody>{s.top.map((t, i) => <tr key={t.id} className="border-t border-line"><td>{i + 1}</td>
        <td className="font-mono" style={{ color: c }}>{showTok(t.text)}</td><td className="text-right">{fmt(t.cosine, 4)}</td></tr>)}</tbody></table>
      {s.targets.map((t) => <div key={t.id} className="num text-[12px] mt-1">target {showTok(t.text)}: cos {fmt(t.cosine, 4)} · rank {t.rank.toLocaleString()}</div>)}
    </Section>);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-2 items-center flex-wrap">
        <Btn kind="primary" onClick={readout} disabled={!!busy}>{busy === "readout" ? "reading…" : `J-lens readout of L${layer} n${neuron}`}</Btn>
        <span className="text-[12px] text-muted">J-share of</span>
        <select value={sign} onChange={(e) => setSign(e.target.value as "+" | "-")}><option>+</option><option>-</option></select>
        <span className="text-[12px] text-muted">w, random baseline</span>
        <select value={nRandom} onChange={(e) => setNRandom(+e.target.value)}>
          <option value={256}>256 (quick)</option><option value={2048}>2,048 (paper)</option><option value={0}>none</option></select>
        <Btn onClick={share} disabled={!!busy}>{busy === "jshare" ? (nRandom === 2048 ? "computing (≈2 min first time)…" : "computing…") : "compute J-share"}</Btn>
      </div>
      <ErrorBox error={err} />
      {ro && <div className="grid grid-cols-2 gap-2">{col("+w — J-lens readout (cosine)", ro.plus, "var(--pos)")}{col("−w — J-lens readout", ro.minus, "var(--neg)")}</div>}
      {ro && <Learn>{ro.method}. Lens: {String(ro.lens.n_prompts_fitted)} prompts of {String(ro.lens.dataset)} (our fit, reference jlens code).</Learn>}
      {js && <Section title={`J-share of ${sign}w`}>
        <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-3">
          <div>
            <div className="text-3xl font-semibold num">{fmt(js.j_share, 4)}</div>
            {js.random_baseline && <div className="text-[13px] font-semibold" style={{ color: CLASS_COLOR[js.random_baseline.class] }}>
              {js.random_baseline.class} · {fmt(js.random_baseline.percentile_vs_random, 1)}th percentile of random</div>}
            <KV k="directions used (≤25)" v={js.k_used} />
            <KV k="‖w − v‖² (unexplained)" v={fmt(js.residual_norm_sq, 4)} />
            <KV k="coefficients all ≥ 0" v={String(js.all_coefficients_nonnegative)} />
            {js.random_baseline && <><KV k="random p1 / median / p99" v={`${fmt(js.random_baseline.random_p1, 4)} / ${fmt(js.random_baseline.random_median, 4)} / ${fmt(js.random_baseline.random_p99, 4)}`} />
              <KV k="baseline" v={js.random_baseline.label} /></>}
            <div className="text-[11px] text-muted mt-2">Selected directions (sorted by coefficient)</div>
            <table className="w-full text-[11px] num"><thead className="text-muted"><tr><th className="text-left">token</th><th className="text-right">coef</th><th className="text-right">cos(d, w)</th></tr></thead>
              <tbody>{js.directions.map((d) => <tr key={d.id} className="border-t border-line"><td className="font-mono">{showTok(d.text)}</td>
                <td className="text-right">{fmt(d.coefficient, 4)}</td><td className="text-right">{fmt(d.cosine_with_w, 4)}</td></tr>)}</tbody></table>
          </div>
          <div>
            {js.random_baseline && <Plot height={220} data={[{ type: "histogram", x: js.random_baseline.j_shares, nbinsx: 40, marker: { color: "#94a3b8" }, name: "random" }]}
              layout={{ shapes: [{ type: "line", x0: js.j_share, x1: js.j_share, yref: "paper", y0: 0, y1: 1, line: { color: "#ef4444", width: 2 } },
                ...[js.random_baseline.random_p1, js.random_baseline.random_p99].map((x) => ({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: "#64748b", dash: "dot", width: 1 } }))],
                xaxis: { title: { text: "J-share" } }, margin: { l: 36, r: 8, t: 8, b: 36 }, showlegend: false }} />}
            <div className="text-[11px] text-muted">red = this neuron; dotted = 1st/99th percentile of random unit directions at this layer</div>
            <Plot height={200} data={[
              { type: "scattergl", mode: "lines", y: js.vectors.w_unit, name: "w (unit)", line: { width: 1, color: "#64748b" } },
              { type: "scattergl", mode: "lines", y: js.vectors.verbalizable, name: "verbalizable v", line: { width: 1, color: "#22c55e" } },
              { type: "scattergl", mode: "lines", y: js.vectors.unexplained, name: "unexplained w − v", line: { width: 1, color: "#ef4444" } }]}
              layout={{ margin: { l: 36, r: 8, t: 8, b: 30 }, legend: { orientation: "h", y: -0.25 } }} />
          </div>
        </div>
        <Learn>v is the best nonnegative combination of at most 25 unit J-lens token directions (nonnegative matching pursuit with NNLS refits).
          J-share = 1 − ‖w − v‖² for unit w. The paper calls a neuron &quot;high&quot; above the 99th percentile of 2,048 random directions, &quot;low&quot; below the 1st.</Learn>
      </Section>}
    </div>
  );
}

export function JResidualPanel({ backend, prompt, position, targets }: { backend: string; prompt: string; position: number; targets: string[] }) {
  const [r, setR] = useState<ResidJ | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  async function run() {
    setBusy(true); setErr(null);
    try { setR(await call<ResidJ>(backend, "/lens/jacobian", { kind: "residual", prompt, position, top_k: 5, target_tokens: targets })); }
    catch (e) { setErr((e as Error).message); }
    setBusy(false);
  }
  return (
    <div className="flex flex-col gap-2">
      <div><Btn kind="primary" onClick={run} disabled={busy}>{busy ? "running…" : "J-lens across depth"}</Btn></div>
      <ErrorBox error={err} />
      {r && <Section title={`J-lens of the residual at position ${r.position} (${JSON.stringify(r.tokens[r.position]?.text)})`}>
        <table className="w-full text-[11px] num"><thead className="text-muted"><tr><th className="text-left">layer</th><th className="text-left" colSpan={5}>top-5 (prob)</th>
          {r.layers[0]?.targets.map((t) => <th key={t.id} className="text-right">{showTok(t.text)} rank</th>)}</tr></thead>
          <tbody>{r.layers.map((l) => <tr key={l.state} className="border-t border-line"><td>{l.state}</td>
            {l.top_k.map((t) => <td key={t.id} className="font-mono">{showTok(t.text)} <span className="text-muted">{fmt(t.prob, 2)}</span></td>)}
            {l.targets.map((t) => <td key={t.id} className="text-right">{t.rank.toLocaleString()}</td>)}</tr>)}</tbody></table>
        <Learn>W_U · RMSNorm_final(J_l h_l): the remaining layers are replaced by their corpus-averaged linear map. Compare with the logit lens, which simply skips them.</Learn>
        <div className="mt-2"><div className="text-[11px] text-muted">actual model prediction</div><TopKTable rows={r.model_prediction} /></div>
      </Section>}
    </div>
  );
}

export function JValidatePanel({ backend, prompt, layer, nLayers }: { backend: string; prompt: string; layer: number; nLayers: number }) {
  const [target, setTarget] = useState(" John");
  const [L, setL] = useState(layer);
  const [v, setV] = useState<ValResp | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  async function run() {
    setBusy(true); setErr(null);
    try { setV(await call<ValResp>(backend, "/lens/validate", { prompt, layer: L, target, direction: "token_jlens" })); }
    catch (e) { setErr((e as Error).message); }
    setBusy(false);
  }
  return (
    <Section title="Validation: linear prediction vs. real intervention">
      <div className="flex gap-2 items-center text-[12px]">inject the J-lens direction of
        <input className="w-24 font-mono" value={target} onChange={(e) => setTarget(e.target.value)} /> at layer
        <input type="number" className="w-14 num" min={0} max={nLayers - 1} value={L} onChange={(e) => setL(+e.target.value)} />
        <Btn kind="primary" onClick={run} disabled={busy}>{busy ? "running ~20 passes…" : "run"}</Btn></div>
      <ErrorBox error={err} />
      {v && <>
        <Plot height={280} data={[
          { type: "scatter", mode: "lines+markers", name: "observed (real forward passes)", x: v.rows.map((r) => r.eps_rel), y: v.rows.map((r) => r.observed), line: { color: "#ef4444" } },
          { type: "scatter", mode: "lines", name: "context Jacobian (this prompt, exact 1st order)", x: v.rows.map((r) => r.eps_rel), y: v.rows.map((r) => r.context_jvp), line: { color: "#22c55e", dash: "dash" } },
          { type: "scatter", mode: "lines", name: "corpus J-lens (averaged)", x: v.rows.map((r) => r.eps_rel), y: v.rows.map((r) => r.corpus_jlens), line: { color: "#3b82f6", dash: "dot" } }]}
          layout={{ xaxis: { title: { text: "ε / ‖h_l‖" } }, yaxis: { title: { text: "Δ logit(target)" } }, margin: { l: 50, r: 8, t: 8, b: 40 }, legend: { orientation: "h", y: -0.3 } }} />
        <div className="grid grid-cols-2 gap-x-6 text-[12px]">
          <KV k="slope: context / lens" v={`${fmt(v.dz_deps_context, 4)} / ${fmt(v.dz_deps_corpus_jlens, 4)}`} />
          <KV k="lens ÷ context slope" v={fmt(v.slope_ratio_lens_over_context ?? NaN, 2)} />
          <KV k="rel. error, small ε: context" v={fmt(v.relative_error.context_jvp_small_eps ?? NaN, 3)} />
          <KV k="rel. error, small ε: lens" v={fmt(v.relative_error.corpus_jlens_small_eps ?? NaN, 3)} />
        </div>
        <Learn>The context Jacobian should track the observed curve near ε = 0 and diverge as ε grows (nonlinearity). The corpus J-lens is an
          average over contexts that sums effects over all later positions, so its magnitude is not a same-position prediction. Its sign and
          direction are what the readouts use.</Learn>
      </>}
    </Section>
  );
}

type Census = { layers: { layer: number; frac_high: number; frac_middle: number; frac_low: number; median_used: number; median_random: number; ratio_median: number }[];
  overall: { high: number; middle: number; low: number; n: number }; paper_reference: Record<string, string>; n_random: number; k: number; dictionary: string };

export function JCensusPanel({ backend }: { backend: string }) {
  const [c, setC] = useState<Census | null>(null);
  const [err, setErr] = useState<string | null>(null);
  async function load() { setErr(null); try { setC(await call<Census>(backend, "/lens/j-share/census")); } catch (e) { setErr((e as Error).message); } }
  return (
    <div className="flex flex-col gap-2">
      <div><Btn kind="primary" onClick={load}>load J-share census</Btn></div>
      <ErrorBox error={err} />
      {c && <>
        <Section title={`J-share census — ${c.overall.n.toLocaleString()} neurons (sign used), ${c.n_random.toLocaleString()} random directions per layer`}>
          <Plot height={300} data={[
            { type: "bar", name: "high (> random p99)", x: c.layers.map((l) => l.layer), y: c.layers.map((l) => l.frac_high), marker: { color: "#16a34a" } },
            { type: "bar", name: "middle", x: c.layers.map((l) => l.layer), y: c.layers.map((l) => l.frac_middle), marker: { color: "#cbd5e1" } },
            { type: "bar", name: "low (< random p1)", x: c.layers.map((l) => l.layer), y: c.layers.map((l) => l.frac_low), marker: { color: "#dc2626" } }]}
            layout={{ barmode: "stack", xaxis: { title: { text: "layer" } }, yaxis: { title: { text: "share of neurons" }, range: [0, 1] }, margin: { l: 50, r: 8, t: 8, b: 40 }, legend: { orientation: "h", y: -0.25 } }} />
          <Plot height={220} data={[{ type: "scatter", mode: "lines+markers", name: "median neuron / median random", x: c.layers.map((l) => l.layer), y: c.layers.map((l) => l.ratio_median) }]}
            layout={{ shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: { dash: "dot", color: "#64748b" } }],
              xaxis: { title: { text: "layer" } }, yaxis: { title: { text: "J-share ratio" } }, margin: { l: 50, r: 8, t: 8, b: 40 } }} />
          <div className="grid grid-cols-2 gap-x-6 text-[12px]">
            <KV k="overall high / middle / low" v={`${fmt(100 * c.overall.high, 1)}% / ${fmt(100 * c.overall.middle, 1)}% / ${fmt(100 * c.overall.low, 1)}%`} />
            <KV k="paper (4 larger models)" v={`${c.paper_reference.high} / ${c.paper_reference.middle} / ${c.paper_reference.low}`} />
            <KV k="median ratio range" v={`${fmt(Math.min(...c.layers.map((l) => l.ratio_median)), 2)}–${fmt(Math.max(...c.layers.map((l) => l.ratio_median)), 2)}`} />
            <KV k="paper median ratio" v={c.paper_reference.median_ratio} />
          </div>
          <Learn>Our version of the paper&apos;s Figure 2a/b on Qwen3-0.6B, with our own J-lens. Late layers resemble the paper (many high, few low);
            early and middle layers do not (median neuron below random). Dictionary: {c.dictionary}; up to {c.k} directions per vector.</Learn>
        </Section>
      </>}
    </div>
  );
}
