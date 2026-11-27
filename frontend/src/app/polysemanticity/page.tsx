"use client";

import { useState } from "react";
import { ModelPanel } from "@/components/ModelPanel";
import Plot from "@/components/Plot";
import { Workspace } from "@/components/Shell";
import { Btn, ErrorBox, Expand, fmt, KV, Learn, Section, showTok, TokenStrip, TopKTable } from "@/components/ui";
import { useNeuronHover } from "@/components/NeuronHover";
import { ScanView, Type2View, Type3View } from "@/components/PolyViews";
import { call, Tok, TopK } from "@/lib/api";
import { useLab } from "@/lib/state";

type Ctx = { label: string; prompt: string; target: string };
type Stats = { logit: number; logprob: number; prob: number; rank: number };
type CtxResult = { label: string; prompt: string; tokens: Tok[]; prediction_position: number; target: { id: number; text: string };
  activation: number[]; gate: number[]; sign_used_with_open_gate: boolean[]; activation_at_prediction: number; gate_at_prediction: number;
  direct_logit_attribution_at_prediction: number; baseline: Stats; ablated: Stats;
  delta: { logprob: number; logit: number; prob: number; rank: number }; baseline_top_k: TopK[]; ablated_top_k: TopK[];
  controls: { kind: string; n: number; delta_logprob: number[]; summary: { mean: number; std: number; ci95_mean: number[] | null }; z: number | null; percentile_of_neuron: number | null };
  status: string; reason: string };
type Type1Resp = { layer: number; neuron: number; sign: string; overall: string; contexts: CtxResult[]; criteria: { supported: string; note: string };
  readout_logit_lens: { plus: { top_by_score: { text: string; score: number }[] }; minus: { top_by_score: { text: string; score: number }[] } };
  runtime_seconds: number; provenance: Record<string, unknown> };
type Cand = { layer: number; neuron: number; min_dla: number; dla_per_context: number[]; activation_per_context: number[]; signs: string[] };

const STATUS_COLOR: Record<string, string> = { supported: "var(--ok)", contradicted: "var(--bad)", inconclusive: "var(--warn)", insufficient_data: "var(--muted)" };

const DEFAULT: Ctx[] = [
  { label: "Gospel", prompt: "The fourth Gospel was written by the apostle", target: " John" },
  { label: "Elton", prompt: "The British singer and pianist Elton", target: " John" },
  { label: "Beatles", prompt: "The Beatles were Paul McCartney, George Harrison, Ringo Starr and", target: " John" },
];

function Type1View() {
  const { backend, arch, layer, setLayer, neuron, setNeuron } = useLab();
  const [ctx, setCtx] = useState<Ctx[]>(DEFAULT);
  const [sign, setSign] = useState<"+" | "-">("-");
  const [nCtrl, setNCtrl] = useState(20);
  const [seed, setSeed] = useState(0);
  const [cands, setCands] = useState<Cand[] | null>(null);
  const [res, setRes] = useState<Type1Resp | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const hov = useNeuronHover();
  const upd = (i: number, k: keyof Ctx, v: string) => setCtx(ctx.map((c, j) => (j === i ? { ...c, [k]: v } : c)));

  async function findCands() {
    setBusy("candidates"); setErr(null);
    try { setCands((await call<{ candidates: Cand[] }>(backend, "/experiments/polysemanticity/candidates", { contexts: ctx, top: 20 })).candidates); }
    catch (e) { setErr((e as Error).message); }
    setBusy(null);
  }
  async function runExp() {
    setBusy("experiment"); setErr(null);
    try { setRes(await call<Type1Resp>(backend, "/experiments/polysemanticity", { layer, neuron, sign, contexts: ctx, n_controls: nCtrl, seed })); }
    catch (e) { setErr((e as Error).message); }
    setBusy(null);
  }

  return (
    <Workspace
      left={<>
        <ModelPanel />
        <Section title="Type 1 — contexts (same target, unrelated topics)">
          {ctx.map((c, i) => (
            <div key={i} className="border-b border-line pb-1 mb-1">
              <div className="flex gap-1"><input className="w-24" value={c.label} onChange={(e) => upd(i, "label", e.target.value)} />
                <input className="flex-1 font-mono" title="target (single token, leading space matters)" value={c.target} onChange={(e) => upd(i, "target", e.target.value)} />
                <button className="text-bad" onClick={() => setCtx(ctx.filter((_, j) => j !== i))}>×</button></div>
              <textarea className="w-full h-12 font-mono text-[11px] mt-1" value={c.prompt} onChange={(e) => upd(i, "prompt", e.target.value)} />
            </div>))}
          <Btn onClick={() => setCtx([...ctx, { label: `ctx ${ctx.length + 1}`, prompt: "", target: ctx[0]?.target ?? " John" }])}>+ context</Btn>
          <Learn>The target is the next token after each prompt. Check in the Model Explorer that the model actually predicts it.
            Whether the contexts are unrelated is your judgment; NeuronLab does not test it.</Learn>
        </Section>
        <Section title="1 · Find candidates (exploratory)">
          <Btn onClick={findCands} disabled={!arch || !!busy}>{busy === "candidates" ? "scanning…" : "scan all neurons"}</Btn>
          {cands && <table className="w-full text-[11px] num mt-1"><tbody>{cands.map((c) => (
            <tr key={`${c.layer}-${c.neuron}`} className="border-t border-line cursor-pointer hover:bg-panel-2" {...hov(c.layer, c.neuron)} onClick={() => { setLayer(c.layer); setNeuron(c.neuron); setSign(c.signs[0] as "+" | "-"); }}>
              <td className="text-accent">L{c.layer} n{c.neuron} ({c.signs.join("")})</td><td className="text-right">min DLA {fmt(c.min_dla, 3)}</td></tr>))}</tbody></table>}
          <Learn>Ranks every neuron by the smallest direct logit attribution a·(u_target·w)/rms over the contexts. This only finds candidates; it is not evidence.</Learn>
        </Section>
        {arch && <Section title="2 · Test a signed neuron">
          <div className="flex gap-2 items-center">
            <label className="text-muted">L</label><input type="number" className="w-14 num" value={layer} min={0} max={arch.n_layers - 1} onChange={(e) => setLayer(+e.target.value)} />
            <label className="text-muted">n</label><input type="number" className="w-20 num" value={neuron} min={0} max={arch.d_mlp - 1} onChange={(e) => setNeuron(+e.target.value)} />
            <select value={sign} onChange={(e) => setSign(e.target.value as "+" | "-")}><option>+</option><option>-</option></select>
          </div>
          <div className="flex gap-2 items-center mt-1 text-[12px]">controls <input type="number" className="w-14 num" value={nCtrl} min={2} max={500} onChange={(e) => setNCtrl(+e.target.value)} />
            seed <input type="number" className="w-14 num" value={seed} onChange={(e) => setSeed(+e.target.value)} /></div>
          <div className="mt-2"><Btn kind="primary" onClick={runExp} disabled={!!busy}>{busy === "experiment" ? "running…" : "run experiment"}</Btn></div>
        </Section>}
        <ErrorBox error={err} />
      </>}
      center={<>
        {!res && <div className="text-muted text-[12px] p-2">Scan for candidates or enter a neuron, then run the experiment. Each context runs one baseline pass, one ablation
          pass and N control passes, all real forward passes.</div>}
        {res && <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(res.contexts.length, 3)}, minmax(0, 1fr))` }}>
          {res.contexts.map((c, i) => (
            <Section key={i} title={c.label} right={<span className="text-[11px] font-semibold" style={{ color: STATUS_COLOR[c.status] }}>{c.status}</span>}>
              <TokenStrip tokens={c.tokens} selected={c.prediction_position} onSelect={() => {}} values={c.activation}
                gateOpen={c.gate.map((g) => g > 0)} />
              <div className="mt-1">
                <KV k="target" v={<span className="font-mono">{showTok(c.target.text)}</span>} />
                <KV k="a at prediction / gate" v={`${fmt(c.activation_at_prediction, 3)} / ${fmt(c.gate_at_prediction, 3)}`} />
                <KV k="sign used with open gate" v={`${c.sign_used_with_open_gate.filter(Boolean).length}/${c.tokens.length} tokens`} />
                <KV k="direct logit attribution" v={fmt(c.direct_logit_attribution_at_prediction, 4)} />
                <KV k="p base → ablated" v={`${fmt(c.baseline.prob, 4)} → ${fmt(c.ablated.prob, 4)}`} />
                <KV k="Δ log p (nats)" v={fmt(c.delta.logprob, 4)} />
                <KV k="Δ logit / Δ rank" v={`${fmt(c.delta.logit, 4)} / ${c.delta.rank}`} />
                <KV k="controls mean ± sd" v={`${fmt(c.controls.summary.mean, 4)} ± ${fmt(c.controls.summary.std, 4)}`} />
                <KV k="z vs controls / percentile" v={`${fmt(c.controls.z ?? NaN, 2)} / ${fmt(c.controls.percentile_of_neuron ?? NaN, 0)}%`} />
              </div>
              <Plot height={170} data={[{ type: "histogram", x: c.controls.delta_logprob, name: "random directions", nbinsx: 20, marker: { color: "#94a3b8" } }]}
                layout={{ shapes: [{ type: "line", x0: c.delta.logprob, x1: c.delta.logprob, yref: "paper", y0: 0, y1: 1, line: { color: "#ef4444", width: 2 } }],
                  xaxis: { title: { text: "Δ log p(target)" } }, margin: { l: 32, r: 8, t: 8, b: 36 }, showlegend: false }} />
              <div className="text-[11px] text-muted">red line = this neuron; grey = {c.controls.n} matched random directions</div>
              <div className="text-[11px] mt-1">{c.reason}</div>
              <Expand label="top-k before / after ablation">
                <TopKTable rows={c.baseline_top_k} /><div className="h-1" /><TopKTable rows={c.ablated_top_k} compareTo={c.baseline_top_k} />
              </Expand>
            </Section>))}
        </div>}
      </>}
      right={<>
        {res && <Section title="Verdict">
          <div className="text-lg font-semibold" style={{ color: STATUS_COLOR[res.overall] }}>{res.overall}</div>
          <div className="text-[12px]">L{res.layer} n{res.neuron} ({res.sign}) · {fmt(res.runtime_seconds, 1)} s</div>
          <div className="text-[11px] text-muted mt-1">{res.criteria.supported}. {res.criteria.note}</div>
        </Section>}
        {res && <Section title={`Logit-lens readout of ${res.sign}w`}>
          <div className="font-mono text-[12px] leading-5">{(res.sign === "+" ? res.readout_logit_lens.plus : res.readout_logit_lens.minus).top_by_score.slice(0, 15)
            .map((r, i) => <div key={i}>{showTok(r.text)} <span className="text-muted num">{fmt(r.score, 3)}</span></div>)}</div>
          <Learn>The paper reads neurons with the J-lens (Phase 3). Treat this readout as a hypothesis, never as a classification.</Learn>
        </Section>}
        <Section title="Method">
          <div className="text-[12px] leading-5">
            <div>Type 1 = activates on unrelated inputs but adds one concept in all of them.</div>
            <div className="num mt-1">ablation: a_t ← 0 ∀t (removes a_t·w)</div>
            <div className="num">control: add −a_t·r ∀t, ‖r‖ = ‖w‖</div>
            <Learn>The control removes a vector of exactly the same size at the same positions and layer, but in a random direction.
              A neuron matters specifically only if its ablation lowers the target much more than these do (z ≤ −3, drop ≥ 0.03 nats; thresholds from Appendix B).</Learn>
          </div>
        </Section>
        {res && <Section title="Provenance"><Expand label="show"><pre className="text-[10px] num whitespace-pre-wrap break-all">{JSON.stringify(res.provenance, null, 1)}</pre></Expand></Section>}
      </>}
    />
  );
}

const TABS = { type1: "Type 1: one concept, many inputs", type2: "Type 2: two signs, two concepts", type3: "Type 3: one sign, two concepts (4 checks)", scan: "Type 3 scanner (exploratory)" } as const;

export default function Polysemanticity() {
  const [tab, setTab] = useState<keyof typeof TABS>("type1");
  return (
    <div className="h-full flex flex-col">
      <div className="flex gap-1 px-2 pt-1 border-b border-line shrink-0">
        {Object.entries(TABS).map(([k, v]) => <button key={k} onClick={() => setTab(k as keyof typeof TABS)}
          className={`px-2 py-1 text-[12px] rounded-t ${tab === k ? "bg-panel border border-b-0 border-line font-medium" : "text-muted"}`}>{v}</button>)}
      </div>
      <div className="flex-1 min-h-0">{tab === "type1" ? <Type1View /> : tab === "type2" ? <Type2View /> : tab === "type3" ? <Type3View /> : <ScanView />}</div>
    </div>
  );
}
