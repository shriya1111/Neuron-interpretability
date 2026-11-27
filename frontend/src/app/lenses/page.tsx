"use client";

import { useState } from "react";
import { JCensusPanel, JNeuronPanel, JResidualPanel, JValidatePanel } from "@/components/JLensPanels";
import { ModelPanel } from "@/components/ModelPanel";
import { useNeuronHover } from "@/components/NeuronHover";
import Plot from "@/components/Plot";
import { Workspace } from "@/components/Shell";
import { Btn, ErrorBox, fmt, Learn, Section, showTok, TokenStrip, TopKTable } from "@/components/ui";
import { call, Tok, TopK } from "@/lib/api";
import { useLab } from "@/lib/state";

type Row = { id: number; text: string; score: number; cosine: number; rank?: number };
type Side = { top_by_score: Row[]; top_by_cosine: Row[]; bottom_by_score: Row[];
  targets: { id: number; text: string; score: number; cosine: number; rank_by_score: number; rank_by_cosine: number }[] };
type DirResp = { method: string; norm_v: number; plus: Side; minus: Side; note?: string; layer?: number; neuron?: number };
type ResidResp = { method: string; position: number; tokens: Tok[]; note: string; model_prediction: TopK[];
  last_layer_matches_model_max_abs_logit_error: number;
  layers: { state: string; top_k: TopK[]; resid_norm: number; targets: { id: number; text: string; prob: number; rank: number; logit: number }[] }[] };
type SearchResp = { method: string; candidates: { layer: number; neuron: number; sign: string; cosine: number; score: number;
  rank_of_target: number; readout_top5: string[] }[] };

type Mode = "neuron" | "residual" | "search" | "jneuron" | "jresidual" | "jvalidate" | "jcensus";

export default function Lenses() {
  const { backend, arch, prompt, setPrompt, layer, setLayer, neuron, setNeuron, position, setPosition } = useLab();
  const [mode, setMode] = useState<Mode>("neuron");
  const [targets, setTargets] = useState(" John");
  const [sign, setSign] = useState<"+" | "-">("+");
  const [rank, setRank] = useState<"score" | "cosine">("score");
  const [dir, setDir] = useState<DirResp | null>(null);
  const [resid, setResid] = useState<ResidResp | null>(null);
  const [search, setSearch] = useState<SearchResp | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const tg = targets.split(",").filter((s) => s.length > 0);
  const hov = useNeuronHover();

  async function run() {
    setBusy(true); setErr(null);
    try {
      if (mode === "neuron") setDir(await call<DirResp>(backend, "/lens/logit", { kind: "neuron", layer, neuron, top_k: 25, target_tokens: tg }));
      else if (mode === "residual") setResid(await call<ResidResp>(backend, "/lens/logit", { kind: "residual", prompt, position, top_k: 5, target_tokens: tg }));
      else setSearch(await call<SearchResp>(backend, "/lens/neuron-search", { target: tg[0] ?? " John", sign, top: 40 }));
    } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  }

  const sideTable = (title: string, s: Side, color: string) => {
    const rows = rank === "score" ? s.top_by_score : s.top_by_cosine;
    return (
      <Section title={title}>
        <table className="w-full text-[12px] num"><thead className="text-muted"><tr><th className="text-left">#</th><th className="text-left">token</th>
          <th className="text-right">score</th><th className="text-right">cos</th></tr></thead>
          <tbody>{rows.map((r, i) => <tr key={r.id} className="border-t border-line"><td>{i + 1}</td>
            <td className="font-mono" style={{ color }}>{showTok(r.text)}</td><td className="text-right">{fmt(r.score, 4)}</td><td className="text-right">{fmt(r.cosine, 4)}</td></tr>)}</tbody></table>
        {s.targets.length > 0 && <div className="mt-2 text-[12px]">{s.targets.map((t) => (
          <div key={t.id} className="num">target {showTok(t.text)}: score {fmt(t.score, 4)} · rank {t.rank_by_score.toLocaleString()} by score, {t.rank_by_cosine.toLocaleString()} by cosine</div>))}</div>}
      </Section>
    );
  };

  return (
    <Workspace
      left={<>
        <ModelPanel />
        <Section title="Lens">
          <div className="flex flex-col gap-1 text-[12px]">
            <label><input type="radio" checked={mode === "neuron"} onChange={() => setMode("neuron")} /> neuron output vector (direction)</label>
            <label><input type="radio" checked={mode === "residual"} onChange={() => setMode("residual")} /> residual stream state at a token</label>
            <label><input type="radio" checked={mode === "search"} onChange={() => setMode("search")} /> find neurons that promote a token</label>
            <div className="text-[11px] text-muted mt-1">J-lens (fitted on 1,000 WikiText prompts)</div>
            <label><input type="radio" checked={mode === "jneuron"} onChange={() => setMode("jneuron")} /> neuron: J-lens readout + J-share</label>
            <label><input type="radio" checked={mode === "jresidual"} onChange={() => setMode("jresidual")} /> residual: J-lens across depth</label>
            <label><input type="radio" checked={mode === "jvalidate"} onChange={() => setMode("jvalidate")} /> validate linear predictions</label>
            <label><input type="radio" checked={mode === "jcensus"} onChange={() => setMode("jcensus")} /> J-share census, all neurons (Figure 2)</label>
          </div>
        </Section>
        <Section title="Inputs">
          {(mode === "neuron" || mode === "jneuron") && arch && <div className="flex gap-2 items-center">
            <label className="text-muted">L</label><input type="number" className="w-16 num" min={0} max={arch.n_layers - 1} value={layer} onChange={(e) => setLayer(+e.target.value)} />
            <label className="text-muted">n</label><input type="number" className="w-20 num" min={0} max={arch.d_mlp - 1} value={neuron} onChange={(e) => setNeuron(+e.target.value)} />
          </div>}
          {(mode === "residual" || mode === "jresidual" || mode === "jvalidate") && <>
            <textarea className="w-full h-20 font-mono text-[12px]" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            {resid && <TokenStrip tokens={resid.tokens} selected={resid.position} onSelect={setPosition} />}
          </>}
          {mode === "search" && <div className="flex gap-2 items-center text-[12px]">sign
            <select value={sign} onChange={(e) => setSign(e.target.value as "+" | "-")}><option>+</option><option>-</option></select></div>}
          <label className="text-[11px] text-muted mt-1 block">target token(s), exact strings, comma-separated</label>
          <input className="w-full font-mono" value={targets} onChange={(e) => setTargets(e.target.value)} />
          {!mode.startsWith("j") && <div className="mt-2"><Btn kind="primary" onClick={run} disabled={!arch || busy}>{busy ? "running…" : "run"}</Btn></div>}
          <ErrorBox error={err} />
        </Section>
      </>}
      center={<>
        {mode === "jneuron" && arch && <JNeuronPanel backend={backend} layer={layer} neuron={neuron} targets={tg} />}
        {mode === "jresidual" && <JResidualPanel backend={backend} prompt={prompt} position={position} targets={tg} />}
        {mode === "jcensus" && <JCensusPanel backend={backend} />}
        {mode === "jvalidate" && arch && <JValidatePanel backend={backend} prompt={prompt} layer={layer} nLayers={arch.n_layers} />}
        {mode === "neuron" && dir && <>
          <div className="flex items-center gap-2 text-[12px]">rank by
            <select value={rank} onChange={(e) => setRank(e.target.value as "score" | "cosine")}><option value="score">score (direct logit effect)</option><option value="cosine">cosine</option></select>
            <span className="text-muted num">‖w‖ = {fmt(dir.norm_v, 4)}</span></div>
          <div className="grid grid-cols-2 gap-2">
            {sideTable(`+w — L${dir.layer} n${dir.neuron} adds these when a > 0`, dir.plus, "var(--pos)")}
            {sideTable(`−w — adds these when a < 0`, dir.minus, "var(--neg)")}
          </div>
          <Learn>{dir.note}</Learn>
        </>}
        {mode === "residual" && resid && <>
          <Section title={`Logit lens across depth at position ${resid.position} (${JSON.stringify(resid.tokens[resid.position]?.text)})`}>
            <table className="w-full text-[11px] num">
              <thead className="text-muted"><tr><th className="text-left">state</th><th className="text-left" colSpan={5}>top-5 tokens (prob)</th>
                {resid.layers[0]?.targets.map((t) => <th key={t.id} className="text-right">{showTok(t.text)} rank</th>)}</tr></thead>
              <tbody>{resid.layers.map((l) => <tr key={l.state} className="border-t border-line">
                <td>{l.state}</td>{l.top_k.map((t) => <td key={t.id} className="font-mono">{showTok(t.text)} <span className="text-muted">{fmt(t.prob, 2)}</span></td>)}
                {l.targets.map((t) => <td key={t.id} className="text-right">{t.rank.toLocaleString()}</td>)}</tr>)}</tbody>
            </table>
            <div className="text-[11px] text-muted mt-1 num">last layer vs. model logits: max |Δ| = {fmt(resid.last_layer_matches_model_max_abs_logit_error)}</div>
          </Section>
          {resid.layers[0]?.targets.length > 0 && <Section title="Target token probability by depth (logit lens)">
            <Plot height={220} data={resid.layers[0].targets.map((t, j) => ({ type: "scatter", mode: "lines+markers", name: showTok(t.text),
              x: resid.layers.map((l) => l.state), y: resid.layers.map((l) => l.targets[j].prob) }))}
              layout={{ yaxis: { type: "log", title: { text: "p (log)" } }, margin: { l: 56, r: 8, t: 8, b: 40 } }} />
          </Section>}
          <Learn>{resid.note}</Learn>
        </>}
        {mode === "search" && search && <Section title={`Neurons whose ${sign}w most points toward the target (logit lens)`}>
          <table className="w-full text-[12px] num"><thead className="text-muted"><tr><th className="text-left">neuron</th><th className="text-right">cosine</th>
            <th className="text-right">score</th><th className="text-right">target rank in readout</th><th className="text-left">readout top-5</th></tr></thead>
            <tbody>{search.candidates.map((c) => <tr key={`${c.layer}-${c.neuron}`} className="border-t border-line cursor-pointer hover:bg-panel-2" {...hov(c.layer, c.neuron)}
              onClick={() => { setLayer(c.layer); setNeuron(c.neuron); setMode("neuron"); }}>
              <td className="text-accent">L{c.layer} n{c.neuron} ({c.sign})</td><td className="text-right">{fmt(c.cosine, 4)}</td><td className="text-right">{fmt(c.score, 4)}</td>
              <td className="text-right">{c.rank_of_target.toLocaleString()}</td><td className="font-mono">{c.readout_top5.map(showTok).join(" ")}</td></tr>)}</tbody></table>
          <Learn>Exploratory: these neurons have output vectors aligned with the token in the unembedding. That says nothing about when they fire,
            or whether they matter causally. Click a row to open its readout; test causality in the Polysemanticity Lab or Microscope.</Learn>
        </Section>}
      </>}
      right={<>
        <Section title="What is being computed">
          {mode === "neuron" && <div className="text-[12px] leading-5">
            <div className="num">score = W_U (γ_final ⊙ w)</div>
            <div className="num">cos_t = score_t / (‖W_U[t] ⊙ γ‖ ‖w‖)</div>
            <Learn>This is a <b>direction readout</b>: which tokens the vector w pushes up through the unembedding alone,
              per unit of activation. The final-norm gain γ is folded into W_U, as the paper specifies. A neuron with activation a adds a·w, so
              the −w column is what it adds when a &lt; 0. It is <b>not</b> a prediction of the next token.</Learn>
          </div>}
          {mode === "residual" && <div className="text-[12px] leading-5">
            <div className="num">p_L = softmax(W_U · RMSNorm_final(h_L))</div>
            <Learn>This lens applies the model&apos;s real final RMSNorm (including the 1/rms scale) to a <b>complete residual state</b>.
              Only the last row is the model&apos;s actual prediction.</Learn>
            {resid && <div className="mt-2"><div className="text-[11px] text-muted">model prediction</div><TopKTable rows={resid.model_prediction} /></div>}
          </div>}
          {mode === "search" && <div className="text-[12px] leading-5">
            <div className="num">u_t = W_U[t] ⊙ γ_final</div><div className="num">rank neurons by cos(u_t, ±w)</div>
            <Learn>An exact scan over every neuron in every layer. For each top candidate, the target&apos;s exact rank among all vocabulary tokens in that neuron&apos;s readout is also computed.</Learn>
          </div>}
        </Section>
        {mode.startsWith("j") && <Section title="J-lens: what is being computed">
          <div className="num text-[12px] leading-5">
            <div>J_l = E[∂h_final,t′ / ∂h_l,t], t′ ≥ t</div>
            <div>d_l,t = row t of (W_U·diag γ)·J_l, normalized</div>
            <div>readout(w) = tokens by cos(d_l,t, w)</div>
            <div>J-share(w) = 1 − ‖ŵ − v‖², v = Σ c_j d_j, c ≥ 0, ≤ 25 terms</div>
          </div>
          <Learn>A neuron in block l writes into the output of block l, so it is read with J_l; the last layer uses the unembedding.
            The lens is our own fit with Anthropic&apos;s reference code (1,000 WikiText-103 prompts × 128 tokens), not the paper&apos;s file. The paper studied larger models.</Learn>
        </Section>}
        <Section title="Limits of the logit lens">
          <div className="text-[12px] text-muted leading-5">It only sees the direct path to the output. Neurons in early and middle layers often act through
            later layers, and their logit-lens readouts look incoherent. The paper therefore reads neurons with the J-lens (Phase 3), which follows those indirect paths to first order.</div>
        </Section>
      </>}
    />
  );
}
