"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ModelPanel } from "@/components/ModelPanel";
import { useNeuronHover } from "@/components/NeuronHover";
import { NeuronGrid } from "@/components/NeuronGrid";
import Plot from "@/components/Plot";
import { Workspace } from "@/components/Shell";
import { Btn, ErrorBox, Expand, fmt, KV, Learn, Section, showTok, TokenStrip, TopKTable } from "@/components/ui";
import { call, InspectResp, Intervention, InterventionResp, Op } from "@/lib/api";
import { useDebounced, useLab } from "@/lib/state";

type Edit = { op: Op; value: number } | null;

export default function Microscope() {
  const { backend, arch, prompt, setPrompt, layer, setLayer, neuron, setNeuron, position, setPosition } = useLab();
  const [ins, setIns] = useState<InspectResp | null>(null);
  const [cmp, setCmp] = useState<InterventionResp | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [edit, setEdit] = useState<Edit>(null);
  const [scope, setScope] = useState<"token" | "all">("token");
  const [scale, setScale] = useState(2);
  const [targets, setTargets] = useState(" John");
  const [search, setSearch] = useState("");
  const [vec, setVec] = useState<"w" | "g" | "u" | "x" | "contribution">("w");
  const [gridPoint, setGridPoint] = useState<"act" | "gate_pre" | "up">("act");

  const L = arch ? Math.min(layer, arch.n_layers - 1) : 0;
  const hov = useNeuronHover();
  const N = arch ? Math.min(neuron, arch.d_mlp - 1) : 0;
  const dPrompt = useDebounced(prompt, 500);

  const inspect = useCallback(async () => {
    if (!arch) return;
    try {
      setIns(await call<InspectResp>(backend, "/neurons/inspect", { prompt: dPrompt, layer: L, neuron: N, position }));
      setErr(null);
    } catch (e) { setErr((e as Error).message); setIns(null); }
  }, [arch, backend, dPrompt, L, N, position]);

  useEffect(() => { inspect(); }, [inspect]);
  useEffect(() => { setEdit(null); setCmp(null); }, [L, N, dPrompt]);

  const dEdit = useDebounced(edit, 250);
  const pos = ins?.position;
  useEffect(() => {
    if (!arch || !dEdit || pos === undefined) { setCmp(null); return; }
    const iv: Intervention = { layer: L, neurons: [N], op: dEdit.op, value: dEdit.value, positions: scope === "token" ? [pos] : null };
    const tg = targets.split(",").map((s) => s.replace(/^"|"$/g, "")).filter((s) => s.length > 0);
    call<InterventionResp>(backend, "/interventions/run", { prompt: dPrompt, position: pos, top_k: 12, interventions: [iv], target_tokens: tg })
      .then((r) => { setCmp(r); setErr(null); }).catch((e) => setErr((e as Error).message));
  }, [dEdit, scope, targets, arch, backend, dPrompt, L, N, pos]);

  const a = ins?.steps.activation ?? 0;
  const sliderMax = Math.max(1, Math.abs(a) * 4, ins ? ins.trace.max_abs_activation * 1.5 : 1);
  const effective = edit ? (edit.op === "zero" ? 0 : edit.op === "negate" ? -a : edit.op === "scale" ? a * edit.value : edit.op === "set" ? edit.value : a + edit.value) : a;

  const tracePlot = useMemo(() => ins ? [
    { type: "bar", name: "activation a", x: ins.tokens.map((t, i) => `${i}:${showTok(t.text)}`), y: ins.trace.activation,
      marker: { color: ins.trace.activation.map((v, i) => ins.trace.gate_open[i] ? (v >= 0 ? "#f97316" : "#3b82f6") : "#9ca3af") } },
    { type: "scatter", mode: "lines+markers", name: "gate g·x", x: ins.tokens.map((t, i) => `${i}:${showTok(t.text)}`), y: ins.trace.gate, yaxis: "y2", line: { width: 1 } },
  ] : [], [ins]);

  const vecPlot = useMemo(() => ins?.vectors ? [{ type: "scattergl", mode: "lines", y: ins.vectors[vec], line: { width: 1 }, name: vec }] : [], [ins, vec]);

  const residPlot = useMemo(() => cmp ? [{ type: "bar", x: cmp.residual_delta_by_layer.map((r) => r.layer),
    y: cmp.residual_delta_by_layer.map((r) => r.delta_norm / Math.max(1e-9, r.base_norm)), name: "‖Δh‖/‖h‖" }] : [], [cmp]);

  return (
    <Workspace
      left={<>
        <ModelPanel />
        <Section title="Prompt">
          <textarea className="w-full h-20 font-mono text-[12px]" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        </Section>
        {arch && <Section title="Neuron">
          <div className="flex gap-2 items-center">
            <label className="text-muted">L</label>
            <input type="number" className="w-16 num" min={0} max={arch.n_layers - 1} value={L} onChange={(e) => setLayer(+e.target.value)} />
            <label className="text-muted">n</label>
            <input type="number" className="w-20 num" min={0} max={arch.d_mlp - 1} value={N} onChange={(e) => setNeuron(+e.target.value)} />
          </div>
          <input className="w-full mt-1 num" placeholder='search e.g. "L12 n345" then Enter' value={search} onChange={(e) => setSearch(e.target.value)}
            onKeyDown={(e) => {
              const m = search.match(/L?\s*(\d+)\D+n?\s*(\d+)/i);
              if (e.key === "Enter" && m) { setLayer(+m[1]); setNeuron(+m[2]); }
            }} />
          <div className="text-[11px] text-muted mt-1 num">layer range 0–{arch.n_layers - 1}, neuron range 0–{arch.d_mlp - 1}</div>
        </Section>}
        {ins && <Section title="Token position">
          <TokenStrip tokens={ins.tokens} selected={ins.position} onSelect={setPosition} values={ins.trace.activation} gateOpen={ins.trace.gate_open} />
          <Learn>Tokens are colored by this neuron&apos;s activation. ⊘ = gate closed (g·x ≤ 0).</Learn>
        </Section>}
        {ins && <Section title="Intervene on this neuron">
          <div className="flex gap-1 flex-wrap">
            <Btn onClick={() => setEdit({ op: "zero", value: 0 })}>zero</Btn>
            <Btn onClick={() => setEdit({ op: "negate", value: 0 })}>reverse sign</Btn>
            <Btn onClick={() => setEdit({ op: "scale", value: scale })}>scale ×</Btn>
            <input type="number" step="0.1" className="w-16 num" value={scale} onChange={(e) => { setScale(+e.target.value); if (edit?.op === "scale") setEdit({ op: "scale", value: +e.target.value }); }} />
            <Btn kind="danger" onClick={() => setEdit(null)}>restore</Btn>
          </div>
          <div className="mt-2">
            <div className="flex justify-between text-[11px] text-muted"><span>override a</span><span className="num">{fmt(effective, 4)}</span></div>
            <input type="range" className="w-full" min={-sliderMax} max={sliderMax} step={sliderMax / 500}
              value={edit?.op === "set" ? edit.value : a} onChange={(e) => setEdit({ op: "set", value: +e.target.value })} />
          </div>
          <div className="flex gap-2 text-[12px] mt-1">
            <label><input type="radio" checked={scope === "token"} onChange={() => setScope("token")} /> this token only</label>
            <label><input type="radio" checked={scope === "all"} onChange={() => setScope("all")} /> all positions</label>
          </div>
          <div className="mt-1">
            <label className="text-[11px] text-muted">track target tokens (exact strings, comma-separated; leading spaces matter)</label>
            <input className="w-full font-mono" value={targets} onChange={(e) => setTargets(e.target.value)} />
          </div>
          <Learn>Edits replace a_i before down_proj in a separate forward pass. The baseline pass is never modified, and model weights are never changed.</Learn>
        </Section>}
        <ErrorBox error={err} />
      </>}
      center={<>
        {!arch && <ErrorBox error="No model loaded." />}
        {ins && <Section title={`L${ins.layer} n${ins.neuron} across tokens`}>
          <Plot data={tracePlot} height={230} onClick={(p) => setPosition(parseInt(String(p.x).split(":")[0]))}
            layout={{ yaxis2: { overlaying: "y", side: "right", showgrid: false, title: { text: "gate" } }, yaxis: { title: { text: "a" } }, margin: { l: 48, r: 48, t: 8, b: 70 } }} />
          <Learn>Bars: activation a = silu(g·x)(u·x). Grey bars have a closed gate (g·x ≤ 0); SiLU is not exactly zero there
            (minimum ≈ −0.28), so small values leak through. The paper only counts a sign as used when it occurs with the gate open.</Learn>
        </Section>}
        {arch && <Section title={`Layer ${L} — all neurons × tokens`} right={
          <select value={gridPoint} onChange={(e) => setGridPoint(e.target.value as "act" | "gate_pre" | "up")}>
            <option value="act">activation a</option><option value="gate_pre">gate g·x</option><option value="up">up u·x</option>
          </select>}>
          <NeuronGrid backend={backend} prompt={dPrompt} layer={L} dMlp={arch.d_mlp} selected={N} onSelect={setNeuron} point={gridPoint} />
        </Section>}
        {ins?.vectors && <Section title="Learned weights and vectors" right={
          <select value={vec} onChange={(e) => setVec(e.target.value as typeof vec)}>
            <option value="w">w (down_proj[:, i]) — output vector</option>
            <option value="g">g (gate_proj[i]) — gate row</option>
            <option value="u">u (up_proj[i]) — up row</option>
            <option value="x">x — normalized MLP input at token</option>
            <option value="contribution">a·w — residual contribution</option>
          </select>}>
          <Plot data={vecPlot} height={200} layout={{ xaxis: { title: { text: `residual dim (0–${(arch?.d_model ?? 1) - 1})` } }, margin: { l: 48, r: 8, t: 8, b: 36 } }} />
          <Expand label="raw values">
            <pre className="text-[10px] num max-h-40 overflow-auto whitespace-pre-wrap">{ins.vectors[vec].map((v, i) => `${i}\t${v}`).join("\n")}</pre>
          </Expand>
        </Section>}
        {cmp && <Section title="Change in residual stream at the selected token, by layer">
          <Plot data={residPlot} height={180} layout={{ xaxis: { title: { text: "layer" } }, yaxis: { title: { text: "‖Δh‖ / ‖h‖" } }, margin: { l: 56, r: 8, t: 8, b: 36 } }} />
          <Learn>Zero before the intervened layer (nothing upstream changes); after it, the change propagates through real downstream computation.</Learn>
        </Section>}
      </>}
      right={<>
        {ins && <Section title={`Computation at token ${ins.position} (${JSON.stringify(ins.tokens[ins.position]?.text)})`}>
          <Learn>x is the RMS-normalized MLP input, not the raw residual stream. All steps below are recomputed in float64 from the captured x and the neuron&apos;s weights.</Learn>
          <div className="num text-[12px] leading-5">
            <div>gate = g·x = <b>{fmt(ins.steps.gate, 6)}</b> <span className={ins.steps.gate_open ? "text-ok" : "text-warn"}>({ins.steps.gate_open ? "open" : "closed"})</span></div>
            <div>up = u·x = <b>{fmt(ins.steps.up, 6)}</b></div>
            <div>σ(gate) = {fmt(ins.steps.sigmoid_gate, 6)}</div>
            <div>silu = gate·σ(gate) = <b>{fmt(ins.steps.silu, 6)}</b></div>
            <div>a = silu·up = <b style={{ color: a >= 0 ? "var(--pos)" : "var(--neg)" }}>{fmt(a, 6)}</b> → {ins.steps.sign_meaning}</div>
            <div>contribution = a·w, ‖a·w‖ = {fmt(ins.contribution.norm, 5)} (‖w‖ = {fmt(ins.contribution.w_norm, 4)})</div>
            {ins.steps.closed_gate_leak && <div className="text-warn">closed-gate leakage: this sign is not counted as a used sign</div>}
          </div>
          <Expand label="hooked values & recompute error">
            <KV k="hooked gate / up / a" v={`${fmt(ins.hooked.gate, 6)} / ${fmt(ins.hooked.up, 6)} / ${fmt(ins.hooked.activation, 6)}`} />
            <KV k="|Δ| gate / up / a" v={`${fmt(ins.recompute_error.gate)} / ${fmt(ins.recompute_error.up)} / ${fmt(ins.recompute_error.activation)}`} />
          </Expand>
          <Expand label="x vs. raw residual stream">
            {Object.entries(ins.mlp_input_vs_raw_residual).map(([k, v]) => <KV key={k} k={k} v={typeof v === "number" ? fmt(v) : <span className="text-[10px]">{v}</span>} />)}
          </Expand>
          <Expand label="contribution relative to the MLP output">
            {Object.entries(ins.contribution).map(([k, v]) => <KV key={k} k={k} v={fmt(v)} />)}
          </Expand>
          <Expand label="tensor shapes">
            {Object.entries(ins.shapes).map(([k, v]) => <KV key={k} k={k} v={`[${v.join(", ")}]`} />)}
          </Expand>
        </Section>}
        {ins && <Section title="MLP reconstruction check">
          <div className="num text-[12px]">{ins.reconstruction.formula}</div>
          <KV k="max |error| (all tokens)" v={fmt(ins.reconstruction.max_abs_error)} />
          <KV k="max relative error" v={<span className={ins.reconstruction.max_rel_error < 1e-3 ? "text-ok" : "text-bad"}>{fmt(ins.reconstruction.max_rel_error)}</span>} />
          <KV k="bias included" v={String(ins.reconstruction.includes_bias)} />
          <Learn>The sum of every neuron&apos;s contribution is compared with the MLP output the hook captured.</Learn>
        </Section>}
        {ins && <Section title={`Top neurons in L${ins.layer} at this token`}>
          <div className="text-[11px] text-muted">by ‖a·w‖ ({ins.layer_top_neurons_at_position.n_gate_open}/{ins.layer_top_neurons_at_position.d_mlp} gates open)</div>
          <div className="flex flex-wrap gap-1 mt-1">{ins.layer_top_neurons_at_position.by_contribution_norm.map((n) => (
            <button key={n.neuron} {...hov(L, n.neuron)} onClick={() => setNeuron(n.neuron)} className="num text-[11px] border border-line rounded px-1"
              style={{ color: n.activation >= 0 ? "var(--pos)" : "var(--neg)" }}>n{n.neuron} {fmt(n.contribution_norm, 2)}</button>))}</div>
        </Section>}
        {cmp && <Section title="Baseline vs. modified (separate forward passes)">
          <KV k="intervention" v={`${cmp.interventions?.[0]?.op ?? edit?.op} ${edit?.op === "scale" || edit?.op === "set" ? fmt(edit.value, 3) : ""} @ ${scope === "token" ? `pos ${cmp.position}` : "all positions"}`} />
          <KV k="KL(base ‖ mod)" v={fmt(cmp.comparison.kl_base_to_mod, 6)} />
          <KV k="top-1 changed" v={String(cmp.comparison.top1_changed)} />
          {cmp.comparison.targets.map((t) => (
            <div key={t.id} className="border-t border-line mt-1 pt-1">
              <div className="font-mono text-[12px]">{showTok(t.text)} <span className="text-muted">id {t.id}</span></div>
              <KV k="p base → mod" v={`${fmt(t.base.prob, 5)} → ${fmt(t.mod.prob, 5)}`} />
              <KV k="logit Δ / logprob Δ" v={`${fmt(t.mod.logit - t.base.logit, 4)} / ${fmt(t.mod.logprob - t.base.logprob, 4)}`} />
              <KV k="rank base → mod" v={`${t.base.rank} → ${t.mod.rank}`} />
            </div>))}
          <div className="mt-2"><div className="text-[11px] text-muted">baseline</div><TopKTable rows={cmp.baseline.top_k} /></div>
          <div className="mt-2"><div className="text-[11px] text-muted">modified</div><TopKTable rows={cmp.modified.top_k} compareTo={cmp.baseline.top_k} /></div>
          <div className="text-[11px] text-muted mt-2">Largest changes across the full vocabulary (Δ log-prob)</div>
          <table className="w-full text-[11px] num">
            <tbody>
              {[...cmp.comparison.largest_logprob_increase.slice(0, 6), ...cmp.comparison.largest_logprob_decrease.slice(0, 6)].map((r, i) => (
                <tr key={i} className="border-t border-line"><td className="font-mono">{showTok(r.text)}</td>
                  <td className="text-right" style={{ color: r.delta_logprob >= 0 ? "var(--pos)" : "var(--neg)" }}>{fmt(r.delta_logprob, 4)}</td>
                  <td className="text-right text-muted">{fmt(r.base_prob, 5)}→{fmt(r.mod_prob, 5)}</td></tr>))}
            </tbody>
          </table>
          <Expand label="intervention integrity checks">
            {cmp.layer_checks.map((c) => (<div key={c.layer}>
              <KV k={`L${c.layer} activations changed`} v={`${c.n_activations_changed} at positions [${c.positions_changed.join(", ")}]`} />
              <KV k="pre-edit a matches baseline" v={String(c.pre_edit_activation_matches_baseline)} />
              <KV k="max |Δmlp_out − Σ(a′−a)w|" v={fmt(c.max_abs_error_observed_vs_expected_delta)} />
            </div>))}
            <KV k="hooks remaining" v={String(cmp.hooks_remaining)} />
          </Expand>
          <Expand label="provenance"><pre className="text-[10px] num whitespace-pre-wrap break-all">{JSON.stringify(cmp.provenance, null, 1)}</pre></Expand>
        </Section>}
      </>}
    />
  );
}
