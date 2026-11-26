"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ArchitectureGraph } from "@/components/ArchitectureGraph";
import { ModelPanel } from "@/components/ModelPanel";
import Plot from "@/components/Plot";
import { Workspace } from "@/components/Shell";
import { Btn, ErrorBox, Expand, fmt, KV, Learn, Section, showTok, TokenStrip, TopKTable } from "@/components/ui";
import { call, InferenceResp } from "@/lib/api";
import { useLab } from "@/lib/state";

const METRICS = {
  resid_post_norm: "‖resid_post‖", mlp_out_norm: "‖MLP out‖", attn_out_norm: "‖attn out‖",
  gate_open_frac: "gate-open fraction", act_abs_max: "max |activation|",
} as const;
type Metric = keyof typeof METRICS;

export default function Explorer() {
  const { backend, arch, prompt, setPrompt, layer, setLayer, setNeuron, position, setPosition } = useLab();
  const [inf, setInf] = useState<InferenceResp | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [metric, setMetric] = useState<Metric>("mlp_out_norm");
  const [head, setHead] = useState(0);
  const [showAttn, setShowAttn] = useState(false);
  const [logScale, setLogScale] = useState(true);
  const L = arch ? Math.min(layer, arch.n_layers - 1) : 0;

  const run = useCallback(async () => {
    if (!arch) return;
    setBusy(true); setErr(null);
    try {
      setInf(await call<InferenceResp>(backend, "/inference", { prompt, position, top_k: 15, top_neurons_per_layer: 10,
        attention_layer: showAttn ? L : null }));
    } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  }, [arch, backend, prompt, position, showAttn, L]);

  useEffect(() => { if (arch) run(); }, [arch, position, L, showAttn]); // eslint-disable-line react-hooks/exhaustive-deps

  const heat = useMemo(() => {
    if (!inf?.layers) return null;
    const z = inf.layers.map((l) => l[metric]);
    return [{ type: "heatmap", z: logScale ? z.map((r) => r.map((v) => Math.log10(1e-6 + v))) : z,
      customdata: z, colorbar: { title: { text: logScale ? "log10" : "" } }, x: inf.tokens.map((t, i) => `${i}:${showTok(t.text)}`),
      y: inf.layers.map((l) => `L${l.layer}`), colorscale: "Viridis", hovertemplate: "%{y} %{x}<br>%{customdata:.4f}<extra></extra>" }];
  }, [inf, metric, logScale]);

  const s = inf?.layers?.[L];
  const t = inf?.position ?? 0;
  const la = arch?.layers[L];

  return (
    <Workspace
      left={<>
        <ModelPanel />
        <Section title="Prompt">
          <textarea className="w-full h-24 font-mono text-[12px]" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          <div className="flex gap-1 mt-1"><Btn kind="primary" onClick={run} disabled={!arch || busy}>{busy ? "running…" : "run forward pass"}</Btn></div>
          <ErrorBox error={err} />
        </Section>
        {inf && <Section title={`Tokens (${inf.tokens.length}) — click to select position`}>
          <TokenStrip tokens={inf.tokens} selected={inf.position} onSelect={setPosition} />
          <Learn>· marks a space, ↵ a newline. Hover a token for its id and raw tokenizer piece.</Learn>
        </Section>}
        {arch && <Section title="Selection">
          <div className="flex items-center gap-2">
            <label className="text-muted">layer</label>
            <input type="number" min={0} max={arch.n_layers - 1} value={L} className="w-16 num" onChange={(e) => setLayer(+e.target.value)} />
            <label className="text-muted">head</label>
            <input type="number" min={0} max={arch.n_heads - 1} value={head} className="w-14 num" onChange={(e) => setHead(+e.target.value)} />
          </div>
          <label className="flex gap-1 mt-1 items-center text-[12px]"><input type="checkbox" checked={showAttn} onChange={(e) => setShowAttn(e.target.checked)} /> capture attention pattern of layer {L}</label>
        </Section>}
      </>}
      center={<>
        {!arch && <ErrorBox error="No model loaded. Load one in the left panel (backend must be running)." />}
        {arch && <Section title="Architecture (actual loaded model) — click a block">
          <ArchitectureGraph arch={arch} inf={inf} selectedLayer={L} onSelectLayer={setLayer} />
        </Section>}
        {heat && <Section title="Layer × token" right={<div className="flex gap-2 items-center">
          <label className="text-[11px]"><input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} /> log color</label>
          <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
            {Object.entries(METRICS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select></div>}>
          <Plot data={heat} height={Math.max(260, 14 * (arch?.n_layers ?? 20))}
            layout={{ yaxis: { autorange: "reversed" }, margin: { l: 44, r: 8, t: 8, b: 80 } }}
            onClick={(p) => { setLayer(parseInt(String(p.y).slice(1))); setPosition(parseInt(String(p.x).split(":")[0])); }} />
          <Learn>Each cell is computed from the real forward pass. Click a cell to select that layer and token. The first
            position often has residual norms orders of magnitude larger than the rest (an attention sink), so a log color scale is the default.</Learn>
        </Section>}
        {arch && la && <Section title={`Block ${L} internals at position ${t}`}>
          <div className="grid grid-cols-2 gap-x-4 text-[12px]">
            <div>
              <div className="font-medium mb-1">Attention</div>
              <KV k="input" v="RMSNorm_in(resid_pre)" />
              <KV k="q / k / v / o" v={`${la.attention.q_proj.join("×")} / ${la.attention.k_proj.join("×")} / ${la.attention.v_proj.join("×")} / ${la.attention.o_proj.join("×")}`} />
              <KV k="heads / kv heads / head_dim" v={`${arch.n_heads} / ${arch.n_kv_heads} / ${arch.head_dim}`} />
              <KV k="qkv bias / qk-norm" v={`${la.attention.q_bias} / ${la.attention.qk_norm}`} />
              {s && <KV k="‖resid_pre‖ → ‖attn_out‖" v={`${fmt(s.resid_pre_norm[t], 2)} → ${fmt(s.attn_out_norm[t], 2)}`} />}
            </div>
            <div>
              <div className="font-medium mb-1">MLP (SwiGLU)</div>
              <KV k="x" v="RMSNorm_post(resid_mid)" />
              <KV k="W_gate / W_up / W_down" v={`${la.mlp.gate_proj.join("×")} / ${la.mlp.up_proj.join("×")} / ${la.mlp.down_proj.join("×")}`} />
              <KV k="a" v="silu(W_gate x) ⊙ (W_up x)" />
              <KV k="bias" v={String(la.mlp.bias)} />
              {s && <KV k="‖mlp_out‖ / gate-open" v={`${fmt(s.mlp_out_norm[t], 2)} / ${fmt(100 * s.gate_open_frac[t], 1)}%`} />}
            </div>
          </div>
          {s && <div className="mt-2">
            <div className="text-[11px] text-muted mb-1">Largest |activation| neurons at this token (click to open in microscope)</div>
            <div className="flex flex-wrap gap-1">
              {s.top_neurons_at_position.map((n) => (
                <Link key={n.neuron} href="/microscope" onClick={() => { setNeuron(n.neuron); setLayer(L); }}
                  className="border border-line rounded px-1 num text-[11px]" style={{ color: n.activation >= 0 ? "var(--pos)" : "var(--neg)" }}>
                  n{n.neuron} {fmt(n.activation, 2)}
                </Link>))}
            </div>
          </div>}
          {inf?.attention && inf.attention.layer === L && <div className="mt-2">
            <div className="text-[11px] text-muted">Head {head} attention probabilities (query row → key column), layer {L}</div>
            <Plot height={300} data={[{ type: "heatmap", z: inf.attention.pattern[Math.min(head, inf.attention.pattern.length - 1)],
              x: inf.tokens.map((k, i) => `${i}:${showTok(k.text)}`), y: inf.tokens.map((k, i) => `${i}:${showTok(k.text)}`),
              colorscale: "Blues", zmin: 0, zmax: 1, hovertemplate: "q %{y}<br>k %{x}<br>p %{z:.4f}<extra></extra>" }]}
              layout={{ yaxis: { autorange: "reversed" }, margin: { l: 90, r: 8, t: 8, b: 90 } }} />
          </div>}
        </Section>}
      </>}
      right={<>
        {inf && <Section title={`Next-token prediction at position ${inf.position}`}>
          <Learn>Full forward pass: logits = W_U · RMSNorm_final(h_last). This is the model&apos;s prediction, not a lens readout.</Learn>
          <TopKTable rows={inf.top_k} />
          <div className="text-[11px] text-muted mt-1 num">forward {fmt(inf.forward_seconds, 3)} s</div>
        </Section>}
        {arch && <Section title="Dataflow (per block)">
          <pre className="text-[11px] num whitespace-pre-wrap">{arch.dataflow.join("\n")}</pre>
        </Section>}
        {inf && <Section title="Provenance">
          <Expand label="show reproducibility record">
            <pre className="text-[10px] num whitespace-pre-wrap break-all">{JSON.stringify(inf.provenance, null, 1)}</pre>
          </Expand>
        </Section>}
      </>}
    />
  );
}
