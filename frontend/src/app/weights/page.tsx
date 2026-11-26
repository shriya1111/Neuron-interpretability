"use client";

import { useCallback, useEffect, useState } from "react";
import { ModelPanel } from "@/components/ModelPanel";
import { Workspace } from "@/components/Shell";
import { Btn, ErrorBox, Expand, fmt, KV, Learn, Section, showTok, TopKTable } from "@/components/ui";
import { call, DeltaRow, TopK } from "@/lib/api";
import { useLab } from "@/lib/state";

type Edit = { id: number; kind: string; layer: number; neurons: number[]; params: Record<string, unknown>; created_utc: string;
  touches: string[]; checksum_before: string; checksum_after: string };
type Stats = { logit: number; logprob: number; prob: number; rank: number };
type CmpRow = { prompt: string; original_top_k: TopK[]; edited_top_k: TopK[];
  comparison: { kl_base_to_mod: number; top1_changed: boolean; largest_logprob_increase: DeltaRow[]; largest_logprob_decrease: DeltaRow[];
    targets: { id: number; text: string; base: Stats; mod: Stats }[] } };
type CmpResp = { prompts: CmpRow[]; heldout?: { n_texts: number; dataset: string; original: { mean_nll: number; perplexity: number; tokens: number };
  edited: { mean_nll: number; perplexity: number }; delta_mean_nll: number; perplexity_ratio: number } };

const KINDS = {
  prune: "prune — zero g, u and w (neuron removed)",
  scale_output: "scale output vector w by a factor",
  scale_input: "scale input rows g and/or u",
  add_direction: "add a direction to w (token, neuron or vector)",
  set_output: "replace w with another neuron's w",
} as const;
type Kind = keyof typeof KINDS;

function parseNeurons(s: string): number[] {
  const out: number[] = [];
  for (const part of s.split(",").map((x) => x.trim()).filter(Boolean)) {
    const m = part.match(/^(\d+)\s*-\s*(\d+)$/);
    if (m) { for (let i = +m[1]; i <= +m[2]; i++) out.push(i); } else if (/^\d+$/.test(part)) out.push(+part);
    else throw new Error(`bad neuron list item: ${part}`);
  }
  return out;
}

export default function Weights() {
  const { backend, arch, layer, setLayer, neuron } = useLab();
  const [kind, setKind] = useState<Kind>("prune");
  const [neurons, setNeurons] = useState(String(neuron));
  const [factor, setFactor] = useState(0);
  const [gate, setGate] = useState(true);
  const [up, setUp] = useState(true);
  const [dirSource, setDirSource] = useState<"token" | "neuron">("token");
  const [dirToken, setDirToken] = useState(" John");
  const [srcL, setSrcL] = useState(0);
  const [srcN, setSrcN] = useState(0);
  const [alpha, setAlpha] = useState(1);
  const [relative, setRelative] = useState(true);
  const [edits, setEdits] = useState<Edit[]>([]);
  const [prompts, setPrompts] = useState("The fourth Gospel was written by the apostle | John\nThe British singer and pianist Elton | John\nThe capital of France is | Paris");
  const [nHeld, setNHeld] = useState(10);
  const [cmp, setCmp] = useState<CmpResp | null>(null);
  const [exportName, setExportName] = useState("my-edited-qwen3");
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try { setEdits((await call<{ edits: Edit[] }>(backend, "/weights/edits")).edits); } catch { /* backend offline */ }
    window.dispatchEvent(new Event("nl-weights-changed"));
  }, [backend]);
  useEffect(() => { refresh(); }, [refresh]);

  async function apply() {
    setErr(null); setMsg(null); setBusy(true);
    try {
      const params: Record<string, unknown> = {};
      if (kind === "scale_output") params.factor = factor;
      if (kind === "scale_input") { params.factor = factor; params.which = [gate && "gate", up && "up"].filter(Boolean); }
      if (kind === "add_direction") {
        params.alpha = alpha; params.alpha_relative_to_norm = relative;
        params.direction = dirSource === "token" ? { source: "token", token: dirToken } : { source: "neuron", layer: srcL, neuron: srcN };
      }
      if (kind === "set_output") {
        const r = await call<{ vectors: { w: number[] } }>(backend, "/neurons/inspect", { prompt: "x", layer: srcL, neuron: srcN, position: 0 });
        params.vector = r.vectors.w;
      }
      const r = await call<{ applied: Edit }>(backend, "/weights/edit", { kind, layer, neurons: parseNeurons(neurons), params });
      setMsg(`applied edit #${r.applied.id}`); await refresh();
    } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  }
  async function revert(id: number | null) {
    setErr(null);
    try { await call(backend, "/weights/revert", id === null ? { all: true } : { edit_id: id }); await refresh(); setCmp(null); }
    catch (e) { setErr((e as Error).message); }
  }
  async function compare() {
    setErr(null); setBusy(true);
    try {
      const ps = prompts.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
        const [p, t] = l.split("|"); return { prompt: p.trim(), targets: t ? t.split(",").map((x) => " " + x.trim()) : [] };
      });
      setCmp(await call<CmpResp>(backend, "/weights/compare", { prompts: ps, top_k: 8, n_heldout: nHeld }));
    } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  }
  async function doExport() {
    setErr(null); setBusy(true);
    try { const r = await call<{ path: string }>(backend, "/weights/export", { name: exportName }); setMsg(`saved to ${r.path}`); }
    catch (e) { setErr((e as Error).message); }
    setBusy(false);
  }

  return (
    <Workspace
      left={<>
        <ModelPanel />
        {arch && <Section title="New weight edit">
          <select className="w-full" value={kind} onChange={(e) => setKind(e.target.value as Kind)}>
            {Object.entries(KINDS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <div className="flex gap-2 items-center mt-1">
            <label className="text-muted">layer</label>
            <input type="number" className="w-14 num" min={0} max={arch.n_layers - 1} value={layer} onChange={(e) => setLayer(+e.target.value)} />
            <label className="text-muted">neurons</label>
            <input className="flex-1 num" value={neurons} onChange={(e) => setNeurons(e.target.value)} title="e.g. 5, 81, 100-120" />
          </div>
          {(kind === "scale_output" || kind === "scale_input") && <div className="flex gap-2 items-center mt-1">
            <label className="text-muted">factor</label>
            <input type="number" step="0.1" className="w-20 num" value={factor} onChange={(e) => setFactor(+e.target.value)} />
            {kind === "scale_input" && <><label><input type="checkbox" checked={gate} onChange={(e) => setGate(e.target.checked)} /> g</label>
              <label><input type="checkbox" checked={up} onChange={(e) => setUp(e.target.checked)} /> u</label></>}
          </div>}
          {kind === "add_direction" && <div className="mt-1 flex flex-col gap-1 text-[12px]">
            <div className="flex gap-2"><label><input type="radio" checked={dirSource === "token"} onChange={() => setDirSource("token")} /> token direction</label>
              <label><input type="radio" checked={dirSource === "neuron"} onChange={() => setDirSource("neuron")} /> another neuron&apos;s w</label></div>
            {dirSource === "token" ? <input className="font-mono" value={dirToken} onChange={(e) => setDirToken(e.target.value)} />
              : <div className="flex gap-1">L<input type="number" className="w-14 num" value={srcL} onChange={(e) => setSrcL(+e.target.value)} />
                n<input type="number" className="w-20 num" value={srcN} onChange={(e) => setSrcN(+e.target.value)} /></div>}
            <div className="flex gap-2 items-center">α <input type="number" step="0.1" className="w-20 num" value={alpha} onChange={(e) => setAlpha(+e.target.value)} />
              <label><input type="checkbox" checked={relative} onChange={(e) => setRelative(e.target.checked)} /> × ‖w‖</label></div>
          </div>}
          {kind === "set_output" && <div className="flex gap-1 mt-1 text-[12px] items-center">copy w from L<input type="number" className="w-14 num" value={srcL} onChange={(e) => setSrcL(+e.target.value)} />
            n<input type="number" className="w-20 num" value={srcN} onChange={(e) => setSrcN(+e.target.value)} /></div>}
          <div className="mt-2"><Btn kind="primary" onClick={apply} disabled={busy}>apply edit</Btn></div>
          {msg && <div className="text-ok text-[12px] mt-1 break-all">{msg}</div>}
          <Learn>These edits change the model&apos;s weights in memory and stay active for <b>every</b> page and request until you revert them.
            The files on disk are never touched. Use the Microscope&apos;s interventions for temporary, single-run changes instead.</Learn>
        </Section>}
        <ErrorBox error={err} />
      </>}
      center={<>
        <Section title="Compare edited vs. original weights">
          <label className="text-[11px] text-muted">one prompt per line: <span className="font-mono">prompt | target1, target2</span> (a leading space is added to each target)</label>
          <textarea className="w-full h-24 font-mono text-[12px]" value={prompts} onChange={(e) => setPrompts(e.target.value)} />
          <div className="flex gap-2 items-center mt-1 text-[12px]">held-out WikiText texts
            <input type="number" className="w-14 num" min={0} max={40} value={nHeld} onChange={(e) => setNHeld(+e.target.value)} />
            <Btn kind="primary" onClick={compare} disabled={busy || !edits.length}>{busy ? "running…" : "compare"}</Btn>
            {!edits.length && <span className="text-muted">apply an edit first</span>}
          </div>
          <Learn>The original weights are swapped back in temporarily (under the model lock) for the comparison runs, then the edits are restored.</Learn>
        </Section>
        {cmp?.heldout && <Section title="General capability: held-out text">
          <div className="grid grid-cols-2 gap-x-6">
            <KV k="perplexity original" v={fmt(cmp.heldout.original.perplexity, 3)} />
            <KV k="perplexity edited" v={fmt(cmp.heldout.edited.perplexity, 3)} />
            <KV k="Δ mean NLL (nats/token)" v={<span style={{ color: cmp.heldout.delta_mean_nll > 0 ? "var(--bad)" : "var(--ok)" }}>{fmt(cmp.heldout.delta_mean_nll, 5)}</span>} />
            <KV k="perplexity ratio" v={fmt(cmp.heldout.perplexity_ratio, 4)} />
          </div>
          <div className="text-[11px] text-muted mt-1">{cmp.heldout.n_texts} texts, {cmp.heldout.original.tokens} predicted tokens · {cmp.heldout.dataset}</div>
        </Section>}
        {cmp?.prompts.map((r, i) => (
          <Section key={i} title={r.prompt}>
            <div className="flex gap-4 text-[12px]"><KV k="KL(original ‖ edited)" v={fmt(r.comparison.kl_base_to_mod, 5)} /><KV k="top-1 changed" v={String(r.comparison.top1_changed)} /></div>
            {r.comparison.targets.map((t) => <div key={t.id} className="num text-[12px]">{showTok(t.text)}: p {fmt(t.base.prob, 4)} → {fmt(t.mod.prob, 4)} · rank {t.base.rank} → {t.mod.rank}</div>)}
            <div className="grid grid-cols-2 gap-2 mt-1">
              <div><div className="text-[11px] text-muted">original weights</div><TopKTable rows={r.original_top_k} /></div>
              <div><div className="text-[11px] text-muted">edited weights</div><TopKTable rows={r.edited_top_k} compareTo={r.original_top_k} /></div>
            </div>
          </Section>))}
      </>}
      right={<>
        <Section title={`Active edits (${edits.length})`} right={edits.length > 0 ? <button className="text-bad text-[11px]" onClick={() => revert(null)}>revert all</button> : undefined}>
          {edits.length === 0 && <div className="text-[12px] text-muted">The model has its original weights.</div>}
          {edits.map((e) => (
            <div key={e.id} className="border-t border-line py-1 text-[12px]">
              <div className="flex justify-between"><b>#{e.id} {e.kind}</b><button className="text-bad text-[11px]" onClick={() => revert(e.id)}>revert</button></div>
              <div className="num">L{e.layer} n[{e.neurons.length > 8 ? `${e.neurons.slice(0, 8).join(",")}… (${e.neurons.length})` : e.neurons.join(",")}]</div>
              <div className="num text-muted">{JSON.stringify(e.params)}</div>
              <Expand label="details"><div className="num text-[10px] break-all">touches: {e.touches.join("; ")}<br />checksum {e.checksum_before} → {e.checksum_after}<br />{e.created_utc}</div></Expand>
            </div>))}
          <Learn>Reverting restores the saved original values exactly. An edit that overlaps a later one must be reverted after it.</Learn>
        </Section>
        <Section title="Export edited model">
          <input className="w-full num" value={exportName} onChange={(e) => setExportName(e.target.value)} />
          <div className="mt-1"><Btn onClick={doExport} disabled={busy || !edits.length}>save copy to data/edited_models/</Btn></div>
          <div className="text-[11px] text-muted mt-1">Writes safetensors, the tokenizer and neuronlab_edits.json (the edit log) to a new folder. The original checkpoint is unchanged.</div>
        </Section>
        <Section title="What each edit does">
          <div className="num text-[11px] leading-5">
            <div>prune: g_i, u_i, w_i ← 0 ⇒ a_i ≡ 0</div>
            <div>scale_output: w_i ← s·w_i</div>
            <div>scale_input: g_i ← s·g_i and/or u_i ← s·u_i</div>
            <div>add_direction: w_i ← w_i + α·d/‖d‖</div>
            <div>set_output: w_i ← w_j (copied)</div>
          </div>
          <Learn>Scaling u by s scales the activation exactly by s (a = silu(g·x)(u·x) is linear in u). Scaling g changes <i>when</i> the neuron fires,
            because SiLU is nonlinear. Pruning is the permanent version of zero-ablating the neuron at every position.</Learn>
        </Section>
      </>}
    />
  );
}
