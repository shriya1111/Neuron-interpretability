"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { JNeuronPanel } from "@/components/JLensPanels";
import { useNeuronHover, useNeuronHoverRaw } from "@/components/NeuronHover";
import Plot from "@/components/Plot";
import { Btn, ErrorBox, fmt, KV, Learn, showTok } from "@/components/ui";
import { call, InterventionResp } from "@/lib/api";
import { useLab } from "@/lib/state";

type Firing = { doc: number; position: number; domain: string; tokens: string[]; peak_index: number; activation: number; prefix_token_ids: number[] };
type Side = { firings: Firing[]; n_open: number; fire_rate: number; threshold_q999: number; threshold_q999_excl_pos0: number;
  strong_rate_by_domain: Record<string, number>; writes_logit_lens: { id: number; text: string; score: number }[];
  writes_jlens?: { id: number; text: string; cosine: number }[];
  j_share?: { value: number; class: string; random_p1: number; random_p99: number; sign_used?: boolean } };
type NeuronResp = { layer: number; neuron: number; max_abs: number; plus: Side; minus: Side; notes: Record<string, string> };
type Overview = { meta: { model_id: string; n_docs: number; positions_scanned_excl_pos0: number; corpus: string }; layers: { layer: number; max_abs: number[]; fire_rate: number[] }[] };
type Row = { neuron: number; max_abs: number; fire_rate_pos: number; fire_rate_neg: number; top_pos: number; top_neg: number; writes_pos: string[]; writes_neg: string[] };

const slugOf = (m: string) => m.split("/").pop()!.toLowerCase();

/** One column per layer; each neuron is one pixel colored by its firing rate (log scale). */
function LayerStrip({ ov, selected, onPick }: { ov: Overview; selected: { L: number; n: number | null }; onPick: (L: number, n: number | null) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const hover = useNeuronHoverRaw();
  const nL = ov.layers.length, nN = ov.layers[0].fire_rate.length;
  const cols = 32, rows = Math.ceil(nN / cols), px = 3, gap = 6;
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    for (const l of ov.layers) {
      const x0 = l.layer * (cols * px + gap);
      for (let i = 0; i < nN; i++) {
        const r = l.fire_rate[i];
        const t = Math.min(1, Math.max(0, (Math.log10(r + 1e-6) + 6) / 6));
        ctx.fillStyle = `hsl(${250 - 70 * (l.layer / nL)}, 70%, ${88 - 55 * t}%)`;
        ctx.fillRect(x0 + (i % cols) * px, Math.floor(i / cols) * px, px, px);
      }
      if (selected.L === l.layer) { ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 2; ctx.strokeRect(x0 - 1, -1, cols * px + 2, rows * px + 2); }
    }
    if (selected.n !== null) {
      const x0 = selected.L * (cols * px + gap);
      ctx.fillStyle = "#ef4444"; ctx.fillRect(x0 + (selected.n % cols) * px - 1, Math.floor(selected.n / cols) * px - 1, px + 2, px + 2);
    }
  }, [ov, selected, nL, nN, cols, rows]);
  const locate = (e: { target: EventTarget; clientX: number; clientY: number }) => {
    const b = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const x = e.clientX - b.left, y = e.clientY - b.top;
    const L = Math.floor(x / (cols * px + gap)), xi = Math.floor((x - L * (cols * px + gap)) / px), yi = Math.floor(y / px);
    const n = yi * cols + xi;
    return L >= 0 && L < nL && xi >= 0 && xi < cols && n >= 0 && n < nN ? { L, n } : null;
  };
  return (
    <div className="overflow-x-auto">
      <canvas ref={ref} width={nL * (cols * px + gap)} height={rows * px} className="cursor-crosshair"
        onClick={(e) => { const h = locate(e); if (h) onPick(h.L, h.n); }}
        onMouseMove={(e) => { const h = locate(e); if (h) hover.show(h.L, h.n, e); else hover.hide(); }}
        onMouseLeave={hover.hide} />
      <div className="flex num text-[10px] text-muted">{ov.layers.map((l) => <div key={l.layer} style={{ width: cols * px + gap }}
        className="cursor-pointer" onClick={() => onPick(l.layer, null)}>L{l.layer}</div>)}</div>
    </div>
  );
}

function Snippet({ f, sign }: { f: Firing; sign: "+" | "-" }) {
  return (
    <div className="font-mono text-[11px] leading-5 border border-line rounded px-2 py-1 bg-panel-2">
      {f.tokens.map((t, i) => <span key={i} className={i === f.peak_index ? "rounded px-[1px] font-semibold" : ""}
        style={i === f.peak_index ? { background: `color-mix(in srgb, var(${sign === "+" ? "--pos" : "--neg"}) 45%, transparent)` } : undefined}>
        {t.replace(/\n/g, "↵")}</span>)}
      <span className="float-right text-muted num">{fmt(f.activation, 3)} · {f.domain}</span>
    </div>
  );
}

function SideCard({ title, sign, s }: { title: string; sign: "+" | "-"; s: Side }) {
  const domains = Object.entries(s.strong_rate_by_domain).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).slice(0, 8);
  return (
    <div className="border border-line rounded p-2 bg-panel">
      <div className="flex items-baseline justify-between">
        <div className="font-semibold" style={{ color: sign === "+" ? "var(--pos)" : "var(--neg)" }}>{title}</div>
        <div className="num text-[11px] text-muted">fires (gate open) on {fmt(100 * s.fire_rate, 3)}% of tokens · q99.9 = {fmt(s.threshold_q999_excl_pos0, 3)}
          {s.j_share && <> · J-share <b style={{ color: s.j_share.class === "high" ? "var(--ok)" : s.j_share.class === "low" ? "var(--bad)" : "var(--warn)" }}>{fmt(s.j_share.value, 3)} ({s.j_share.class})</b>{s.j_share.sign_used ? " · sign used" : ""}</>}</div>
      </div>
      <div className="grid grid-cols-[minmax(0,3fr)_minmax(0,2fr)] gap-3 mt-1">
        <div>
          <div className="text-[11px] text-muted mb-1">Fires on (strongest {s.firings.length} of 7k Pile docs, gate open)</div>
          <div className="flex flex-col gap-1">{s.firings.slice(0, 8).map((f, i) => <Snippet key={i} f={f} sign={sign} />)}</div>
          {s.firings.length === 0 && <div className="text-[12px] text-muted">never fires with this sign and an open gate</div>}
        </div>
        <div>
          <div className="text-[11px] text-muted mb-1">Writes (logit lens of {sign}w)</div>
          <div className="flex flex-wrap gap-1">{s.writes_logit_lens.map((w) => <span key={w.id} title={`score ${fmt(w.score, 4)} · id ${w.id}`}
            className="font-mono text-[11px] border border-line rounded px-1">{showTok(w.text)}</span>)}</div>
          <div className="text-[11px] text-muted mt-2 mb-1">Writes (J-lens of {sign}w): through later layers</div>
          {s.writes_jlens ? <div className="flex flex-wrap gap-1">{s.writes_jlens.map((w) => <span key={w.id} title={`cosine ${fmt(w.cosine, 4)} · id ${w.id}`}
            className="font-mono text-[11px] border rounded px-1" style={{ borderColor: sign === "+" ? "var(--pos)" : "var(--neg)" }}>{showTok(w.text)}</span>)}</div>
            : <div className="text-[11px] text-muted">no J-lens readouts found (run backend/scripts/precompute_jlens_readouts.py)</div>}
          {domains.length > 0 && <>
            <div className="text-[11px] text-muted mt-2">Strong firings by Pile domain (rate above q99.9)</div>
            <Plot height={130} data={[{ type: "bar", orientation: "h", x: domains.map((d) => d[1]), y: domains.map((d) => d[0]),
              marker: { color: sign === "+" ? "#f97316" : "#3b82f6" } }]} layout={{ margin: { l: 110, r: 8, t: 4, b: 20 }, yaxis: { autorange: "reversed" } }} />
          </>}
        </div>
      </div>
    </div>
  );
}

export default function Atlas() {
  const params = useParams<{ slug?: string[] }>();
  const router = useRouter();
  const { backend, prov, setLayer, setNeuron, setPrompt } = useLab();
  const model = prov ? slugOf(prov.model_id) : "";
  const slug = params.slug ?? [];
  const L = slug[1]?.match(/^L(\d+)$/i) ? +slug[1].slice(1) : null;
  const N = slug[2]?.match(/^n(\d+)$/i) ? +slug[2].slice(1) : null;
  const [ov, setOv] = useState<Overview | null>(null);
  const [rows, setRows] = useState<Row[] | null>(null);
  const [sort, setSort] = useState("max_abs");
  const [neuron, setNeuronData] = useState<NeuronResp | null>(null);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<{ layer: number; neuron: number; sign?: string; rank_in_readout?: number }[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [test, setTest] = useState<{ sign: string; r: InterventionResp; f: Firing } | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const hov = useNeuronHover();

  const go = useCallback((l: number | null, n: number | null) => {
    if (!model) return;
    router.push(l === null ? `/atlas/${model}` : n === null ? `/atlas/${model}/L${l}` : `/atlas/${model}/L${l}/n${n}`);
  }, [model, router]);

  useEffect(() => { if (prov) call<Overview>(backend, "/atlas/overview").then(setOv).catch((e) => setErr(e.message)); }, [backend, prov]);
  useEffect(() => { if (L !== null && prov) call<{ rows: Row[] }>(backend, `/atlas/layer/${L}?sort=${sort}&limit=200`).then((r) => setRows(r.rows)).catch((e) => setErr(e.message)); }, [backend, prov, L, sort]);
  useEffect(() => {
    setTest(null);
    if (L !== null && N !== null && prov) call<NeuronResp>(backend, `/atlas/neuron/${L}/${N}`).then((r) => { setNeuronData(r); setErr(null); }).catch((e) => setErr(e.message));
    else setNeuronData(null);
  }, [backend, prov, L, N]);

  const nN = ov?.layers[0]?.fire_rate.length ?? 0, nL = ov?.layers.length ?? 0;
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT" || !nL) return;
      if (e.key === "/") { e.preventDefault(); searchRef.current?.focus(); }
      if (e.key === "r") go(Math.floor(Math.random() * nL), Math.floor(Math.random() * nN));
      if (L !== null && N !== null && e.key === "j") go(L, Math.min(nN - 1, N + 1));
      if (L !== null && N !== null && e.key === "k") go(L, Math.max(0, N - 1));
    };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, [go, L, N, nL, nN]);

  async function search() {
    setErr(null);
    try {
      const r = await call<{ kind: string; results: { layer: number; neuron: number }[] }>(backend, `/atlas/search?q=${encodeURIComponent(q)}`);
      if (r.kind === "id" && r.results[0]) { go(r.results[0].layer, r.results[0].neuron); setHits(null); } else setHits(r.results);
    } catch (e) { setErr((e as Error).message); }
  }

  async function ablate(f: Firing, sign: string) {
    if (L === null || N === null) return;
    try {
      const r = await call<InterventionResp>(backend, "/interventions/run", { token_ids: f.prefix_token_ids, position: -1, top_k: 8,
        interventions: [{ layer: L, neurons: [N], op: "zero", value: 0, positions: null }] });
      setTest({ sign, r, f });
    } catch (e) { setErr((e as Error).message); }
  }

  async function prune() {
    if (L === null || N === null) return;
    try { await call(backend, "/weights/edit", { kind: "prune", layer: L, neurons: [N] }); window.dispatchEvent(new Event("nl-weights-changed")); router.push("/weights"); }
    catch (e) { setErr((e as Error).message); }
  }

  const sel = useMemo(() => ({ L: L ?? -1, n: N }), [L, N]);
  if (!prov) return <div className="p-4"><ErrorBox error="Load a model first (Model Explorer)." /></div>;

  return (
    <div className="h-full overflow-y-auto p-4 max-w-[1400px]">
      <div className="flex items-center gap-3">
        <Link href={`/atlas/${model}`} className="text-xl font-semibold">Neuron Atlas</Link>
        <span className="text-muted num text-[12px]">{prov.model_id}{ov && ` · ${ov.meta.n_docs.toLocaleString()} docs · ${ov.meta.positions_scanned_excl_pos0.toLocaleString()} tokens scanned`}</span>
      </div>
      <div className="flex gap-2 mt-2">
        <input ref={searchRef} className="flex-1" placeholder='search a word (e.g. "John") or an ID like L6 n100   [/]' value={q} onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") search(); }} />
        <Btn onClick={search}>search</Btn>
        <Btn onClick={() => nL && go(Math.floor(Math.random() * nL), Math.floor(Math.random() * nN))}>random [r]</Btn>
      </div>
      <ErrorBox error={err} />
      {hits && <div className="mt-2 text-[12px]"><span className="text-muted">{hits.length} neurons whose ±w logit-lens readout contains it:</span>{" "}
        {hits.map((h, i) => <button key={i} className="num text-accent mr-2" {...hov(h.layer, h.neuron)} onClick={() => go(h.layer, h.neuron)}>L{h.layer}·n{h.neuron}{h.sign} (#{h.rank_in_readout})</button>)}</div>}
      {ov && <div className="mt-3"><LayerStrip ov={ov} selected={sel} onPick={go} />
        <div className="text-[10px] text-muted">Each pixel is one neuron; darker = fires (gate open) on more tokens. Click a pixel for its page, or a layer label for its table.</div></div>}

      {neuron && <div className="mt-4">
        <div className="flex items-center gap-3">
          <h2 className="text-2xl font-semibold num">L{neuron.layer}·n{neuron.neuron}</h2>
          <span className="text-muted num text-[12px]">max |a| {fmt(neuron.max_abs, 3)}</span>
          <span className="text-muted text-[11px]">j / k: next / previous neuron</span>
          <div className="ml-auto flex gap-1">
            <Link href="/microscope" onClick={() => { setLayer(neuron.layer); setNeuron(neuron.neuron); const f = neuron.plus.firings[0] ?? neuron.minus.firings[0]; if (f) setPrompt(f.tokens.join("")); }}
              className="border border-line rounded px-2 text-[12px]">open in Microscope</Link>
            <Link href="/lenses" onClick={() => { setLayer(neuron.layer); setNeuron(neuron.neuron); }} className="border border-line rounded px-2 text-[12px]">Vocabulary Lenses</Link>
            <Link href="/weights" onClick={() => { setLayer(neuron.layer); setNeuron(neuron.neuron); }}
              className="border border-line rounded px-2 text-[12px]" title="Open the Weight Editor with this neuron selected">edit weights…</Link>
            <Btn kind="danger" onClick={prune} title="Weight edit: zero g, u, w of this neuron (revertible)">prune (weight edit)</Btn>
          </div>
        </div>
        <div className="grid grid-cols-1 gap-2 mt-2">
          <SideCard title="+ side (a > 0, adds +w)" sign="+" s={neuron.plus} />
          <SideCard title="− side (a < 0, adds −w)" sign="-" s={neuron.minus} />
        </div>
        <div className="border border-line rounded p-2 mt-2 bg-panel">
          <div className="font-semibold mb-1">J-share and full J-lens readout</div>
          <JNeuronPanel key={`${neuron.layer}-${neuron.neuron}`} backend={backend} layer={neuron.layer} neuron={neuron.neuron} targets={[]} />
        </div>
        <div className="border border-line rounded p-2 mt-2 bg-panel">
          <div className="font-semibold">Test it: does it matter where it fires?</div>
          <div className="text-[12px] text-muted">Re-runs the model on a firing context (the document up to the firing token), zeroes this neuron at every position,
            and shows how the next-token prediction changes. These are real forward passes.</div>
          <div className="flex gap-2 mt-1 flex-wrap">
            {neuron.plus.firings.slice(0, 3).map((f, i) => <Btn key={`p${i}`} onClick={() => ablate(f, "+")}>ablate on + firing {i + 1}</Btn>)}
            {neuron.minus.firings.slice(0, 3).map((f, i) => <Btn key={`m${i}`} onClick={() => ablate(f, "-")}>ablate on − firing {i + 1}</Btn>)}
          </div>
          {test && <div className="grid grid-cols-3 gap-3 mt-2 text-[12px]">
            <div><Snippet f={test.f} sign={test.sign as "+" | "-"} /><KV k="KL(base ‖ ablated)" v={fmt(test.r.comparison.kl_base_to_mod, 5)} />
              <KV k="top-1 changed" v={String(test.r.comparison.top1_changed)} /></div>
            <div><div className="text-muted">largest drops (Δ log p)</div>{test.r.comparison.largest_logprob_decrease.slice(0, 6).map((r) =>
              <div key={r.id} className="num"><span className="font-mono">{showTok(r.text)}</span> {fmt(r.delta_logprob, 3)} <span className="text-muted">({fmt(r.base_prob, 4)}→{fmt(r.mod_prob, 4)})</span></div>)}</div>
            <div><div className="text-muted">largest gains (Δ log p)</div>{test.r.comparison.largest_logprob_increase.slice(0, 6).map((r) =>
              <div key={r.id} className="num"><span className="font-mono">{showTok(r.text)}</span> +{fmt(r.delta_logprob, 3)} <span className="text-muted">({fmt(r.base_prob, 4)}→{fmt(r.mod_prob, 4)})</span></div>)}</div>
          </div>}
          <Learn>A single context is anecdotal. For a controlled test with matched random directions, use the Polysemanticity Lab.</Learn>
        </div>
        <details className="mt-2 text-[11px] text-muted"><summary>How these fields are computed</summary>
          {Object.entries(neuron.notes).map(([k, v]) => <div key={k}><b>{k}</b>: {v}</div>)}</details>
      </div>}

      {L !== null && N === null && rows && <div className="mt-4">
        <div className="flex items-center gap-2"><h2 className="text-xl font-semibold">Layer {L}</h2>
          <span className="text-[12px] text-muted">sort by</span>
          <select value={sort} onChange={(e) => setSort(e.target.value)}>
            <option value="max_abs">max |a|</option><option value="fire_rate">firing rate</option><option value="top_pos">strongest +</option>
            <option value="top_neg">strongest −</option><option value="index">index</option></select></div>
        <table className="w-full text-[12px] num mt-1">
          <thead className="text-muted"><tr><th className="text-left">neuron</th><th className="text-right">max |a|</th><th className="text-right">fires + / −</th>
            <th className="text-left pl-3">writes +w</th><th className="text-left">writes −w</th></tr></thead>
          <tbody>{rows.map((r) => <tr key={r.neuron} className="border-t border-line cursor-pointer hover:bg-panel-2" {...hov(L, r.neuron)} onClick={() => go(L, r.neuron)}>
            <td className="text-accent">n{r.neuron}</td><td className="text-right">{fmt(r.max_abs, 2)}</td>
            <td className="text-right">{fmt(100 * r.fire_rate_pos, 2)}% / {fmt(100 * r.fire_rate_neg, 2)}%</td>
            <td className="font-mono pl-3" style={{ color: "var(--pos)" }}>{r.writes_pos.map(showTok).join(" ")}</td>
            <td className="font-mono" style={{ color: "var(--neg)" }}>{r.writes_neg.map(showTok).join(" ")}</td></tr>)}</tbody>
        </table>
      </div>}
    </div>
  );
}
