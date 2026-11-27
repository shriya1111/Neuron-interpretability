"use client";

import { useVirtualizer } from "@tanstack/react-virtual";
import { useEffect, useRef, useState } from "react";
import { call, CaptureResp, Tok } from "@/lib/api";
import { useNeuronHover } from "./NeuronHover";
import { color, fmt, showTok } from "./ui";

const PAGE = 256;

/**
 * Virtualized neuron × token grid for one layer. Rows are neurons (all d_mlp of them, loaded
 * in pages of 256 from /activations/capture as you scroll); columns are tokens.
 */
export function NeuronGrid({ backend, prompt, layer, dMlp, selected, onSelect, point = "act" }: {
  backend: string; prompt: string; layer: number; dMlp: number; selected: number; onSelect: (n: number) => void;
  point?: "act" | "gate_pre" | "up";
}) {
  const [pages, setPages] = useState<Record<number, CaptureResp>>({});
  const [tokens, setTokens] = useState<Tok[]>([]);
  const [overview, setOverview] = useState<number[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [jump, setJump] = useState("");
  const loading = useRef<Set<number>>(new Set());
  const parent = useRef<HTMLDivElement>(null);
  const hov = useNeuronHover();

  useEffect(() => { setPages({}); loading.current = new Set(); setOverview(null); }, [backend, prompt, layer, point]);

  const v = useVirtualizer({ count: dMlp, getScrollElement: () => parent.current, estimateSize: () => 16, overscan: 30 });
  const items = v.getVirtualItems();

  useEffect(() => {
    const need = new Set(items.map((it) => Math.floor(it.index / PAGE)));
    need.forEach(async (p) => {
      if (pages[p] || loading.current.has(p)) return;
      loading.current.add(p);
      try {
        const r = await call<CaptureResp>(backend, "/activations/capture",
          { prompt, layer, point, neuron_start: p * PAGE, neuron_count: PAGE, order: "index" });
        setPages((s) => ({ ...s, [p]: r })); setTokens(r.tokens); setOverview(r.layer_col_max_abs); setErr(null);
      } catch (e) { setErr((e as Error).message); }
      loading.current.delete(p);
    });
  }, [items, pages, backend, prompt, layer, point]);

  const maxAbs = overview ? Math.max(...overview) : 1;
  const cell = (n: number, t: number) => {
    const pg = pages[Math.floor(n / PAGE)];
    if (!pg) return null;
    const j = n - pg.neurons[0];
    return { v: pg.values[t][j], open: pg.gate_open[t][j] };
  };
  const topN = overview ? overview.map((x, i) => [x, i] as const).sort((a, b) => b[0] - a[0]).slice(0, 12) : [];

  return (
    <div>
      <div className="flex items-center gap-2 mb-1 text-[11px]">
        <span className="text-muted">{dMlp.toLocaleString()} neurons × {tokens.length} tokens · {point} · color scale ±{fmt(maxAbs, 2)}</span>
        <input placeholder="go to neuron" value={jump} onChange={(e) => setJump(e.target.value)} className="w-24 num"
          onKeyDown={(e) => { const n = parseInt(jump); if (e.key === "Enter" && n >= 0 && n < dMlp) { v.scrollToIndex(n, { align: "center" }); onSelect(n); } }} />
        <span className="text-muted">top max|a|:</span>
        {topN.map(([, i]) => <button key={i} className="num text-accent" {...hov(layer, i)} onClick={() => { v.scrollToIndex(i, { align: "center" }); onSelect(i); }}>n{i}</button>)}
      </div>
      {err && <div className="text-bad text-[11px]">{err}</div>}
      <div className="flex font-mono text-[10px] text-muted pl-14 overflow-hidden">
        {tokens.map((t, i) => <div key={i} className="w-[22px] shrink-0 truncate" title={`${i}: ${JSON.stringify(t.text)}`}>{showTok(t.text)}</div>)}
      </div>
      <div ref={parent} className="h-[320px] overflow-auto border border-line rounded">
        <div style={{ height: v.getTotalSize(), position: "relative" }}>
          {items.map((it) => (
            <div key={it.key} onClick={() => onSelect(it.index)}
              className={`absolute left-0 flex items-center cursor-pointer ${it.index === selected ? "outline outline-1 outline-accent" : ""}`}
              style={{ top: it.start, height: it.size }}>
              <div className="w-14 shrink-0 text-right pr-1 num text-[10px] text-muted hover:text-accent" {...hov(layer, it.index)}>n{it.index}</div>
              {tokens.map((_, t) => {
                const c = cell(it.index, t);
                return <div key={t} className="w-[22px] h-[14px] shrink-0 border-r border-b border-line/30"
                  title={c ? `n${it.index} t${t} ${point}=${fmt(c.v)} gate ${c.open ? "open" : "closed"}` : "loading"}
                  style={{ background: c ? color(c.v, maxAbs) : "transparent", opacity: c && !c.open ? 0.45 : 1 }} />;
              })}
            </div>
          ))}
        </div>
      </div>
      <div className="text-[10px] text-muted mt-1">Warm = positive (adds +w), cool = negative (adds −w). Faded cells: gate closed (g·x ≤ 0) — SiLU leakage.</div>
    </div>
  );
}
