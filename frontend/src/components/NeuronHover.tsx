"use client";

import { createContext, MouseEvent, ReactNode, useCallback, useContext, useEffect, useRef, useState } from "react";
import { call } from "@/lib/api";
import { useLab } from "@/lib/state";
import { fmt, showTok } from "./ui";

type Firing = { tokens: string[]; peak_index: number; activation: number; domain: string } | null;
type Side = { fire_rate: number; logit_lens: string[]; jlens?: string[]; j_share?: { value: number; class: string }; top_firing: Firing };
type Summary = { layer: number; neuron: number; max_abs: number; sign_used: string | null; plus: Side; minus: Side };

type Ctx = { show: (layer: number, neuron: number, e: { clientX: number; clientY: number }) => void; hide: () => void };
const HoverCtx = createContext<Ctx>({ show: () => {}, hide: () => {} });

const cache = new Map<string, Summary | string>();
const CLASS_COLOR: Record<string, string> = { high: "var(--ok)", middle: "var(--warn)", low: "var(--bad)" };

/** Renders one floating neuron card next to the cursor; any component can request it via useNeuronHover(). */
export function NeuronHoverProvider({ children }: { children: ReactNode }) {
  const { backend, prov } = useLab();
  const [target, setTarget] = useState<{ layer: number; neuron: number; x: number; y: number } | null>(null);
  const [data, setData] = useState<Summary | string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const current = useRef<string>("");

  const show = useCallback((layer: number, neuron: number, e: { clientX: number; clientY: number }) => {
    const key = `${layer}-${neuron}`;
    setTarget({ layer, neuron, x: e.clientX, y: e.clientY });
    if (current.current === key) return;
    current.current = key;
    if (timer.current) clearTimeout(timer.current);
    if (cache.has(key)) { setData(cache.get(key)!); return; }
    setData(null);
    timer.current = setTimeout(async () => {
      try {
        const s = await call<Summary>(backend, `/atlas/summary/${layer}/${neuron}`);
        cache.set(key, s);
        if (current.current === key) setData(s);
      } catch (err) {
        const msg = (err as Error).message;
        if (current.current === key) setData(msg);
      }
    }, 120);
  }, [backend]);

  const hide = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    current.current = "";
    setTarget(null);
  }, []);

  useEffect(() => { cache.clear(); }, [prov?.model_id]);

  return (
    <HoverCtx.Provider value={{ show, hide }}>
      {children}
      {target && <Card target={target} data={data} />}
    </HoverCtx.Provider>
  );
}

export function useNeuronHover() {
  const { show, hide } = useContext(HoverCtx);
  /** Spread onto any element: shows the card on hover, follows the cursor, hides on leave. Clicks are untouched. */
  return (layer: number, neuron: number) => ({
    onMouseEnter: (e: MouseEvent) => show(layer, neuron, e),
    onMouseMove: (e: MouseEvent) => show(layer, neuron, e),
    onMouseLeave: hide,
  });
}

export function useNeuronHoverRaw() { return useContext(HoverCtx); }

function SideBlock({ label, s, color }: { label: string; s: Side; color: string }) {
  const f = s.top_firing;
  return (
    <div className="mt-1.5">
      <div className="flex justify-between text-[11px]">
        <b style={{ color }}>{label}</b>
        <span className="num text-muted">fires {fmt(100 * s.fire_rate, 2)}%{s.j_share && <> · J-share <b style={{ color: CLASS_COLOR[s.j_share.class] }}>{fmt(s.j_share.value, 3)} {s.j_share.class}</b></>}</span>
      </div>
      {s.jlens && <div className="text-[11px]"><span className="text-muted">J-lens: </span><span className="font-mono">{s.jlens.map(showTok).join("  ")}</span></div>}
      <div className="text-[11px]"><span className="text-muted">logit lens: </span><span className="font-mono">{s.logit_lens.map(showTok).join("  ")}</span></div>
      {f ? <div className="font-mono text-[10.5px] mt-0.5 bg-panel-2 rounded px-1 py-0.5 leading-4">
        {f.tokens.map((t, i) => <span key={i} style={i === f.peak_index ? { background: `color-mix(in srgb, ${color} 45%, transparent)`, fontWeight: 600 } : undefined}>{t.replace(/\n/g, "↵")}</span>)}
        <span className="text-muted num"> {fmt(f.activation, 2)} · {f.domain}</span>
      </div> : <div className="text-[11px] text-muted">no open-gate firings of this sign</div>}
    </div>
  );
}

function Card({ target, data }: { target: { layer: number; neuron: number; x: number; y: number }; data: Summary | string | null }) {
  const W = 420;
  const vw = typeof window !== "undefined" ? window.innerWidth : 1600;
  const vh = typeof window !== "undefined" ? window.innerHeight : 1000;
  const left = target.x + 16 + W > vw ? Math.max(8, target.x - 16 - W) : target.x + 16;
  const top = Math.min(target.y + 14, vh - 300);
  return (
    <div className="fixed z-50 pointer-events-none rounded border border-line bg-panel shadow-xl p-2 text-fg" style={{ left, top, width: W }}>
      <div className="flex items-baseline justify-between">
        <span className="font-semibold num">L{target.layer}·n{target.neuron}</span>
        {typeof data === "object" && data && <span className="text-[11px] text-muted num">max |a| {fmt(data.max_abs, 2)}{data.sign_used && <> · uses {data.sign_used}</>}</span>}
      </div>
      {data === null && <div className="text-[11px] text-muted">loading…</div>}
      {typeof data === "string" && <div className="text-[11px] text-muted">{data.includes("no atlas") ? "no Atlas data for this model" : data}</div>}
      {typeof data === "object" && data && <>
        <SideBlock label="+ side (adds +w)" s={data.plus} color="var(--pos)" />
        <SideBlock label="− side (adds −w)" s={data.minus} color="var(--neg)" />
        <div className="text-[10px] text-muted mt-1">click for the full page · readouts are hypotheses, not proof</div>
      </>}
    </div>
  );
}
