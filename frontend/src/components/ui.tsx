"use client";

import { ReactNode, useState } from "react";
import { Tok, TopK } from "@/lib/api";
import { useLab } from "@/lib/state";

export function fmt(x: number | null | undefined, d = 4): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(2);
  return x.toFixed(d);
}

export function Section({ title, children, right, dense }: { title: string; children: ReactNode; right?: ReactNode; dense?: boolean }) {
  return (
    <section className="border border-line rounded bg-panel">
      <header className="flex items-center justify-between px-2 py-1 border-b border-line bg-panel-2 rounded-t">
        <h3 className="text-[11px] font-semibold uppercase tracking-wide text-muted">{title}</h3>
        {right}
      </header>
      <div className={dense ? "p-1" : "p-2"}>{children}</div>
    </section>
  );
}

export function Learn({ children }: { children: ReactNode }) {
  const { mode } = useLab();
  if (mode !== "learning") return null;
  return <div className="text-[12px] leading-snug text-muted border-l-2 border-accent pl-2 my-1">{children}</div>;
}

export function Expand({ label, children }: { label: string; children: ReactNode }) {
  const [o, setO] = useState(false);
  return (
    <div className="mt-1">
      <button className="text-[11px] text-accent" onClick={() => setO(!o)}>{o ? "▾" : "▸"} {label}</button>
      {o && <div className="mt-1">{children}</div>}
    </div>
  );
}

export function Btn({ children, onClick, disabled, kind = "default", title }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; kind?: "default" | "primary" | "danger"; title?: string;
}) {
  const k = kind === "primary" ? "bg-accent text-white border-accent" : kind === "danger" ? "border-bad text-bad" : "border-line";
  return <button title={title} disabled={disabled} onClick={onClick} className={`px-2 py-0.5 rounded border text-[12px] ${k}`}>{children}</button>;
}

export function ErrorBox({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="text-[12px] text-bad border border-bad rounded p-2 whitespace-pre-wrap">{error}</div>;
}

/** Shows a token with whitespace made visible (· for space, ↵ for newline). */
export function showTok(s: string): string {
  return s.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥") || "∅";
}

export function color(v: number, maxAbs: number): string {
  const m = maxAbs || 1;
  const a = Math.min(1, Math.abs(v) / m);
  return v >= 0 ? `color-mix(in srgb, var(--pos) ${Math.round(a * 85)}%, transparent)`
    : `color-mix(in srgb, var(--neg) ${Math.round(a * 85)}%, transparent)`;
}

export function TokenStrip({ tokens, selected, onSelect, values, gateOpen }: {
  tokens: Tok[]; selected: number; onSelect: (i: number) => void; values?: number[]; gateOpen?: boolean[];
}) {
  const maxAbs = values ? Math.max(1e-9, ...values.map(Math.abs)) : 1;
  return (
    <div className="flex flex-wrap gap-[2px] font-mono text-[12px]">
      {tokens.map((t, i) => (
        <button key={i} onClick={() => onSelect(i)}
          title={`pos ${i} · id ${t.id} · piece ${JSON.stringify(t.piece)} · text ${JSON.stringify(t.text)}` +
            (values ? ` · value ${fmt(values[i])}` : "") + (gateOpen ? ` · gate ${gateOpen[i] ? "open" : "closed"}` : "")}
          className={`px-1 rounded border ${i === selected ? "border-accent ring-1 ring-accent" : "border-line"}`}
          style={values ? { background: color(values[i], maxAbs) } : undefined}>
          {showTok(t.text)}
          {gateOpen && !gateOpen[i] && <sup className="text-muted">⊘</sup>}
        </button>
      ))}
    </div>
  );
}

export function TopKTable({ rows, compareTo }: { rows: TopK[]; compareTo?: TopK[] }) {
  const ref = new Map(compareTo?.map((r) => [r.id, r]) ?? []);
  return (
    <table className="w-full text-[12px] num">
      <thead className="text-muted"><tr><th className="text-left">#</th><th className="text-left">token</th><th className="text-right">id</th>
        <th className="text-right">logit</th><th className="text-right">p</th>{compareTo && <th className="text-right">Δp</th>}</tr></thead>
      <tbody>
        {rows.map((r) => {
          const o = ref.get(r.id);
          return (
            <tr key={r.id} className="border-t border-line">
              <td>{r.rank}</td><td className="font-mono">{showTok(r.text)}</td><td className="text-right text-muted">{r.id}</td>
              <td className="text-right">{fmt(r.logit, 3)}</td><td className="text-right">{fmt(r.prob, 4)}</td>
              {compareTo && <td className="text-right">{o ? fmt(r.prob - o.prob, 4) : "new"}</td>}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function KV({ k, v, title }: { k: string; v: ReactNode; title?: string }) {
  return <div className="flex justify-between gap-2 text-[12px]" title={title}><span className="text-muted">{k}</span><span className="num text-right">{v}</span></div>;
}
