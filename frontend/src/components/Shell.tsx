"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ReactNode, useEffect, useState } from "react";
import { call } from "@/lib/api";
import { useLab } from "@/lib/state";

// status: built | partial | planned (badge shows the phase that completes it)
export const NAV = [
  { href: "/atlas", label: "Neuron Atlas", phase: 8, status: "built" },
  { href: "/explorer", label: "Model Explorer", phase: 1, status: "built" },
  { href: "/microscope", label: "Neuron Microscope", phase: 1, status: "built" },
  { href: "/lenses", label: "Vocabulary Lenses", phase: 3, status: "built" },
  { href: "/polysemanticity", label: "Polysemanticity Lab", phase: 4, status: "built" },
  { href: "/weights", label: "Weight Editor", phase: 8, status: "built" },
  { href: "/steering", label: "Cross-Layer Steering", phase: 5, status: "planned" },
  { href: "/recall", label: "Factual Recall Lab", phase: 6, status: "planned" },
  { href: "/custom", label: "Custom Experiments", phase: 7, status: "planned" },
  { href: "/history", label: "Experiment History", phase: 7, status: "planned" },
];

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const { arch, prov, mode, setMode, dark, setDark, backend, setBackend, backendError, refreshModel } = useLab();
  const [editing, setEditing] = useState(false);
  const [url, setUrl] = useState(backend);
  const [health, setHealth] = useState<string | null>(null);
  const [nEdits, setNEdits] = useState(0);

  useEffect(() => {
    const check = () => call<{ edits: unknown[] }>(backend, "/weights/edits").then((r) => setNEdits(r.edits.length)).catch(() => setNEdits(0));
    check();
    window.addEventListener("nl-weights-changed", check);
    return () => window.removeEventListener("nl-weights-changed", check);
  }, [backend, path, prov]);

  return (
    <div className="h-screen flex flex-col">
      <header className="flex items-center gap-3 px-3 h-9 border-b border-line bg-panel shrink-0">
        <span className="font-semibold tracking-tight">NeuronLab</span>
        {nEdits > 0 && <Link href="/weights" className="text-[11px] font-semibold px-2 rounded bg-bad text-white"
          title="The loaded model's weights are edited. Every result reflects the edited model until you revert.">
          WEIGHTS MODIFIED · {nEdits} edit{nEdits > 1 ? "s" : ""}</Link>}
        <span className="text-muted text-[11px] num">
          {prov ? `${prov.model_id} · ${prov.dtype} · ${prov.device} · rev ${String(prov.model_revision ?? "?").slice(0, 8)}` : "no model loaded"}
        </span>
        <div className="ml-auto flex items-center gap-2 text-[12px]">
          {editing ? (
            <>
              <input className="w-64 num" value={url} onChange={(e) => setUrl(e.target.value)} />
              <button className="text-accent" onClick={async () => {
                setBackend(url); setEditing(false);
                try { const h = await call<{ ok: boolean; cuda: boolean }>(url, "/health"); setHealth(h.cuda ? "cuda" : "no cuda"); }
                catch (e) { setHealth((e as Error).message); }
              }}>save</button>
            </>
          ) : (
            <button className="num text-muted" title="backend URL (click to change)" onClick={() => { setUrl(backend); setEditing(true); }}>
              {backend} {backendError ? <span className="text-bad">● offline</span> : <span className="text-ok">●</span>} {health && <span className="text-muted">{health}</span>}
            </button>
          )}
          <button className="text-muted" onClick={() => refreshModel()} title="refresh model status">↻</button>
          <div className="flex border border-line rounded overflow-hidden">
            {(["learning", "research"] as const).map((m) => (
              <button key={m} onClick={() => setMode(m)} className={`px-2 ${mode === m ? "bg-accent text-white" : ""}`}>{m}</button>
            ))}
          </div>
          <button className="border border-line rounded px-2" onClick={() => setDark(!dark)}>{dark ? "light" : "dark"}</button>
        </div>
      </header>
      {backendError?.includes("access key") && <div className="px-3 py-2 bg-bad text-white text-[13px]">
        This NeuronLab deployment needs an access key. Open the full link you were given (it ends in <span className="num">?key=…</span>);
        the key is then remembered in this browser.</div>}
      <div className="flex flex-1 min-h-0">
        <nav className="w-44 shrink-0 border-r border-line bg-panel p-1 flex flex-col gap-[1px]">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href}
              className={`px-2 py-1 rounded text-[12px] flex justify-between ${path?.startsWith(n.href) ? "bg-panel-2 text-accent font-medium" : "text-fg"}`}>
              <span>{n.label}</span>{n.status !== "built" && <span className="text-[10px] text-muted" title={n.status === "partial" ? `partly built; completed in Phase ${n.phase}` : `planned for Phase ${n.phase}`}>{n.status === "partial" ? "◐" : ""}P{n.phase}</span>}
            </Link>
          ))}
          <div className="mt-auto text-[10px] text-muted p-1 leading-tight">
            {arch ? `${arch.n_layers} layers · d_model ${arch.d_model} · d_mlp ${arch.d_mlp} · ${arch.total_neurons.toLocaleString()} neurons · vocab ${arch.vocab_size.toLocaleString()}` : ""}
          </div>
        </nav>
        <main className="flex-1 min-w-0 min-h-0">{children}</main>
      </div>
    </div>
  );
}

/** Three-panel workspace: left = controls, center = visualizations, right = numbers. */
export function Workspace({ left, center, right }: { left: ReactNode; center: ReactNode; right: ReactNode }) {
  return (
    <div className="h-full grid grid-cols-[300px_minmax(0,1fr)_380px]">
      <aside className="border-r border-line overflow-y-auto p-2 flex flex-col gap-2">{left}</aside>
      <div className="overflow-y-auto p-2 flex flex-col gap-2">{center}</div>
      <aside className="border-l border-line overflow-y-auto p-2 flex flex-col gap-2">{right}</aside>
    </div>
  );
}
