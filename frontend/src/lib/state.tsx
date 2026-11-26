"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { Architecture, call, DEFAULT_BACKEND, Provenance } from "./api";

export type Mode = "learning" | "research";

type Lab = {
  backend: string; setBackend: (s: string) => void;
  arch: Architecture | null; prov: Provenance | null; refreshModel: () => Promise<void>;
  setModel: (a: Architecture, p: Provenance) => void;
  prompt: string; setPrompt: (s: string) => void;
  layer: number; setLayer: (n: number) => void;
  neuron: number; setNeuron: (n: number) => void;
  position: number; setPosition: (n: number) => void;
  mode: Mode; setMode: (m: Mode) => void;
  dark: boolean; setDark: (d: boolean) => void;
  backendError: string | null;
};

const Ctx = createContext<Lab | null>(null);

function usePersisted<T>(key: string, init: T): [T, (v: T) => void] {
  const [v, setV] = useState<T>(init);
  useEffect(() => {
    try { const s = localStorage.getItem(key); if (s !== null) setV(JSON.parse(s)); } catch { /* storage unavailable */ }
  }, [key]);
  const set = useCallback((x: T) => {
    setV(x);
    try { localStorage.setItem(key, JSON.stringify(x)); } catch { /* storage unavailable */ }
  }, [key]);
  return [v, set];
}

export function LabProvider({ children }: { children: React.ReactNode }) {
  const [backend, setBackend] = usePersisted("nl.backend", DEFAULT_BACKEND);
  const [prompt, setPrompt] = usePersisted("nl.prompt", "The fourth Gospel was written by the apostle");
  const [layer, setLayer] = usePersisted("nl.layer", 10);
  const [neuron, setNeuron] = usePersisted("nl.neuron", 0);
  const [position, setPosition] = usePersisted("nl.position", -1);
  const [mode, setMode] = usePersisted<Mode>("nl.mode", "learning");
  const [dark, setDark] = usePersisted("nl.dark", true);
  const [arch, setArch] = useState<Architecture | null>(null);
  const [prov, setProv] = useState<Provenance | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  useEffect(() => { document.documentElement.classList.toggle("dark", dark); }, [dark]);

  const refreshModel = useCallback(async () => {
    try {
      const r = await call<{ architecture: Architecture; provenance: Provenance }>(backend, "/models/config");
      setArch(r.architecture); setProv(r.provenance); setBackendError(null);
    } catch (e) {
      setArch(null); setProv(null); setBackendError((e as Error).message);
    }
  }, [backend]);

  useEffect(() => { refreshModel(); }, [refreshModel]);

  const setModel = useCallback((a: Architecture, p: Provenance) => { setArch(a); setProv(p); setBackendError(null); }, []);

  const value = useMemo<Lab>(() => ({
    backend, setBackend, arch, prov, refreshModel, setModel, prompt, setPrompt, layer, setLayer, neuron, setNeuron,
    position, setPosition, mode, setMode, dark, setDark, backendError,
  }), [backend, setBackend, arch, prov, refreshModel, setModel, prompt, setPrompt, layer, setLayer, neuron, setNeuron,
    position, setPosition, mode, setMode, dark, setDark, backendError]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useLab(): Lab {
  const c = useContext(Ctx);
  if (!c) throw new Error("useLab outside LabProvider");
  return c;
}

export function useDebounced<T>(v: T, ms = 300): T {
  const [d, setD] = useState(v);
  useEffect(() => { const t = setTimeout(() => setD(v), ms); return () => clearTimeout(t); }, [v, ms]);
  return d;
}
