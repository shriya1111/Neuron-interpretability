"use client";

import { useState } from "react";
import { Architecture, call, Provenance } from "@/lib/api";
import { useLab } from "@/lib/state";
import { Btn, ErrorBox, KV, Learn, Section } from "./ui";

const MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B"];

export function ModelPanel() {
  const { backend, arch, prov, setModel } = useLab();
  const [model, setModelId] = useState(MODELS[0]);
  const [dtype, setDtype] = useState("");
  const [device, setDevice] = useState("auto");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function load() {
    setBusy(true); setErr(null);
    try {
      const r = await call<{ architecture: Architecture; provenance: Provenance }>(backend, "/models/load",
        { model_id: model, dtype: dtype || null, device });
      setModel(r.architecture, r.provenance);
    } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  }

  return (
    <Section title="Model">
      <div className="flex flex-col gap-1">
        <input list="models" value={model} onChange={(e) => setModelId(e.target.value)} className="num" />
        <datalist id="models">{MODELS.map((m) => <option key={m} value={m} />)}</datalist>
        <div className="flex gap-1">
          <select value={device} onChange={(e) => setDevice(e.target.value)}>
            {["auto", "cuda", "mps", "cpu"].map((d) => <option key={d}>{d}</option>)}
          </select>
          <select value={dtype} onChange={(e) => setDtype(e.target.value)}>
            <option value="">dtype: default</option>{["float32", "bfloat16", "float16"].map((d) => <option key={d}>{d}</option>)}
          </select>
          <Btn kind="primary" onClick={load} disabled={busy}>{busy ? "loading…" : "load"}</Btn>
        </div>
        <ErrorBox error={err} />
        {arch && prov && (
          <div className="mt-1">
            <KV k="type" v={arch.model_type} />
            <KV k="layers × d_mlp" v={`${arch.n_layers} × ${arch.d_mlp}`} />
            <KV k="d_model / heads" v={`${arch.d_model} / ${arch.n_heads} (kv ${arch.n_kv_heads})`} />
            <KV k="tied embed/unembed" v={String(arch.tied_embeddings)} />
            <KV k="revision" v={String(prov.model_revision ?? "unknown").slice(0, 12)} />
          </div>
        )}
        <Learn>Primary model: Qwen3-0.6B (same family as the paper&apos;s Qwen3-8B/14B). Cross-check: Qwen2.5-0.5B. Dimensions are read
          from the loaded checkpoint. Paper neuron IDs (e.g. Llama L28 n2760) are refused for any other checkpoint.</Learn>
      </div>
    </Section>
  );
}
