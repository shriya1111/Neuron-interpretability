"use client";

import { Background, Controls, Edge, Node, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import { Architecture, InferenceResp } from "@/lib/api";
import { fmt } from "./ui";

/** Vertical dataflow of the real loaded model; blocks are clickable. Values are from the actual forward pass. */
export function ArchitectureGraph({ arch, inf, selectedLayer, onSelectLayer }: {
  arch: Architecture; inf: InferenceResp | null; selectedLayer: number; onSelectLayer: (l: number) => void;
}) {
  const { nodes, edges } = useMemo(() => {
    const t = inf?.position ?? 0;
    const W = 230, gap = 54, colW = 260;
    const nCols = Math.max(1, Math.ceil(arch.n_layers / 12));
    const perCol = Math.ceil(arch.n_layers / nCols);
    let x = 0;
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const mk = (id: string, y: number, label: string, sub: string, extra: Partial<Node> = {}) =>
      nodes.push({ id, position: { x, y }, data: { label: <div className="text-left"><div className="font-medium">{label}</div><div className="text-[10px] opacity-70 num">{sub}</div></div> },
        style: { width: W, fontSize: 11, padding: 4, background: "var(--panel)", color: "var(--fg)", border: "1px solid var(--border)" }, ...extra });
    let y = 0;
    const tok = inf ? inf.tokens.length : 0;
    mk("tok", y, "Tokenization", inf ? `${tok} tokens · selected pos ${t}` : "run a prompt"); y += gap;
    mk("emb", y, "Token embedding", `W_E ${arch.embed.join("×")}`); y += gap;
    const top = y;
    for (let L = 0; L < arch.n_layers; L++) {
      if (L > 0 && L % perCol === 0) { x += colW; y = top; }
      const s = inf?.layers?.[L];
      const sel = L === selectedLayer;
      mk(`b${L}`, y, `Block ${L}`, s
        ? `‖h‖ ${fmt(s.resid_post_norm[t], 1)} · ‖attn‖ ${fmt(s.attn_out_norm[t], 1)} · ‖mlp‖ ${fmt(s.mlp_out_norm[t], 1)} · gate-open ${fmt(100 * s.gate_open_frac[t], 0)}%`
        : `attn ${arch.n_heads}h · MLP ${arch.layers[L].mlp.n_neurons} neurons`,
        { style: { width: W, fontSize: 11, padding: 4, cursor: "pointer", color: "var(--fg)",
          background: sel ? "color-mix(in srgb, var(--accent) 20%, var(--panel))" : "var(--panel)",
          border: `1px solid ${sel ? "var(--accent)" : "var(--border)"}` } });
      y += gap;
    }
    mk("fn", y, "Final norm", `${arch.final_norm.type} γ [${arch.final_norm.weight.join("×")}]`); y += gap;
    mk("ue", y, "Unembedding", `W_U ${arch.unembed.join("×")}${arch.tied_embeddings ? " (tied to W_E)" : ""}`); y += gap;
    mk("pr", y, "Next-token probabilities", inf ? `top: ${JSON.stringify(inf.top_k[0]?.text)} p=${fmt(inf.top_k[0]?.prob, 3)}` : "");
    const ids = nodes.map((n) => n.id);
    for (let i = 0; i + 1 < ids.length; i++) edges.push({ id: `e${i}`, source: ids[i], target: ids[i + 1] });
    return { nodes, edges };
  }, [arch, inf, selectedLayer]);

  return (
    <div style={{ height: 560 }} className="border border-line rounded">
      <ReactFlow nodes={nodes} edges={edges} fitView fitViewOptions={{ padding: 0.05 }} nodesDraggable={false} nodesConnectable={false} minZoom={0.1}
        onNodeClick={(_, n) => { if (n.id.startsWith("b")) onSelectLayer(parseInt(n.id.slice(1))); }}
        proOptions={{ hideAttribution: true }}>
        <Background /><Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
