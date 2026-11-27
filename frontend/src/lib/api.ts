// Typed client for the NeuronLab backend. The backend URL is configurable at
// runtime so the UI can run locally against a remote GPU host.

export const DEFAULT_BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export type Tok = { id: number; text: string; piece: string; position: number };
export type TopK = { id: number; text: string; piece: string; rank: number; logit: number; logprob: number; prob: number };

export type Provenance = Record<string, unknown> & {
  model_id: string; model_revision: string | null; dtype: string; device: string;
  input_token_ids?: number[]; code_version?: { git_sha: string | null; dirty_worktree: boolean | null };
};

export type LayerArch = {
  index: number;
  attention: { q_proj: number[]; k_proj: number[]; v_proj: number[]; o_proj: number[]; q_bias: boolean; qk_norm: boolean };
  mlp: { gate_proj: number[]; up_proj: number[]; down_proj: number[]; bias: boolean; n_neurons: number };
};
export type Architecture = {
  model_type: string; n_layers: number; d_model: number; d_mlp: number; n_heads: number; n_kv_heads: number;
  head_dim: number; vocab_size: number; tied_embeddings: boolean; total_neurons: number; embed: number[];
  unembed: number[]; final_norm: { type: string; weight: number[] }; layers: LayerArch[]; dataflow: string[];
  rms_norm_eps: number | null;
};

export type LayerSummary = {
  layer: number; resid_pre_norm: number[]; attn_out_norm: number[]; mlp_out_norm: number[]; resid_post_norm: number[];
  gate_open_frac: number[]; act_abs_max: number[]; top_neurons_at_position: { neuron: number; activation: number }[];
};
export type InferenceResp = {
  tokens: Tok[]; position: number; top_k: TopK[]; layers?: LayerSummary[]; forward_seconds: number;
  attention?: { layer: number; shape: number[]; pattern: number[][][] }; provenance: Provenance;
};

export type InspectResp = {
  layer: number; neuron: number; position: number; seq_len: number; tokens: Tok[];
  shapes: Record<string, number[]>;
  steps: { gate: number; up: number; sigmoid_gate: number; silu: number; activation: number; gate_open: boolean;
    sign: string; sign_meaning: string; closed_gate_leak: boolean };
  hooked: { gate: number; up: number; activation: number };
  recompute_error: { gate: number; up: number; activation: number };
  mlp_input_vs_raw_residual: Record<string, number | string>;
  contribution: Record<string, number>;
  reconstruction: { formula: string; includes_bias: boolean; max_abs_error: number; max_rel_error: number; per_position_rel_error: number[] };
  trace: { activation: number[]; gate: number[]; up: number[]; gate_open: boolean[]; max_abs_activation: number };
  layer_top_neurons_at_position: { by_abs_activation: { neuron: number; activation: number }[];
    by_contribution_norm: { neuron: number; activation: number; contribution_norm: number }[]; n_gate_open: number; d_mlp: number };
  vectors?: { x: number[]; g: number[]; u: number[]; w: number[]; contribution: number[] };
  provenance: Provenance;
};

export type CaptureResp = {
  layer: number; point: string; shape_full: number[]; tokens: Tok[]; neurons: number[]; values: number[][];
  gate_open: boolean[][]; layer_col_max_abs: number[];
};

export type Op = "zero" | "scale" | "negate" | "set" | "add";
export type Intervention = { layer: number; neurons: number[]; op: Op; value: number; positions: number[] | null; label?: string };

export type DeltaRow = { id: number; text: string; piece: string; base_prob: number; mod_prob: number; delta_prob: number;
  base_logprob: number; mod_logprob: number; delta_logprob: number; delta_logit: number };
export type TokStats = { logit: number; logprob: number; prob: number; rank: number };
export type InterventionResp = {
  position: number; tokens: Tok[]; baseline: { top_k: TopK[] }; modified: { top_k: TopK[] };
  comparison: { kl_base_to_mod: number; top1_changed: boolean; largest_prob_increase: DeltaRow[]; largest_prob_decrease: DeltaRow[];
    largest_logprob_increase: DeltaRow[]; largest_logprob_decrease: DeltaRow[];
    targets: { id: number; text: string; base: TokStats; mod: TokStats }[] };
  layer_checks: { layer: number; n_activations_changed: number; positions_changed: number[]; upstream_intervened: boolean;
    pre_edit_activation_matches_baseline: boolean; max_abs_error_observed_vs_expected_delta: number;
    expected_delta_norm_per_position: number[]; observed_delta_norm_per_position: number[] }[];
  residual_delta_by_layer: { layer: number; delta_norm: number; base_norm: number }[];
  interventions: Intervention[]; hooks_remaining: number; runtime_seconds: number; provenance: Provenance;
};

export class ApiError extends Error {}

/** Access key for hosted deployments: taken from ?key= in the link once, then remembered in this browser. */
export function accessKey(): string {
  if (typeof window === "undefined") return "";
  try {
    const k = new URLSearchParams(window.location.search).get("key");
    if (k) localStorage.setItem("nl.key", k);
    return k || localStorage.getItem("nl.key") || "";
  } catch { return ""; }
}

export async function call<T>(base: string, path: string, body?: unknown): Promise<T> {
  let r: Response;
  try {
    const key = accessKey();
    const headers: Record<string, string> = body === undefined ? {} : { "Content-Type": "application/json" };
    if (key) headers["x-neuronlab-key"] = key;
    r = await fetch(base.replace(/\/$/, "") + path, {
      method: body === undefined ? "GET" : "POST",
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(`cannot reach backend at ${base}`);
  }
  const txt = await r.text();
  let data: unknown;
  try { data = JSON.parse(txt); } catch { data = txt; }
  if (!r.ok) {
    const d = (data as { detail?: unknown })?.detail;
    throw new ApiError(typeof d === "string" ? d : JSON.stringify(d ?? data));
  }
  return data as T;
}
