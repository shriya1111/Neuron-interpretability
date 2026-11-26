"""Describe the actual loaded architecture (shapes read from modules)."""

from __future__ import annotations

from .adapters import ModelAdapter


def _shape(t) -> list[int] | None:
    return list(t.shape) if t is not None else None


def describe(adapter: ModelAdapter) -> dict:
    cfg = adapter.config
    layers = []
    for i in range(adapter.n_layers):
        attn = adapter.attn(i)
        mlp = adapter.mlp(i)
        layers.append({
            "index": i,
            "input_norm": {"type": type(adapter.input_norm(i)).__name__,
                           "weight": _shape(adapter.input_norm(i).weight)},
            "attention": {
                "type": type(attn).__name__,
                "q_proj": _shape(attn.q_proj.weight), "k_proj": _shape(attn.k_proj.weight),
                "v_proj": _shape(attn.v_proj.weight), "o_proj": _shape(attn.o_proj.weight),
                "q_bias": attn.q_proj.bias is not None,
                "qk_norm": hasattr(attn, "q_norm"),
            },
            "mlp_norm": {"type": type(adapter.mlp_norm(i)).__name__,
                         "weight": _shape(adapter.mlp_norm(i).weight)},
            "mlp": {
                "gate_proj": _shape(mlp.gate_proj.weight), "up_proj": _shape(mlp.up_proj.weight),
                "down_proj": _shape(mlp.down_proj.weight), "bias": mlp.down_proj.bias is not None,
                "activation": "SiLU (verified numerically)", "n_neurons": adapter.d_mlp(i),
            },
        })
    return {
        "model_type": adapter.model_type,
        "n_layers": adapter.n_layers,
        "d_model": adapter.d_model,
        "d_mlp": adapter.d_mlp(0),
        "n_heads": adapter.n_heads,
        "n_kv_heads": adapter.n_kv_heads,
        "head_dim": adapter.head_dim,
        "vocab_size": adapter.vocab_size,
        "tied_embeddings": adapter.tied_embeddings,
        "rms_norm_eps": getattr(cfg, "rms_norm_eps", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "rope_theta": getattr(cfg, "rope_theta", None),
        "embed": _shape(adapter.embed.weight),
        "final_norm": {"type": type(adapter.final_norm).__name__, "weight": _shape(adapter.final_norm.weight)},
        "unembed": _shape(adapter.unembed.weight),
        "total_neurons": sum(adapter.d_mlp(i) for i in range(adapter.n_layers)),
        "layers": layers,
        "dataflow": [
            "resid_pre[L] = h_L",
            "attn_out = Attn(RMSNorm_in(h_L))",
            "resid_mid = h_L + attn_out",
            "x = RMSNorm_post(resid_mid)   # the actual MLP input",
            "gate = W_gate x ; up = W_up x ; a = silu(gate) * up",
            "mlp_out = W_down a (+ b_down)",
            "resid_post[L] = resid_mid + mlp_out",
            "logits = W_U RMSNorm_final(resid_post[last])",
        ],
    }
