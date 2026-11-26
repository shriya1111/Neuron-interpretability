"""Baseline vs. modified forward passes.

The baseline and modified runs are two separate real forward passes. The
modified run's change at each intervened layer is checked against the
analytically expected change sum_i (a'_i - a_i) w_i, which validates that the
intervention touched exactly what it claimed to touch.
"""

from __future__ import annotations

import torch

from evaluation import metrics
from .activation_cache import CaptureSpec, run_forward
from .interventions import InterventionPlan


def compare_runs(handle, input_ids: torch.Tensor, plan: InterventionPlan, position: int = -1,
                 k: int = 10, targets: list[int] | None = None) -> dict:
    adapter = handle.adapter
    tok = handle.tokenizer
    spec = CaptureSpec(layers=None, points=("act", "mlp_out", "resid_post"), final=False)
    base = run_forward(handle, input_ids, spec)
    mod = run_forward(handle, input_ids, spec, plan)
    T = base.input_ids.shape[0]
    t = position + T if position < 0 else position
    if not 0 <= t < T:
        raise ValueError(f"position {position} out of range for {T} tokens")

    layer_checks = []
    intervened = sorted(plan.by_layer())
    for L in intervened:
        down = adapter.mlp(L).down_proj
        W = down.weight.detach().double().cpu()  # [d_model, d_mlp]
        bias = down.bias.detach().double().cpu() if down.bias is not None else 0.0
        a_base = base.get("act", L).double()
        a_pre = mod.get("act", L).double()       # modified run, before the edit
        a_post = mod.get("act_mod", L).double()  # modified run, after the edit
        changed = a_post != a_pre
        expected_delta = (a_post - a_pre) @ W.T  # sum_i (a'_i - a_i) w_i
        observed_delta = mod.get("mlp_out", L).double() - (a_pre @ W.T + bias)
        layer_checks.append({
            "layer": L,
            "n_activations_changed": int(changed.sum()),
            "positions_changed": sorted(set(torch.nonzero(changed)[:, 0].tolist())),
            "upstream_intervened": any(l < L for l in intervened),
            "pre_edit_activation_matches_baseline": bool(torch.allclose(a_pre, a_base, atol=1e-5, rtol=1e-5)),
            "expected_delta_norm_per_position": expected_delta.norm(dim=-1).tolist(),
            "observed_delta_norm_per_position": observed_delta.norm(dim=-1).tolist(),
            "max_abs_error_observed_vs_expected_delta": float((observed_delta - expected_delta).abs().max()),
        })

    resid_delta = []
    for L in range(adapter.n_layers):
        d = mod.get("resid_post", L)[t] - base.get("resid_post", L)[t]
        resid_delta.append({"layer": L, "delta_norm": float(d.norm()),
                            "base_norm": float(base.get("resid_post", L)[t].norm())})

    return {
        "position": t,
        "tokens": metrics.tokens_of(tok, base.input_ids),
        "baseline": {"top_k": metrics.top_k(tok, base.logits[t], k)},
        "modified": {"top_k": metrics.top_k(tok, mod.logits[t], k)},
        "comparison": metrics.compare(tok, base.logits[t], mod.logits[t], k, targets),
        "layer_checks": layer_checks,
        "residual_delta_by_layer": resid_delta,
        "interventions": [iv.to_dict() for iv in plan.neuron_interventions],
        "forward_seconds": {"baseline": base.seconds, "modified": mod.seconds},
    }
