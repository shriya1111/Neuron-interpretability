"""Single-neuron decomposition and MLP reconstruction checks.

All "recomputed" quantities are computed in float64 from the captured MLP input
``x`` and the neuron's weights, then compared with the values the hooks saw in
the real forward pass. Disagreement indicates an instrumentation bug.
"""

from __future__ import annotations

import torch

from .activation_cache import CaptureSpec, run_forward


def _f(t) -> float:
    return float(t)


def _vec(t: torch.Tensor) -> list[float]:
    return t.to(torch.float32).tolist()


def reconstruction(adapter, layer: int, act: torch.Tensor, mlp_out: torch.Tensor) -> dict:
    """Check MLP_out == sum_i a_i w_i (+ b) at every position.

    act: [T, d_mlp], mlp_out: [T, d_model] captured from the forward pass.
    """
    down = adapter.mlp(layer).down_proj
    W = down.weight.detach().double().cpu()  # [d_model, d_mlp]
    recon = act.double() @ W.T
    if down.bias is not None:
        recon = recon + down.bias.detach().double().cpu()
    diff = recon - mlp_out.double()
    per_pos_abs = diff.abs().max(-1).values
    per_pos_rel = diff.norm(dim=-1) / mlp_out.double().norm(dim=-1).clamp_min(1e-30)
    return {
        "formula": "MLP_out[t] = sum_i a_i[t] * w_i" + (" + b_down" if down.bias is not None else ""),
        "includes_bias": down.bias is not None,
        "max_abs_error": _f(per_pos_abs.max()),
        "max_rel_error": _f(per_pos_rel.max()),
        "per_position_rel_error": per_pos_rel.tolist(),
        "computed_in": "float64 from captured activations vs. hooked MLP output",
    }


def inspect_neuron(handle, input_ids: torch.Tensor, layer: int, neuron: int, position: int,
                   top_n: int = 20, include_weights: bool = True) -> dict:
    adapter = handle.adapter
    adapter.check_neuron(layer, neuron)
    res = run_forward(handle, input_ids, CaptureSpec(layers=[layer], final=False))
    T = res.input_ids.shape[0]
    t = position + T if position < 0 else position
    if not 0 <= t < T:
        raise ValueError(f"position {position} out of range for {T} tokens")

    mlp = adapter.mlp(layer)
    g = mlp.gate_proj.weight[neuron].detach().double().cpu()
    u = mlp.up_proj.weight[neuron].detach().double().cpu()
    w = mlp.down_proj.weight[:, neuron].detach().double().cpu()
    gb = mlp.gate_proj.bias[neuron].double().cpu() if mlp.gate_proj.bias is not None else torch.tensor(0.0, dtype=torch.float64)
    ub = mlp.up_proj.bias[neuron].double().cpu() if mlp.up_proj.bias is not None else torch.tensor(0.0, dtype=torch.float64)

    x_all = res.get("mlp_in", layer).double()  # [T, d_model]
    x = x_all[t]
    gate = g @ x + gb
    up = u @ x + ub
    sig = torch.sigmoid(gate)
    silu = gate * sig
    act = silu * up
    contrib = act * w

    hooked_gate = res.get("gate_pre", layer)[:, neuron]
    hooked_up = res.get("up", layer)[:, neuron]
    hooked_act = res.get("act", layer)[:, neuron]
    mlp_out = res.get("mlp_out", layer)
    resid_mid = res.get("resid_mid", layer)
    resid_post = res.get("resid_post", layer)

    # Raw residual vs actual MLP input: they differ by RMSNorm (scale + learned gain).
    norm = adapter.mlp_norm(layer)
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    rm = resid_mid[t].double()
    rms = torch.sqrt(rm.pow(2).mean() + eps)
    gamma = norm.weight.detach().double().cpu()
    x_from_resid = gamma * rm / rms

    mo = mlp_out[t].double()
    cn = contrib.norm()

    def cos(a, b):
        return _f(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))

    # Whole-token trace of this neuron (all positions).
    trace_gate = hooked_gate.tolist()
    trace_act = hooked_act.tolist()
    acts_t = res.get("act", layer)[t]
    top_abs = acts_t.abs().topk(min(top_n, acts_t.numel()))
    wnorms = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    contrib_norms = (acts_t.abs() * wnorms)
    top_contrib = contrib_norms.topk(min(top_n, acts_t.numel()))

    out = {
        "layer": layer, "neuron": neuron, "position": t, "seq_len": T,
        "shapes": {"x": [adapter.d_model], "g": [adapter.d_model], "u": [adapter.d_model],
                   "w": [adapter.d_model], "W_gate": list(mlp.gate_proj.weight.shape),
                   "W_up": list(mlp.up_proj.weight.shape), "W_down": list(mlp.down_proj.weight.shape)},
        "steps": {
            "gate": _f(gate), "up": _f(up), "sigmoid_gate": _f(sig), "silu": _f(silu),
            "activation": _f(act), "gate_open": bool(gate > 0),
            "sign": "+" if act > 0 else ("-" if act < 0 else "0"),
            "sign_meaning": ("adds +w" if act > 0 else "adds -w" if act < 0 else "adds nothing"),
            "closed_gate_leak": bool(gate <= 0 and act != 0),
        },
        "hooked": {"gate": _f(hooked_gate[t]), "up": _f(hooked_up[t]), "activation": _f(hooked_act[t])},
        "recompute_error": {
            "gate": abs(_f(gate) - _f(hooked_gate[t])),
            "up": abs(_f(up) - _f(hooked_up[t])),
            "activation": abs(_f(act) - _f(hooked_act[t])),
        },
        "mlp_input_vs_raw_residual": {
            "note": "x is RMSNorm_post(resid_mid) = gamma * resid_mid / rms(resid_mid); it is not the raw residual.",
            "rms_resid_mid": _f(rms), "gamma_mean": _f(gamma.mean()), "gamma_std": _f(gamma.std()),
            "cos_x_resid_mid": cos(x, rm),
            "norm_x": _f(x.norm()), "norm_resid_mid": _f(rm.norm()),
            "max_abs_error_recomputed_norm": _f((x_from_resid - x).abs().max()),
        },
        "contribution": {
            "norm": _f(cn), "w_norm": _f(w.norm()),
            "cos_with_mlp_out": cos(contrib, mo) if cn > 0 else 0.0,
            "fraction_of_mlp_out_along_contribution": _f((contrib @ mo) / (mo @ mo).clamp_min(1e-30)),
            "norm_mlp_out": _f(mo.norm()), "norm_resid_post": _f(resid_post[t].double().norm()),
            "cos_w_with_resid_post": cos(w, resid_post[t].double()),
        },
        "reconstruction": reconstruction(adapter, layer, res.get("act", layer), mlp_out),
        "trace": {
            "activation": trace_act, "gate": trace_gate, "up": hooked_up.tolist(),
            "gate_open": [v > 0 for v in trace_gate],
            "max_abs_activation": max(abs(v) for v in trace_act),
        },
        "layer_top_neurons_at_position": {
            "by_abs_activation": [{"neuron": int(i), "activation": _f(acts_t[i])} for i in top_abs.indices],
            "by_contribution_norm": [{"neuron": int(i), "activation": _f(acts_t[i]), "contribution_norm": _f(contrib_norms[i])}
                                     for i in top_contrib.indices],
            "n_gate_open": int((res.get("gate_pre", layer)[t] > 0).sum()),
            "d_mlp": int(acts_t.numel()),
        },
        "forward_seconds": res.seconds,
    }
    if include_weights:
        out["vectors"] = {"x": _vec(x), "g": _vec(g), "u": _vec(u), "w": _vec(w), "contribution": _vec(contrib)}
    return out
