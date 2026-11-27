"""Linear predictions vs. real interventions.

Inject eps * v_hat into the residual stream at the output of block `layer`, at one position t, and measure the
target token's logit at position t. Three quantities are reported for each eps:

  observed        z_y(eps) - z_y(0)                       real forward passes (VectorAdd at resid_post)
  context_jvp     eps * dz_y/deps |_{eps=0}                exact directional derivative for THIS prompt (autograd)
  corpus_jlens    eps * u_y^T P J_l v_hat / rms(h_final)   the fitted, corpus-averaged J-lens, with the final RMSNorm
                                                           linearized at the real final state:
                                                           d(norm)/dh = diag(gamma)/rms · (I - h h^T / (d·rms^2)),
                                                           so u_y = W_U[y], P = I - h h^T/(d rms^2) and gamma folded in.

The corpus J-lens is an average over contexts and over current-and-future target positions; it is NOT this prompt's
Jacobian, so its prediction is expected to be less accurate than context_jvp. Both are first-order, so both fail for
large eps. eps is expressed as a multiple of ||h_l[t]|| (the residual norm at the injection point).
"""

from __future__ import annotations

import torch

from instrumentation.activation_cache import CaptureSpec, run_forward
from instrumentation.hooks import hooks
from instrumentation.interventions import InterventionPlan, VectorAdd


def _target_logit_with_scalar(handle, ids: torch.Tensor, layer: int, t: int, v: torch.Tensor, y: int) -> float:
    """dz_y/ds at s=0 for h_l[t] += s·v, by autograd."""
    ad = handle.adapter
    p0 = next(handle.model.parameters())
    s = torch.zeros((), dtype=p0.dtype, device=handle.device, requires_grad=True)
    vv = v.to(p0.device, p0.dtype)
    with handle.lock, torch.enable_grad(), hooks() as hs:
        def add(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            d = torch.zeros_like(o)
            d[0, t] = s * vv
            o2 = o + d
            return (o2,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else o2
        hs.forward(ad.block(layer), add)
        logits = handle.model(input_ids=ids.unsqueeze(0).to(handle.device), use_cache=False).logits[0, t, y]
        (g,) = torch.autograd.grad(logits, s)
    return float(g)


def validate(handle, lens, ids: torch.Tensor, layer: int, v: torch.Tensor, target: int, position: int = -1,
             eps_rel: list[float] | None = None) -> dict:
    ad = handle.adapter
    T = ids.numel()
    t = position + T if position < 0 else position
    if not 0 <= t < T:
        raise ValueError("position out of range")
    ad.check_layer(layer)
    eps_rel = eps_rel or [-2.0, -1.0, -0.5, -0.25, -0.1, -0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05, 0.1,
                          0.25, 0.5, 1.0, 2.0]
    v = v.detach().float().cpu()
    v = v / v.norm().clamp_min(1e-12)
    last = ad.n_layers - 1
    base = run_forward(handle, ids, CaptureSpec(layers=sorted({layer, last}), points=("resid_post",), final=False))
    h_l = base.get("resid_post", layer)[t].double()
    h_f = base.get("resid_post", last)[t].double()
    z0 = float(base.logits[t, target])
    scale = float(h_l.norm())

    # context-specific derivative (exact, this prompt)
    dz = _target_logit_with_scalar(handle, ids, layer, t, v, target)

    # corpus-averaged J-lens prediction
    d = h_f.numel()
    eps_n = getattr(ad.final_norm, "variance_epsilon", getattr(ad.final_norm, "eps", 1e-6))
    rms = float(torch.sqrt(h_f.pow(2).mean() + eps_n))
    u = (ad.unembed.weight[target].detach().double() * ad.final_norm.weight.detach().double()).cpu()
    Jv = v.double() if layer == last else lens.jacobians[layer].double() @ v.double()
    Pu = u - h_f * (h_f @ u) / (d * rms ** 2)
    dz_lens = float(Pu @ Jv) / rms

    rows = []
    for e in eps_rel:
        eps = e * scale
        plan = InterventionPlan(vector_adds=[VectorAdd(layer, v * eps, "resid_post", [t], 1.0, "validation")])
        z = float(run_forward(handle, ids, None, plan).logits[t, target])
        rows.append({"eps_rel": e, "eps": eps, "observed": z - z0, "context_jvp": eps * dz, "corpus_jlens": eps * dz_lens})

    def rel_err(key, small_only):
        sel = [r for r in rows if (abs(r["eps_rel"]) <= 0.1 if small_only else True)]
        num = sum((r[key] - r["observed"]) ** 2 for r in sel) ** 0.5
        den = sum(r["observed"] ** 2 for r in sel) ** 0.5
        return num / den if den > 0 else None

    return {
        "layer": layer, "position": t, "target_id": target, "baseline_logit": z0, "residual_norm_at_injection": scale,
        "dz_deps_context": dz, "dz_deps_corpus_jlens": dz_lens,
        "slope_ratio_lens_over_context": dz_lens / dz if dz != 0 else None,
        "rows": rows,
        "relative_error": {"context_jvp_small_eps": rel_err("context_jvp", True), "context_jvp_all": rel_err("context_jvp", False),
                           "corpus_jlens_small_eps": rel_err("corpus_jlens", True), "corpus_jlens_all": rel_err("corpus_jlens", False)},
        "notes": {"injection": "eps·v̂ added to the output of block `layer` at one position; logit read at the same position",
                  "context_jvp": "exact first-order prediction for this prompt (autograd)",
                  "corpus_jlens": "fitted corpus-mean J_l; averages over contexts and future target positions, so it is "
                                  "an approximation for any single prompt",
                  "small_eps": "|eps| ≤ 0.1·||h_l[t]||",
                  "scale": "the reference estimator SUMS effects over target positions t' >= t before averaging over source "
                           "positions, so corpus_jlens magnitudes are not same-position predictions; its direction/sign "
                           "is what the readouts use (cosines are scale-free)"},
    }
