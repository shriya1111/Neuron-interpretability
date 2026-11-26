"""Polysemanticity Type 1: does one signed neuron direction add the same concept in unrelated contexts?

Definition used by the paper: a neuron *adds* a concept in a context only if zeroing it lowers the
probability of that concept's words. A sign counts as used only when it occurs with the gate open.

Protocol per context (all from real forward passes):
  1. Baseline pass: record the neuron's gate pre-activation and activation at every token.
  2. Zero the neuron at every position (ablation pass) and measure the target token's logit,
     log-probability, probability and rank at the prediction position.
  3. Controls (matched magnitude and positions, after Appendix B check 2): for each of N random
     directions r with ||r|| = ||w||, subtract a_t * r at every position t of the MLP output of the
     same layer — exactly the size of the change zero-ablation makes (it subtracts a_t * w) but in a
     random direction.
  4. Status (thresholds borrowed from Appendix B check 2; the paper gives no separate Type 1 test):
       supported      Δlogp ≤ −0.03 nats and z ≤ −3 versus the controls
       contradicted   Δlogp ≥ 0 (ablation does not lower the target)
       inconclusive   anything else
       insufficient   the neuron never fires with the tested sign and an open gate in the context
"""

from __future__ import annotations

import time

import torch

from evaluation import metrics
from evaluation.controls import random_directions
from evaluation.statistical_tests import empirical_percentile, summarize, z_score
from instrumentation.activation_cache import CaptureSpec, run_forward
from instrumentation.interventions import InterventionPlan, NeuronIntervention, VectorAdd
from interpretability.logit_lens import neuron_readout, token_direction

MIN_DROP_NATS = 0.03
Z_THRESHOLD = 3.0


def _pred_pos(ids: torch.Tensor) -> int:
    return ids.numel() - 1


def _target_stats(logits_row: torch.Tensor, tid: int) -> dict:
    return metrics.token_stats(logits_row, tid)


def run_context(handle, layer: int, neuron: int, sign: str, prompt: str, target_id: int,
                n_controls: int, seed: int) -> dict:
    ad, tok = handle.adapter, handle.tokenizer
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    T = ids.numel()
    t = _pred_pos(ids)
    last = ad.n_layers - 1
    base = run_forward(handle, ids, CaptureSpec(layers=sorted({layer, last}), points=("act", "gate_pre", "resid_post"),
                                                final=False))
    a = base.get("act", layer)[:, neuron]
    g = base.get("gate_pre", layer)[:, neuron]
    sg = 1.0 if sign == "+" else -1.0
    used = [(sg * float(a[i]) > 0) and float(g[i]) > 0 for i in range(T)]

    w = ad.output_vector(layer, neuron).detach().float().cpu()
    u = token_direction(ad, target_id)
    # Direct logit attribution of this neuron at the prediction position, through the unembedding only:
    #   DLA = a_t * (u_target · w) / rms(h_final,t)   where u = W_U[target] ⊙ gamma_final.
    eps = getattr(ad.final_norm, "variance_epsilon", getattr(ad.final_norm, "eps", 1e-6))
    rms_final = float(torch.sqrt(base.get("resid_post", last)[t].double().pow(2).mean() + eps))
    dla = float(a[t]) * float(u @ w) / rms_final

    abl = run_forward(handle, ids, None, InterventionPlan([NeuronIntervention(layer, [neuron], "zero")]))
    b_stats = _target_stats(base.logits[t], target_id)
    a_stats = _target_stats(abl.logits[t], target_id)
    d_logp = a_stats["logprob"] - b_stats["logprob"]

    dirs = random_directions(n_controls, ad.d_model, float(w.norm()), seed)
    ctrl = []
    for r in dirs:
        plan = InterventionPlan(vector_adds=[VectorAdd(layer, r, "mlp_out", None, (-a).tolist(), "random control")])
        c = run_forward(handle, ids, None, plan)
        ctrl.append(_target_stats(c.logits[t], target_id)["logprob"] - b_stats["logprob"])

    z = z_score(d_logp, ctrl)
    if not any(used):
        status = "insufficient_data"
        reason = f"the neuron never activates with sign {sign} and an open gate in this context"
    elif d_logp >= 0:
        status = "contradicted"
        reason = "zeroing the neuron does not lower the target's log-probability"
    elif d_logp <= -MIN_DROP_NATS and z is not None and z <= -Z_THRESHOLD:
        status = "supported"
        reason = f"drop ≥ {MIN_DROP_NATS} nats and z ≤ −{Z_THRESHOLD} versus matched random directions"
    else:
        status = "inconclusive"
        reason = "drop is negative but below the size/z thresholds"

    return {
        "prompt": prompt, "tokens": metrics.tokens_of(tok, ids), "prediction_position": t,
        "target": metrics.token_info(tok, target_id),
        "activation": a.tolist(), "gate": g.tolist(), "sign_used_with_open_gate": used,
        "activation_at_prediction": float(a[t]), "gate_at_prediction": float(g[t]),
        "max_activation_tested_sign": max([sg * float(x) for x in a] + [0.0]) * sg,
        "direct_logit_attribution_at_prediction": dla, "rms_final_at_prediction": rms_final,
        "baseline": b_stats, "ablated": a_stats,
        "delta": {"logprob": d_logp, "logit": a_stats["logit"] - b_stats["logit"],
                  "prob": a_stats["prob"] - b_stats["prob"], "rank": a_stats["rank"] - b_stats["rank"]},
        "baseline_top_k": metrics.top_k(tok, base.logits[t], 8),
        "ablated_top_k": metrics.top_k(tok, abl.logits[t], 8),
        "controls": {"kind": "random directions, ||r|| = ||w||, subtract a_t·r at all positions of layer's MLP output",
                     "n": n_controls, "seed": seed, "delta_logprob": ctrl, "summary": summarize(ctrl),
                     "z": z, "percentile_of_neuron": empirical_percentile(d_logp, ctrl)},
        "status": status, "reason": reason,
    }


def type1(handle, layer: int, neuron: int, sign: str, contexts: list[dict], n_controls: int = 20,
          seed: int = 0, readout_k: int = 15) -> dict:
    ad, tok = handle.adapter, handle.tokenizer
    ad.check_neuron(layer, neuron)
    if sign not in ("+", "-"):
        raise ValueError("sign must be '+' or '-'")
    if len(contexts) < 2:
        raise ValueError("Type 1 needs at least two contexts")
    t0 = time.time()
    results = []
    targets = []
    for i, c in enumerate(contexts):
        tid = _single(tok, c["target"])
        targets.append(tid)
        r = run_context(handle, layer, neuron, sign, c["prompt"], tid, n_controls, seed + 1000 * i)
        r["label"] = c.get("label", f"context {i + 1}")
        results.append(r)
    statuses = [r["status"] for r in results]
    if all(s == "supported" for s in statuses):
        overall = "supported"
    elif any(s == "contradicted" for s in statuses):
        overall = "contradicted"
    elif any(s == "insufficient_data" for s in statuses):
        overall = "insufficient_data"
    else:
        overall = "inconclusive"
    ro = neuron_readout(tok, ad, layer, neuron, readout_k, sorted(set(targets)))
    return {
        "layer": layer, "neuron": neuron, "sign": sign, "contexts": results,
        "readout_logit_lens": ro, "overall": overall,
        "criteria": {"supported": f"every context: Δlogp ≤ −{MIN_DROP_NATS} nats and z ≤ −{Z_THRESHOLD}",
                     "note": "Thresholds borrowed from Appendix B check 2. Whether the contexts are 'unrelated' is "
                             "your judgment and is not tested here."},
        "runtime_seconds": round(time.time() - t0, 3),
    }


def _single(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"target {s!r} is {len(ids)} tokens {[tok.decode([i]) for i in ids]}; "
                         f"Type 1 measures a single next token (mind the leading space)")
    return ids[0]


def type1_candidates(handle, contexts: list[dict], sign: str | None = None, top: int = 25) -> dict:
    """Exploratory: neurons whose direct logit attribution to each context's target is positive in *every*
    context, ranked by the smallest of those attributions. Direct path only (misses effects via later layers).
    This is candidate detection, not confirmation — run ``type1`` on a candidate to test it.
    """
    ad, tok = handle.adapter, handle.tokenizer
    per_ctx = []
    for c in contexts:
        tid = _single(tok, c["target"])
        ids = tok(c["prompt"], return_tensors="pt").input_ids[0]
        t = _pred_pos(ids)
        res = run_forward(handle, ids, CaptureSpec(points=("act", "gate_pre", "resid_post")))
        h = res.get("resid_post", ad.n_layers - 1)[t].double()
        eps = getattr(ad.final_norm, "variance_epsilon", getattr(ad.final_norm, "eps", 1e-6))
        rms = float(torch.sqrt(h.pow(2).mean() + eps))
        u = token_direction(ad, tid)
        dla, act, gate = [], [], []
        for L in range(ad.n_layers):
            s = u @ ad.mlp(L).down_proj.weight.detach().float().cpu()
            a = res.get("act", L)[t]
            dla.append(a * s / rms)
            act.append(a)
            gate.append(res.get("gate_pre", L)[t])
        per_ctx.append((torch.stack(dla), torch.stack(act), torch.stack(gate)))
    dla_all = torch.stack([p[0] for p in per_ctx])  # [C, L, N]
    act_all = torch.stack([p[1] for p in per_ctx])
    gate_all = torch.stack([p[2] for p in per_ctx])
    score = dla_all.min(0).values
    if sign in ("+", "-"):
        sg = 1.0 if sign == "+" else -1.0
        ok = ((sg * act_all > 0) & (gate_all > 0)).all(0)
        score = torch.where(ok, score, torch.full_like(score, -1e9))
    flat = score.flatten().topk(top)
    N = score.shape[1]
    out = []
    for v, idx in zip(flat.values.tolist(), flat.indices.tolist()):
        if v <= 0:
            break
        L, n = divmod(idx, N)
        acts = act_all[:, L, n].tolist()
        out.append({"layer": L, "neuron": n, "min_dla": v, "dla_per_context": dla_all[:, L, n].tolist(),
                    "activation_per_context": acts, "gate_per_context": gate_all[:, L, n].tolist(),
                    "signs": ["+" if x > 0 else "-" for x in acts]})
    return {"candidates": out, "method": "min over contexts of a_t·(u_target·w)/rms(h_final) at the prediction position",
            "status": "exploratory — not experimental confirmation"}
