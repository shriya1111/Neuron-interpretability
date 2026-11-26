"""Logit lens.

Two distinct operations, never to be confused:

1. **Direction readout** (a vector, e.g. a neuron output vector w):
       scores = W_U (gamma_final * v)                      [V]
   gamma_final is the final RMSNorm gain, folded into W_U as in the paper (A.1).
   The final norm's 1/rms(h) factor is a positive scalar shared by all tokens, so it
   does not change rankings; it does change magnitudes (see ``direct_logit_attribution``).
   score_t is the direct (unembedding-only) logit effect per unit of v, before the
   per-token 1/rms scaling. Readout for -v is exactly the negation.

2. **Residual-state lens** (a full residual stream state h_L at a position):
       logits_L = W_U RMSNorm_final(h_L)
   This applies the model's real final norm (including 1/rms and eps) to an intermediate
   state. At the last layer it equals the model's actual prediction.
"""

from __future__ import annotations

import torch

from evaluation import metrics
from instrumentation.activation_cache import CaptureSpec, run_forward

def folded(adapter) -> dict:
    """Return W_U (no copy), gamma_final, and row norms of W_U*diag(gamma) (cached on the adapter)."""
    if getattr(adapter, "_lens_cache", None) is None:
        W = adapter.unembed.weight.detach()
        g = adapter.final_norm.weight.detach().to(W.dtype)
        norms = torch.empty(W.shape[0], dtype=torch.float32, device=W.device)
        for i in range(0, W.shape[0], 16384):  # chunked to bound memory
            norms[i:i + 16384] = (W[i:i + 16384] * g).float().norm(dim=-1)
        adapter._lens_cache = {"W_U": W, "gamma": g, "row_norms": norms}
    return adapter._lens_cache


def direction_scores(adapter, v: torch.Tensor) -> torch.Tensor:
    f = folded(adapter)
    v = v.to(f["W_U"].device, f["W_U"].dtype)
    return (f["W_U"] @ (f["gamma"] * v)).float().cpu()


def token_direction(adapter, tid: int) -> torch.Tensor:
    """u_t = W_U[t] * gamma : the residual direction whose dot product gives token t's folded score."""
    f = folded(adapter)
    return (f["W_U"][tid] * f["gamma"]).float().cpu()


def _rank_of(scores: torch.Tensor, tid: int) -> int:
    return int((scores > scores[tid]).sum()) + 1


def direction_readout(tokenizer, adapter, v: torch.Tensor, k: int = 20, targets: list[int] | None = None) -> dict:
    s = direction_scores(adapter, v)
    cos = s / (folded(adapter)["row_norms"].cpu() * v.float().norm().cpu()).clamp_min(1e-30)

    def side(sign: int) -> dict:
        ss, cc = sign * s, sign * cos
        top = ss.topk(k)
        topc = cc.topk(k)
        return {
            "top_by_score": [dict(metrics.token_info(tokenizer, i), score=float(ss[i]), cosine=float(cc[i]), rank=r + 1)
                             for r, i in enumerate(top.indices.tolist())],
            "top_by_cosine": [dict(metrics.token_info(tokenizer, i), score=float(ss[i]), cosine=float(cc[i]), rank=r + 1)
                              for r, i in enumerate(topc.indices.tolist())],
            "bottom_by_score": [dict(metrics.token_info(tokenizer, i), score=float(ss[i]), cosine=float(cc[i]))
                                for i in (-ss).topk(k).indices.tolist()],
            "targets": [dict(metrics.token_info(tokenizer, t), score=float(ss[t]), cosine=float(cc[t]),
                             rank_by_score=_rank_of(ss, t), rank_by_cosine=_rank_of(cc, t)) for t in (targets or [])],
        }

    return {
        "method": "logit lens on a direction: scores = W_U (gamma_final ⊙ v); cosine uses unit rows of W_U·diag(gamma)",
        "norm_v": float(v.float().norm()),
        "plus": side(+1),
        "minus": side(-1),
        "score_stats": {"mean": float(s.mean()), "std": float(s.std()), "max": float(s.max()), "min": float(s.min())},
    }


def neuron_readout(tokenizer, adapter, layer: int, neuron: int, k: int = 20, targets: list[int] | None = None) -> dict:
    w = adapter.output_vector(layer, neuron).detach()
    out = direction_readout(tokenizer, adapter, w, k, targets)
    out.update({"layer": layer, "neuron": neuron, "vector": "w = down_proj.weight[:, neuron]",
                "note": "Logit lens measures only the direct path through the unembedding. Effects that go through "
                        "later layers are invisible to it (the paper uses the J-lens for that; Phase 3)."})
    return out


def residual_lens(handle, input_ids: torch.Tensor, position: int, k: int = 10, targets: list[int] | None = None) -> dict:
    """Logit lens on full residual states: W_U · RMSNorm_final(h_L[t]) for every layer output."""
    ad, tok = handle.adapter, handle.tokenizer
    res = run_forward(handle, input_ids, CaptureSpec(points=("resid_pre", "resid_post"), final=True))
    T = res.input_ids.shape[0]
    t = position + T if position < 0 else position
    if not 0 <= t < T:
        raise ValueError(f"position {position} out of range for {T} tokens")
    states = [("embed", res.get("resid_pre", 0)[t])] + [(f"L{L}", res.get("resid_post", L)[t]) for L in range(ad.n_layers)]
    W = ad.unembed.weight.detach()
    rows = []
    with torch.no_grad():
        for name, h in states:
            hn = ad.final_norm(h.to(W.device, W.dtype).unsqueeze(0))[0]
            logits = (W @ hn).float().cpu()
            rows.append({"state": name, "top_k": metrics.top_k(tok, logits, k),
                         "targets": [dict(metrics.token_info(tok, x), **metrics.token_stats(logits, x)) for x in (targets or [])],
                         "resid_norm": float(h.norm())})
    final_err = float((torch.tensor([r["logit"] for r in rows[-1]["top_k"]]) -
                       res.logits[t][[r["id"] for r in rows[-1]["top_k"]]]).abs().max())
    return {
        "method": "logits_L = W_U · RMSNorm_final(h_L) using the model's own final norm (incl. 1/rms and eps)",
        "position": t, "tokens": metrics.tokens_of(tok, res.input_ids), "layers": rows,
        "model_prediction": metrics.top_k(tok, res.logits[t], k),
        "last_layer_matches_model_max_abs_logit_error": final_err,
        "note": "Only the last row is the model's prediction. Earlier rows ask what the unembedding would say if the "
                "remaining layers were skipped; they are often incoherent in early layers.",
    }


def neuron_search(tokenizer, adapter, tid: int, sign: str = "+", top: int = 30, layers: list[int] | None = None) -> dict:
    """Rank all neurons by how strongly their signed output vector promotes token tid under the logit lens.

    Exact over all neurons for the score u_t·w; the full-vocabulary rank of tid in each top candidate's
    readout is then computed exactly.
    """
    if sign not in ("+", "-"):
        raise ValueError("sign must be '+' or '-'")
    sg = 1.0 if sign == "+" else -1.0
    u = token_direction(adapter, tid)
    rn = float(folded(adapter)["row_norms"][tid])
    cands = []
    for L in (layers if layers is not None else range(adapter.n_layers)):
        Wd = adapter.mlp(L).down_proj.weight.detach().float().cpu()  # [d, d_mlp]
        s = sg * (u @ Wd)
        cos = s / (Wd.norm(dim=0) * rn).clamp_min(1e-30)
        k = min(top, s.numel())
        for i in cos.topk(k).indices.tolist():
            cands.append((float(cos[i]), float(s[i]), L, i))
    cands.sort(reverse=True)
    cands = cands[:top]
    out = []
    for cos_v, s_v, L, i in cands:
        w = sg * adapter.output_vector(L, i).detach()
        scores = direction_scores(adapter, w)
        out.append({"layer": L, "neuron": i, "sign": sign, "cosine": cos_v, "score": s_v,
                    "rank_of_target": _rank_of(scores, tid),
                    "readout_top5": [metrics.token_info(tokenizer, j)["text"] for j in scores.topk(5).indices.tolist()]})
    return {"target": metrics.token_info(tokenizer, tid), "sign": sign, "candidates": out,
            "method": "cosine(u_t, sign·w) over all neurons, u_t = W_U[t] ⊙ gamma_final; exact vocab rank for each candidate"}
