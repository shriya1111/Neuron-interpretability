"""Prediction readouts and baseline-vs-modified comparisons."""

from __future__ import annotations

import torch


def token_info(tokenizer, tid: int) -> dict:
    return {
        "id": int(tid),
        "text": tokenizer.decode([int(tid)]),  # exact decoded string incl. leading space
        "piece": tokenizer.convert_ids_to_tokens(int(tid)),
    }


def tokens_of(tokenizer, ids) -> list[dict]:
    return [dict(token_info(tokenizer, t), position=i) for i, t in enumerate(ids.tolist())]


def top_k(tokenizer, logits_row: torch.Tensor, k: int = 10) -> list[dict]:
    lp = torch.log_softmax(logits_row.double(), -1)
    vals, idx = lp.topk(k)
    return [dict(token_info(tokenizer, i), rank=r + 1, logit=float(logits_row[i]),
                 logprob=float(v), prob=float(v.exp()))
            for r, (v, i) in enumerate(zip(vals, idx.tolist()))]


def token_stats(logits_row: torch.Tensor, tid: int) -> dict:
    lp = torch.log_softmax(logits_row.double(), -1)
    rank = int((logits_row > logits_row[tid]).sum().item()) + 1
    return {"logit": float(logits_row[tid]), "logprob": float(lp[tid]), "prob": float(lp[tid].exp()), "rank": rank}


def compare(tokenizer, base_row: torch.Tensor, mod_row: torch.Tensor, k: int = 10,
            targets: list[int] | None = None) -> dict:
    lp_b = torch.log_softmax(base_row.double(), -1)
    lp_m = torch.log_softmax(mod_row.double(), -1)
    p_b, p_m = lp_b.exp(), lp_m.exp()
    kl = float((p_b * (lp_b - lp_m)).sum())
    d_lp = lp_m - lp_b
    d_p = p_m - p_b

    def rows(idx):
        return [dict(token_info(tokenizer, i), base_prob=float(p_b[i]), mod_prob=float(p_m[i]),
                     delta_prob=float(d_p[i]), base_logprob=float(lp_b[i]), mod_logprob=float(lp_m[i]),
                     delta_logprob=float(d_lp[i]), delta_logit=float(mod_row[i] - base_row[i]))
                for i in idx]

    return {
        "kl_base_to_mod": kl,
        "top1_changed": int(base_row.argmax()) != int(mod_row.argmax()),
        "largest_prob_increase": rows(d_p.topk(k).indices.tolist()),
        "largest_prob_decrease": rows((-d_p).topk(k).indices.tolist()),
        "largest_logprob_increase": rows(d_lp.topk(k).indices.tolist()),
        "largest_logprob_decrease": rows((-d_lp).topk(k).indices.tolist()),
        "targets": [dict(token_info(tokenizer, t), base=token_stats(base_row, t), mod=token_stats(mod_row, t))
                    for t in (targets or [])],
    }
