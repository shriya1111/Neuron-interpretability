"""Compare the edited model against the original weights on the same inputs."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from evaluation import metrics
from instrumentation.activation_cache import run_plain
from instrumentation.weight_edits import get_editor


@lru_cache(maxsize=1)
def heldout_texts(n: int = 40, min_chars: int = 500) -> tuple[str, ...]:
    """Deterministic held-out set: first n WikiText-103 *validation* rows with >= min_chars characters
    (the J-lens is fit on the train split, so this does not overlap it)."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    out = []
    for t in ds["text"]:
        if len(t.strip()) >= min_chars:
            out.append(t.strip()[:2000])
            if len(out) == n:
                break
    return tuple(out)


def mean_nll(handle, texts: list[str], max_tokens: int = 128) -> dict:
    tot, cnt = 0.0, 0
    per = []
    for tx in texts:
        ids = handle.tokenizer(tx, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids[0]
        if ids.numel() < 2:
            continue
        logits = run_plain(handle, ids)
        lp = torch.log_softmax(logits[:-1].double(), -1)
        nll = -lp.gather(1, ids[1:].view(-1, 1)).squeeze(1)
        per.append(float(nll.mean()))
        tot += float(nll.sum())
        cnt += int(nll.numel())
    m = tot / max(cnt, 1)
    return {"mean_nll": m, "perplexity": math.exp(m), "tokens": cnt, "per_text_mean_nll": per}


def compare(handle, prompts: list[dict], k: int = 10, n_heldout: int = 0) -> dict:
    ed = get_editor(handle)
    tok = handle.tokenizer
    rows = []
    for p in prompts:
        ids = tok(p["prompt"], return_tensors="pt").input_ids[0]
        edited = run_plain(handle, ids)[-1]
        with ed.original_weights():
            orig = run_plain(handle, ids)[-1]
        targets = []
        for s in p.get("targets", []):
            t = tok.encode(s, add_special_tokens=False)
            if len(t) != 1:
                raise ValueError(f"target {s!r} is {len(t)} tokens; use a single-token string")
            targets.append(t[0])
        rows.append({"prompt": p["prompt"], "original_top_k": metrics.top_k(tok, orig, k),
                     "edited_top_k": metrics.top_k(tok, edited, k),
                     "comparison": metrics.compare(tok, orig, edited, k, targets)})
    out = {"active_edits": ed.summary(), "prompts": rows}
    if n_heldout:
        texts = list(heldout_texts())[:n_heldout]
        e = mean_nll(handle, texts)
        with ed.original_weights():
            o = mean_nll(handle, texts)
        out["heldout"] = {"dataset": "Salesforce/wikitext wikitext-103-raw-v1 validation (first rows ≥500 chars, 128 tokens)",
                          "n_texts": len(texts), "original": o, "edited": e,
                          "delta_mean_nll": e["mean_nll"] - o["mean_nll"],
                          "perplexity_ratio": e["perplexity"] / o["perplexity"]}
    return out
