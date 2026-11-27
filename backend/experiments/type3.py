"""Type 3 audit (paper, Appendix B.1): does ONE signed direction add TWO unrelated concepts?

A signed direction (neuron, sign) counts as type 3 only if it passes four checks, for topics A and B:

Check 1 (activation).  In topic documents (Pile-10k docs matching the topic's term list, <=150, 1,500-char window,
         first 256 tokens) the neuron activates strongly — s·a > q99.9 of s·a over all positions of the 200 reference
         docs — in >= 5 documents of each topic, and the share of such topic documents is >= 0.07 above the share
         among the 200 reference documents.
Check 2 (cross-topic effect).  Zero the neuron at every position of up to 40 strongly-activating documents of topic X.
         At strongly-activating positions, measure the drop in log-prob of the first tokens of the OTHER topic's words,
         minus the drop for 200 random words. It must be >= 0.03 nats and >= 3 SD above the same quantity for 20
         random directions with the norm of the output vector (each subtracts a_t·r, matching the ablation's size).
         Required in both directions (A docs -> B words and B docs -> A words).
Check 3 (specificity).  Words of other members of each topic's category (supplied by the user; the paper used a Claude
         judge) must NOT drop by >= 1/4 as much as the other topic's words with z >= 3.
Check 4 (whole words).  Among the 15 vocabulary tokens with the largest mean drop, at most half of the positive change
         may fall on word continuations (no leading space, lowercase first letter).

Defaults are the paper's numbers (150 / 40 / 200 / 20). Smaller values may be passed for exploration and are reported.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from datasets_nl import corpus
from evaluation.controls import random_directions
from evaluation.statistical_tests import summarize
from instrumentation.activation_cache import CaptureSpec, run_forward
from instrumentation.interventions import InterventionPlan, NeuronIntervention, VectorAdd

PAPER = {"max_topic_docs": 150, "n_ablate_docs": 40, "n_random_words": 200, "n_random_dirs": 20, "n_reference_docs": 200,
         "min_strong_docs": 5, "min_share_margin": 0.07, "min_drop_nats": 0.03, "min_z": 3.0, "check3_ratio": 0.25,
         "check4_top": 15, "check4_max_continuation_share": 0.5}
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache"))


def thresholds(layer: int, neuron: int, sign: str) -> float:
    from interpretability.atlas import get_store
    z = get_store().layer(layer)
    return float(z["thr_pos"][neuron] if sign == "+" else z["thr_neg"][neuron])


def reference_maxima(handle, layer: int, n_docs: int, progress=None) -> np.ndarray:
    """Per reference doc, max of +a and of -a for every neuron of the layer: [n_docs, 2, d_mlp] (cached on disk)."""
    os.makedirs(CACHE, exist_ok=True)
    tag = handle.model_id.replace("/", "_")
    f = os.path.join(CACHE, f"refmax_{tag}_L{layer:02d}_n{n_docs}.npy")
    if os.path.exists(f):
        return np.load(f)
    docs = corpus.reference_docs(handle.tokenizer, n_docs)
    out = np.zeros((len(docs), 2, handle.adapter.d_mlp(layer)), dtype=np.float32)
    for i, d in enumerate(docs):
        a = run_forward(handle, corpus.tokenize(handle.tokenizer, d["text"]), CaptureSpec(layers=[layer], points=("act",), final=False)).get("act", layer)
        out[i, 0] = a.max(0).values.numpy()
        out[i, 1] = (-a).max(0).values.numpy()
        if progress:
            progress(i / len(docs), f"reference docs {i + 1}/{len(docs)}")
    np.save(f, out)
    return out


def _signed(a: torch.Tensor, sign: str) -> torch.Tensor:
    return a if sign == "+" else -a


def check1(handle, layer, neuron, sign, docs_ids, thr, ref_share) -> tuple[dict, list[tuple[torch.Tensor, list[int]]]]:
    strong_docs = []
    for ids in docs_ids:
        a = run_forward(handle, ids, CaptureSpec(layers=[layer], points=("act",), final=False)).get("act", layer)[:, neuron]
        pos = torch.nonzero(_signed(a, sign) > thr).flatten().tolist()
        pos = [p for p in pos if p < ids.numel() - 1]  # need a next-token prediction at p
        if pos:
            strong_docs.append((ids, pos))
    share = len(strong_docs) / max(1, len(docs_ids))
    res = {"n_docs": len(docs_ids), "n_strong_docs": len(strong_docs), "share": share, "reference_share": ref_share,
           "margin": share - ref_share,
           "pass": len(strong_docs) >= PAPER["min_strong_docs"] and share - ref_share >= PAPER["min_share_margin"]}
    return res, strong_docs


def check2_3_4(handle, layer, neuron, sign, strong_docs, other_words, control_words, random_words, n_dirs, seed, tok,
               progress=None, p0=0.0, p1=1.0) -> dict:
    ad = handle.adapter
    w = ad.output_vector(layer, neuron).detach().float().cpu()
    dirs = random_directions(n_dirs, ad.d_model, float(w.norm()), seed)
    ow, cw, rw = (torch.tensor(x, dtype=torch.long) for x in (other_words, control_words or [0], random_words))
    sum_drop_vocab = torch.zeros(ad.vocab_size, dtype=torch.float64)
    n_pos = 0
    eff = {"other": 0.0, "control": 0.0}
    ctrl_eff = {"other": np.zeros(n_dirs), "control": np.zeros(n_dirs)}

    def metric(drop):  # drop: [P, V] base - modified log-probs at strong positions
        r = drop[:, rw].mean(1)
        return (drop[:, ow].mean(1) - r).sum().item(), (drop[:, cw].mean(1) - r).sum().item()

    for di, (ids, pos) in enumerate(strong_docs):
        base = run_forward(handle, ids, CaptureSpec(layers=[layer], points=("act",), final=False))
        a = base.get("act", layer)[:, neuron]
        lp0 = torch.log_softmax(base.logits[pos].double(), -1)
        abl = run_forward(handle, ids, None, InterventionPlan([NeuronIntervention(layer, [neuron], "zero")]))
        drop = lp0 - torch.log_softmax(abl.logits[pos].double(), -1)
        sum_drop_vocab += drop.sum(0)
        o, c = metric(drop)
        eff["other"] += o
        eff["control"] += c
        for k, r in enumerate(dirs):
            m = run_forward(handle, ids, None, InterventionPlan(vector_adds=[VectorAdd(layer, r, "mlp_out", None, (-a).tolist())]))
            o, c = metric(lp0 - torch.log_softmax(m.logits[pos].double(), -1))
            ctrl_eff["other"][k] += o
            ctrl_eff["control"][k] += c
        n_pos += len(pos)
        if progress:
            progress(p0 + (p1 - p0) * (di + 1) / len(strong_docs), f"ablation + {n_dirs} controls, doc {di + 1}/{len(strong_docs)}")
    if n_pos == 0:
        return {"n_positions": 0}
    for k in eff:
        eff[k] /= n_pos
        ctrl_eff[k] /= n_pos

    def z(x, arr):
        s = arr.std(ddof=1)
        return float((x - arr.mean()) / s) if s > 0 else None

    z_other, z_ctrl = z(eff["other"], ctrl_eff["other"]), z(eff["control"], ctrl_eff["control"])
    c2 = {"drop_other_minus_random": eff["other"], "controls": summarize(ctrl_eff["other"].tolist()), "z": z_other,
          "pass": eff["other"] >= PAPER["min_drop_nats"] and z_other is not None and z_other >= PAPER["min_z"]}
    c3 = {"drop_control_minus_random": eff["control"] if control_words else None, "z": z_ctrl if control_words else None,
          "ratio_to_other": (eff["control"] / eff["other"]) if control_words and eff["other"] != 0 else None,
          "pass": True if not control_words else not (eff["control"] >= PAPER["check3_ratio"] * eff["other"]
                                                      and z_ctrl is not None and z_ctrl >= PAPER["min_z"]),
          "evaluated": bool(control_words)}
    mean_drop = sum_drop_vocab / n_pos
    top = mean_drop.topk(PAPER["check4_top"])
    pos_change = top.values.clamp_min(0)
    cont = torch.tensor([corpus.is_word_continuation(tok, int(t)) for t in top.indices])
    share_cont = float(pos_change[cont].sum() / pos_change.sum()) if pos_change.sum() > 0 else 0.0
    c4 = {"top_tokens": [{"id": int(t), "text": tok.decode([int(t)]), "mean_drop": float(v), "continuation": bool(cc)}
                         for t, v, cc in zip(top.indices, top.values, cont)],
          "continuation_share_of_positive_change": share_cont,
          "pass": share_cont <= PAPER["check4_max_continuation_share"]}
    return {"n_positions": n_pos, "check2": c2, "check3": c3, "check4": c4}


def run(handle, layer: int, neuron: int, sign: str, topic_a: dict, topic_b: dict, *, max_topic_docs: int = 150,
        n_ablate_docs: int = 40, n_random_words: int = 200, n_random_dirs: int = 20, n_reference_docs: int = 200,
        seed: int = 0, progress=None) -> dict:
    handle.adapter.check_neuron(layer, neuron)
    if sign not in ("+", "-"):
        raise ValueError("sign must be + or -")
    tok = handle.tokenizer
    prog = progress or (lambda p, m="": None)
    thr = thresholds(layer, neuron, sign)
    prog(0.0, "reference documents (cached after the first run per layer)")
    ref = reference_maxima(handle, layer, n_reference_docs, lambda p, m: prog(0.25 * p, m))
    ref_share = float((ref[:, 0 if sign == "+" else 1, neuron] > thr).mean())
    rw = list(corpus.random_word_tokens(tok, n_random_words, seed))
    topics = {"A": topic_a, "B": topic_b}
    out = {"layer": layer, "neuron": neuron, "sign": sign, "threshold_q999": thr, "topics": {}, "params": {
        "max_topic_docs": max_topic_docs, "n_ablate_docs": n_ablate_docs, "n_random_words": n_random_words,
        "n_random_dirs": n_random_dirs, "n_reference_docs": n_reference_docs, "seed": seed},
        "paper_defaults": PAPER}
    out["mode"] = "paper_sample_sizes" if (max_topic_docs, n_ablate_docs, n_random_words, n_random_dirs, n_reference_docs) == \
        (150, 40, 200, 20, 200) else "reduced_sample_sizes (exploratory)"
    strong = {}
    for i, (key, t) in enumerate(topics.items()):
        docs = corpus.topic_docs(t["terms"], max_topic_docs)
        ids = [corpus.tokenize(tok, d["text"]) for d in docs]
        prog(0.25 + 0.15 * i, f"check 1 on topic {t.get('label', key)} ({len(ids)} docs)")
        c1, strong[key] = check1(handle, layer, neuron, sign, ids, thr, ref_share)
        out["topics"][key] = {"label": t.get("label", key), "terms": t["terms"], "words": t["words"],
                              "control_words": t.get("control_words", []), "check1": c1,
                              "example_matches": [d["match"] for d in docs[:8]]}
    for i, (key, other) in enumerate((("A", "B"), ("B", "A"))):
        other_words = corpus.word_first_tokens(tok, topics[other]["words"])
        ctrl_words = corpus.word_first_tokens(tok, topics[other].get("control_words", []))
        docs = strong[key][:n_ablate_docs]
        if not docs or not other_words:
            out["topics"][key]["cross"] = {"n_positions": 0}
            continue
        out["topics"][key]["cross"] = check2_3_4(handle, layer, neuron, sign, docs, other_words, ctrl_words, rw,
                                                 n_random_dirs, seed + 17 * (i + 1), tok, prog, 0.55 + 0.22 * i, 0.77 + 0.22 * i)
        out["topics"][key]["cross"]["measured"] = f"{topics[key].get('label', key)} documents → {topics[other].get('label', other)} words"
    out["verdict"] = verdict(out)
    return out


def verdict(out: dict) -> dict:
    T = out["topics"]
    c1 = all(T[k]["check1"]["pass"] for k in "AB")
    crosses = [T[k].get("cross", {}) for k in "AB"]
    if not c1 and any(T[k]["check1"]["n_strong_docs"] < PAPER["min_strong_docs"] for k in "AB"):
        return {"status": "insufficient_data", "reason": "fewer than 5 strongly-activating documents for a topic (check 1)",
                "checks": {"1": c1}}
    if any(c.get("n_positions", 0) == 0 for c in crosses):
        return {"status": "insufficient_data", "reason": "no strongly-activating positions to ablate", "checks": {"1": c1}}
    c2 = all(c["check2"]["pass"] for c in crosses)
    c3 = all(c["check3"]["pass"] for c in crosses)
    c4 = all(c["check4"]["pass"] for c in crosses)
    checks = {"1": c1, "2": c2, "3": c3, "4": c4}
    if all(checks.values()):
        return {"status": "supported", "reason": "passes all four Appendix B checks in both directions", "checks": checks}
    if any(c["check2"]["drop_other_minus_random"] <= 0 for c in crosses):
        return {"status": "contradicted", "reason": "ablation does not lower the other topic's words (check 2)", "checks": checks}
    if not c3 or not c4:
        return {"status": "contradicted", "reason": "effect is category-level (check 3) or on word pieces (check 4)", "checks": checks}
    return {"status": "inconclusive", "reason": "positive effect, but below the size/z thresholds or check 1 margin", "checks": checks}
