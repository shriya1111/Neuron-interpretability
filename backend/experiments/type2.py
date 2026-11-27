"""Type 2: do the two signs of one neuron add two different concepts (+w one thing, -w another)?

Per sign s (contexts = the Atlas's strongest open-gate firings of that sign, run as exact token-id prefixes, read at the
firing position):
  ablation effect on own words   = mean drop in log-prob of s's words − drop for random words
  ablation effect on other words = same for the other sign's words (expected ≈ 0 for a clean type 2 neuron)
  controls                       = N random directions with ||r|| = ||w||, each adding −a_t·r at every position

Words per sign default to the top J-lens readout tokens of that sign (whole-word tokens only); you can override them.
A sign "adds its concept" when the own-word effect is ≥ 0.03 nats and z ≥ 3 against the controls (Appendix B check 2
thresholds). Type 2 is supported when both signs do; whether the two concepts are unrelated is your judgment.

Steering comparison (neutral prompts): set the activation to +A (A = max open-gate +activation on Pile-10k) or −A'
(A' = max −activation) at every position, or zero it, and report the change in each sign's words.
"""

from __future__ import annotations

import numpy as np
import torch

from datasets_nl import corpus
from evaluation.controls import random_directions
from evaluation.statistical_tests import summarize
from instrumentation.activation_cache import CaptureSpec, run_forward
from instrumentation.interventions import InterventionPlan, NeuronIntervention, VectorAdd

NEUTRAL = ["I looked up and saw a", "Let me tell you about", "The next thing I want to mention is", "She said that the",
           "In this article we discuss", "Here is a short story about", "The most important part is", "He pointed at the"]


def default_words(handle, layer: int, neuron: int, sign: str, n: int = 8) -> list[str]:
    from interpretability.atlas import _jlens_readouts
    jr = _jlens_readouts(layer)
    if jr is None:
        raise ValueError("no J-lens readouts; give words explicitly or run scripts/precompute_jlens_readouts.py")
    ids = jr["pos_ids" if sign == "+" else "neg_ids"][neuron].tolist()
    out = []
    for t in ids:
        s = handle.tokenizer.decode([t]).strip()
        # any script counts (e.g. 注明); require >= 2 letters so punctuation and word pieces like "s" are skipped
        if sum(ch.isalpha() for ch in s) >= 2 and s not in out:
            out.append(s)
        if len(out) == n:
            break
    return out


def run(handle, layer: int, neuron: int, words_plus: list[str] | None = None, words_minus: list[str] | None = None,
        n_contexts: int = 12, n_random_dirs: int = 20, n_random_words: int = 200, seed: int = 0, progress=None) -> dict:
    from interpretability.atlas import get_store
    ad, tok = handle.adapter, handle.tokenizer
    ad.check_neuron(layer, neuron)
    prog = progress or (lambda p, m="": None)
    st = get_store()
    nd = st.neuron(tok, layer, neuron, k=n_contexts)
    words = {"+": words_plus or default_words(handle, layer, neuron, "+"),
             "-": words_minus or default_words(handle, layer, neuron, "-")}
    for s in "+-":
        if not words[s]:
            raise ValueError(f"no usable words for the {s} side (readout has no word-like tokens); pass words_{'plus' if s == '+' else 'minus'} explicitly")
    wtok = {s: torch.tensor(corpus.word_first_tokens(tok, words[s])) for s in "+-"}
    rw = torch.tensor(list(corpus.random_word_tokens(tok, n_random_words, seed)))
    w = ad.output_vector(layer, neuron).detach().float().cpu()
    dirs = random_directions(n_random_dirs, ad.d_model, float(w.norm()), seed)

    def effects(lp0, lp1):
        drop = lp0 - lp1
        r = drop[rw].mean()
        return float(drop[wtok["+"]].mean() - r), float(drop[wtok["-"]].mean() - r)

    sides = {}
    total = sum(len(nd[k]["firings"][:n_contexts]) for k in ("plus", "minus"))
    done = 0
    for s, key in (("+", "plus"), ("-", "minus")):
        fir = nd[key]["firings"][:n_contexts]
        own, other = [], []
        ctrl_own = np.zeros((len(fir), n_random_dirs))
        for i, f in enumerate(fir):
            ids = torch.tensor(f["prefix_token_ids"])
            base = run_forward(handle, ids, CaptureSpec(layers=[layer], points=("act", "gate_pre"), final=False))
            a = base.get("act", layer)[:, neuron]
            lp0 = torch.log_softmax(base.logits[-1].double(), -1)
            abl = run_forward(handle, ids, None, InterventionPlan([NeuronIntervention(layer, [neuron], "zero")]))
            ep, em = effects(lp0, torch.log_softmax(abl.logits[-1].double(), -1))
            own.append(ep if s == "+" else em)
            other.append(em if s == "+" else ep)
            for k, r in enumerate(dirs):
                m = run_forward(handle, ids, None, InterventionPlan(vector_adds=[VectorAdd(layer, r, "mlp_out", None, (-a).tolist())]))
                cp, cm = effects(lp0, torch.log_softmax(m.logits[-1].double(), -1))
                ctrl_own[i, k] = cp if s == "+" else cm
            done += 1
            prog(0.85 * done / max(1, total), f"{s} side, context {i + 1}/{len(fir)}")
        if not fir:
            sides[key] = {"n_contexts": 0, "status": "insufficient_data"}
            continue
        mean_own = float(np.mean(own))
        cm = ctrl_own.mean(0)
        sd = cm.std(ddof=1) if len(cm) > 1 else 0.0
        z = float((mean_own - cm.mean()) / sd) if sd > 0 else None
        ok = mean_own >= 0.03 and z is not None and z >= 3
        sides[key] = {"n_contexts": len(fir), "words": words[s], "own_word_effect": mean_own,
                      "other_sign_word_effect": float(np.mean(other)), "per_context_own": own,
                      "controls": summarize(cm.tolist()), "z": z,
                      "gate_open_at_all_firings": True,
                      "status": "supported" if ok else ("contradicted" if mean_own <= 0 else "inconclusive")}

    # steering comparison on neutral prompts
    A_pos = max([f["activation"] for f in nd["plus"]["firings"]] + [0.0])
    A_neg = min([f["activation"] for f in nd["minus"]["firings"]] + [0.0])
    steer = []
    for j, p in enumerate(NEUTRAL):
        ids = tok(p, return_tensors="pt").input_ids[0]
        lp0 = torch.log_softmax(run_forward(handle, ids).logits[-1].double(), -1)
        row = {"prompt": p}
        for name, op, val in (("ablate", "zero", 0.0), ("steer_plus", "set", A_pos), ("steer_minus", "set", A_neg)):
            lp1 = torch.log_softmax(run_forward(handle, ids, None, InterventionPlan([NeuronIntervention(layer, [neuron], op, val)])).logits[-1].double(), -1)
            gain = lp1 - lp0
            row[name] = {"plus_words": float(gain[wtok["+"]].mean() - gain[rw].mean()),
                         "minus_words": float(gain[wtok["-"]].mean() - gain[rw].mean())}
        steer.append(row)
        prog(0.85 + 0.15 * (j + 1) / len(NEUTRAL), f"steering prompt {j + 1}/{len(NEUTRAL)}")
    summary_steer = {name: {k: float(np.mean([r[name][k] for r in steer])) for k in ("plus_words", "minus_words")}
                     for name in ("ablate", "steer_plus", "steer_minus")}
    both = all(sides[k].get("status") == "supported" for k in ("plus", "minus"))
    return {"layer": layer, "neuron": neuron, "words": words, "sides": sides,
            "steering": {"activation_plus": A_pos, "activation_minus": A_neg, "per_prompt": steer, "mean": summary_steer,
                         "note": "Δ log-prob of each sign's words minus random words, on neutral prompts, activation set at every position"},
            "verdict": {"status": "supported" if both else ("insufficient_data" if any(sides[k].get("n_contexts", 0) == 0 for k in ("plus", "minus"))
                                                         else "inconclusive" if not any(sides[k].get("status") == "contradicted" for k in ("plus", "minus")) else "contradicted"),
                        "rule": "each sign's ablation lowers its own words in its own contexts (≥0.03 nats, z≥3); "
                                "whether the two concepts are unrelated is your judgment"},
            "params": {"n_contexts": n_contexts, "n_random_dirs": n_random_dirs, "n_random_words": n_random_words, "seed": seed}}
