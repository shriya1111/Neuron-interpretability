"""Exploratory Type 3 candidate scanner (paper Appendix B.3 heuristics). Candidate detection only — NOT confirmation.

For each neuron and sign in the chosen layers:
  readout split   the top-12 J-lens readout tokens of ±w are split in two by the sign of the leading eigenvector of
                  their centered cosine-similarity matrix (J-lens directions of the layer); score = mean within-group
                  cosine − mean between-group cosine; both groups need ≥ 3 tokens and must not be spelling variants of
                  each other (no stripped/lowercased token appears in both groups), and each group needs ≥ 3 word-like
                  tokens (≥ 3 letters) — punctuation/whitespace clusters are formatting, not topics (our filter).
  pair reconstruction  w is fit by least squares with the best pair among its 8 most-aligned same-layer neurons;
                  reported R² and the |cos| between the two partners (low = unrelated partners).
  firing split    the peak tokens of the Atlas's top firings of that sign (embedding rows, unit) split the same way;
                  score as above.
Candidates are ranked by the readout-split score; the other two are shown as supporting evidence.
"""

from __future__ import annotations

import numpy as np
import torch

from interpretability.atlas import _jlens_readouts, get_store


def _split(S: np.ndarray) -> tuple[float, np.ndarray] | None:
    n = S.shape[0]
    C = S - S.mean()
    vals, vecs = np.linalg.eigh(C)
    g = vecs[:, -1] >= 0
    if g.sum() < 3 or (~g).sum() < 3:
        return None
    iu = ~np.eye(n, dtype=bool)
    within = np.concatenate([S[np.ix_(g, g)][iu[np.ix_(g, g)]], S[np.ix_(~g, ~g)][iu[np.ix_(~g, ~g)]]]).mean()
    between = S[np.ix_(g, ~g)].mean()
    return float(within - between), g


def scan(handle, lens, layers: list[int], top: int = 50, progress=None) -> dict:
    tok = handle.tokenizer
    ad = handle.adapter
    st = get_store()
    E = ad.embed.weight.detach().float()
    E = E / E.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cands = []
    for li, L in enumerate(layers):
        jr = _jlens_readouts(L)
        if jr is None:
            raise ValueError("J-lens readouts missing; run scripts/precompute_jlens_readouts.py")
        D = lens.directions(L)
        Wd = ad.mlp(L).down_proj.weight.detach().float()
        Wn = Wd / Wd.norm(dim=0, keepdim=True)
        G = (Wn.T @ Wn).numpy()
        np.fill_diagonal(G, 0)
        z = st.layer(L)
        for n in range(Wd.shape[1]):
            for s, key in (("+", "pos"), ("-", "neg")):
                ids = jr[f"{key}_ids"][n]
                V = D[ids].numpy()
                sp = _split(V @ V.T)
                if sp is None:
                    continue
                score, g = sp
                words = [tok.decode([int(t)]).strip().lower() for t in ids]
                if set(np.array(words)[g]) & set(np.array(words)[~g]):
                    continue
                wordy = np.array([sum(ch.isalpha() for ch in wd) >= 3 for wd in words])
                if wordy[g].sum() < 3 or wordy[~g].sum() < 3:  # topics need words, not punctuation/whitespace
                    continue
                cands.append({"layer": L, "neuron": n, "sign": s, "readout_split_score": score,
                              "group_a": [tok.decode([int(t)]) for t in ids[g]],
                              "group_b": [tok.decode([int(t)]) for t in ids[~g]]})
        # supporting evidence only for this layer's best candidates (cheap)
        layer_c = sorted([c for c in cands if c["layer"] == L], key=lambda c: -c["readout_split_score"])[:top]
        for c in layer_c:
            n = c["neuron"]
            partners = np.argsort(-np.abs(G[n]))[:8]
            best = (0.0, None, None)
            w = Wn[:, n].numpy()
            for i in range(len(partners)):
                for j in range(i + 1, len(partners)):
                    A = Wn[:, [partners[i], partners[j]]].numpy()
                    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
                    r2 = 1 - float(((w - A @ coef) ** 2).sum())
                    if r2 > best[0]:
                        best = (r2, int(partners[i]), int(partners[j]))
            c["pair_reconstruction"] = {"r2": best[0], "partners": [best[1], best[2]],
                                        "partner_abs_cos": float(abs(G[best[1], best[2]])) if best[1] is not None else None}
            key = "pos" if c["sign"] == "+" else "neg"
            idx = z[f"top_{key}_idx"][n]
            idx = idx[idx >= 0]
            peak = np.array([st.tokens[d][p] for d, p in (divmod(int(i), st.T) for i in idx)])
            if len(peak) >= 6:
                P = E[torch.tensor(peak)].numpy()
                fs = _split(P @ P.T)
                c["firing_split"] = None if fs is None else {"score": fs[0], "group_a": [tok.decode([int(t)]) for t in peak[fs[1]]][:8],
                                                             "group_b": [tok.decode([int(t)]) for t in peak[~fs[1]]][:8]}
        if progress:
            progress((li + 1) / len(layers), f"layer {L} scanned")
    cands.sort(key=lambda c: -c["readout_split_score"])
    return {"candidates": cands[:top], "n_scored": len(cands), "layers": layers,
            "status": "exploratory — candidates only; run the four checks to confirm",
            "method": "readout split (J-lens top-12), pair reconstruction, firing-token split (Appendix B.3 heuristics)"}
