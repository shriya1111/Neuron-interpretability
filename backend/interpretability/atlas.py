"""Serve the precomputed Atlas (cloud/modal_atlas.py output): per-neuron firings, statistics and readouts.

Directory layout (data/atlas/<tag>/): meta.json, docs.json, tokens.npy [n_docs, 256] (−1 = padding),
L00.npz … L{n-1}.npz with arrays described in cloud/modal_atlas.py.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

import numpy as np

DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "atlas"))


class AtlasStore:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        with open(os.path.join(path, "docs.json")) as f:
            self.docs = json.load(f)
        self.tokens = np.load(os.path.join(path, "tokens.npy"), mmap_mode="r")
        self.T = int(self.meta["tokens_per_doc"])
        self.n_layers = len([f for f in os.listdir(path) if re.fullmatch(r"L\d\d\.npz", f)])
        self._layers: dict[int, dict] = {}
        pos = self.meta["positions_per_domain"]
        self.dom_positions = np.array([pos.get(d, 0) for d in self.docs["domains"]], dtype=np.float64)
        self.total_positions = float(self.meta["positions_scanned_excl_pos0"])

    def layer(self, L: int) -> dict:
        if not 0 <= L < self.n_layers:
            raise ValueError(f"layer must be in [0, {self.n_layers - 1}]")
        if L not in self._layers:
            z = np.load(os.path.join(self.path, f"L{L:02d}.npz"))
            self._layers[L] = {k: z[k] for k in z.files}
        return self._layers[L]

    # ------------------------------------------------------------------ summaries
    def layer_summary(self, L: int) -> dict:
        z = self.layer(L)
        return {
            "layer": L, "n_neurons": int(z["max_abs"].shape[0]),
            "fire_rate_pos": (z["n_open_pos"] / self.total_positions).tolist(),
            "fire_rate_neg": (z["n_open_neg"] / self.total_positions).tolist(),
            "max_abs": z["max_abs"].tolist(),
            "top_pos": z["top_pos_val"][:, 0].tolist(), "top_neg": z["top_neg_val"][:, 0].tolist(),
        }

    def overview(self) -> dict:
        """Per-layer arrays for the layer strip (one value per neuron)."""
        return {"meta": self.meta, "layers": [
            {"layer": L, "max_abs": np.round(self.layer(L)["max_abs"], 3).tolist(),
             "fire_rate": np.round((self.layer(L)["n_open_pos"] + self.layer(L)["n_open_neg"]) / self.total_positions, 5).tolist()}
            for L in range(self.n_layers)]}

    def layer_table(self, tokenizer, L: int, sort: str = "max_abs", offset: int = 0, limit: int = 100,
                    readout: int = 5) -> dict:
        z = self.layer(L)
        n = z["max_abs"].shape[0]
        keys = {"max_abs": -z["max_abs"], "fire_rate": -(z["n_open_pos"] + z["n_open_neg"]), "index": np.arange(n),
                "top_pos": -z["top_pos_val"][:, 0], "top_neg": z["top_neg_val"][:, 0]}
        if sort not in keys:
            raise ValueError(f"sort must be one of {sorted(keys)}")
        order = np.argsort(keys[sort], kind="stable")[offset:offset + limit]
        dec = _decoder(tokenizer)
        rows = []
        for i in order.tolist():
            rows.append({"neuron": i, "max_abs": float(z["max_abs"][i]),
                         "fire_rate_pos": float(z["n_open_pos"][i] / self.total_positions),
                         "fire_rate_neg": float(z["n_open_neg"][i] / self.total_positions),
                         "top_pos": float(z["top_pos_val"][i, 0]), "top_neg": float(z["top_neg_val"][i, 0]),
                         "writes_pos": [dec(t) for t in z["read_pos_ids"][i, :readout].tolist()],
                         "writes_neg": [dec(t) for t in z["read_neg_ids"][i, :readout].tolist()]})
        return {"layer": L, "sort": sort, "offset": offset, "total": int(n), "rows": rows}

    # ------------------------------------------------------------------ one neuron
    def _snippet(self, dec, flat: int, before: int = 14, after: int = 4) -> dict:
        d, p = divmod(int(flat), self.T)
        row = self.tokens[d]
        L = int((row >= 0).sum())
        s, e = max(0, p - before), min(L, p + after + 1)
        return {"doc": d, "position": p, "domain": self.docs["domains"][self.docs["doc_domain"][d]],
                "tokens": [dec(int(t)) for t in row[s:e]], "peak_index": p - s, "token_ids": row[s:e].tolist(),
                "prefix_token_ids": row[:p + 1].tolist()}

    def neuron(self, tokenizer, L: int, n: int, k: int = 12) -> dict:
        z = self.layer(L)
        if not 0 <= n < z["max_abs"].shape[0]:
            raise ValueError(f"neuron must be in [0, {z['max_abs'].shape[0] - 1}]")
        dec = _decoder(tokenizer)
        sides = {}
        for s in ("pos", "neg"):
            vals, idxs = z[f"top_{s}_val"][n], z[f"top_{s}_idx"][n]
            fir = [dict(self._snippet(dec, i), activation=float(v)) for v, i in zip(vals[:k], idxs[:k]) if i >= 0]
            strong = z[f"n_strong_{s}"][n].astype(np.float64)
            rates = np.divide(strong, self.dom_positions, out=np.zeros_like(strong), where=self.dom_positions > 0)
            sides[s] = {
                "firings": fir,
                "n_open": int(z[f"n_open_{s}"][n]), "fire_rate": float(z[f"n_open_{s}"][n] / self.total_positions),
                "threshold_q999": float(z[f"thr_{s}"][n]), "threshold_q999_excl_pos0": float(z[f"thr_{s}_nz"][n]),
                "strong_rate_by_domain": {dname: float(r) for dname, r in zip(self.docs["domains"], rates)},
                "writes_logit_lens": [{"id": int(t), "text": dec(int(t)), "score": float(sc)}
                                      for t, sc in zip(z[f"read_{s}_ids"][n].tolist(), z[f"read_{s}_score"][n].tolist())],
            }
        jr = _jlens_readouts(L)
        if jr is not None:
            for s, key in (("pos", "plus"), ("neg", "minus")):
                sides[s]["writes_jlens"] = [{"id": int(t), "text": dec(int(t)), "cosine": float(c)}
                                            for t, c in zip(jr[f"{s}_ids"][n].tolist(), jr[f"{s}_cos"][n].tolist())]
        return {"layer": L, "neuron": n, "max_abs": float(z["max_abs"][n]), "plus": sides["pos"], "minus": sides["neg"],
                "jlens": None if jr is None else {"dir": _jlens_dir(), "method": "cos(d_{l,t}, ±w), J-lens of the neuron's layer"},
                "notes": {"firings": "strongest activations of that sign with the gate open (g·x > 0), position 0 excluded",
                          "fire_rate": "fraction of scanned positions where the gate is open and a has that sign",
                          "threshold_q999": "99.9th percentile of ±a over all positions of 200 random docs (Appendix B)",
                          "strong_rate_by_domain": "fraction of positions per Pile domain above that threshold",
                          "writes_logit_lens": "top tokens of W_U(γ_final ⊙ ±w): direct path only",
                          "writes_jlens": "top tokens by cosine with the fitted corpus-mean J-lens directions of this layer"}}

    def search_word(self, tokenizer, word: str, limit: int = 40) -> list[dict]:
        """Neurons whose ±w logit-lens readout (top 12) contains a token decoding to `word` (case-insensitive, trimmed)."""
        target = word.strip().lower()
        ids = {i for i in range(len(tokenizer)) if tokenizer.decode([i]).strip().lower() == target} if target else set()
        if not ids:
            return []
        idarr = np.array(sorted(ids))
        hits = []
        for L in range(self.n_layers):
            z = self.layer(L)
            for s in ("pos", "neg"):
                R = z[f"read_{s}_ids"]
                m = np.isin(R, idarr)
                for n in np.where(m.any(1))[0].tolist():
                    rank = int(np.argmax(m[n])) + 1
                    hits.append({"layer": L, "neuron": n, "sign": "+" if s == "pos" else "-", "rank_in_readout": rank})
        hits.sort(key=lambda h: (h["rank_in_readout"], h["layer"]))
        return hits[:limit]


def _jlens_dir() -> str:
    from interpretability.j_lens import DEFAULT_LENS
    return os.environ.get("NEURONLAB_JLENS", DEFAULT_LENS)


@lru_cache(maxsize=64)
def _jlens_readouts_cached(path: str, L: int):
    f = os.path.join(path, f"readouts_L{L:02d}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def _jlens_readouts(L: int):
    return _jlens_readouts_cached(_jlens_dir(), L)


def _decoder(tokenizer):
    @lru_cache(maxsize=65536)
    def dec(t: int) -> str:
        return tokenizer.decode([t])
    return dec


_STORES: dict[str, AtlasStore] = {}


def get_store(tag: str | None = None, root: str = DEFAULT_ROOT) -> AtlasStore:
    tag = tag or os.environ.get("NEURONLAB_ATLAS", "qwen3-0.6b-pile10k")
    path = os.path.join(root, tag)
    if path not in _STORES:
        if not os.path.exists(os.path.join(path, "meta.json")):
            raise FileNotFoundError(f"no atlas at {path}; run cloud/modal_atlas.py and download it")
        _STORES[path] = AtlasStore(path)
    return _STORES[path]


def list_atlases(root: str = DEFAULT_ROOT) -> list[dict]:
    out = []
    if os.path.isdir(root):
        for t in sorted(os.listdir(root)):
            mp = os.path.join(root, t, "meta.json")
            if os.path.exists(mp):
                with open(mp) as f:
                    m = json.load(f)
                out.append({"tag": t, "model_id": m.get("model_id"), "n_docs": m.get("n_docs")})
    return out
