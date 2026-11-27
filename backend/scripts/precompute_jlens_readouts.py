"""Precompute J-lens readouts (top-k tokens by cosine) of +w and -w for every MLP neuron.

    cd backend && python scripts/precompute_jlens_readouts.py [--k 12]

Writes <lens dir>/readouts_L{LL}.npz with pos_ids, pos_cos, neg_ids, neg_cos ([n_neurons, k]).
Layer l uses J_l (output of block l); the last layer uses the unembedding (J = I).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interpretability.j_lens import JLens  # noqa: E402
from models.loader import load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--chunk", type=int, default=512)
a = ap.parse_args()
torch.set_num_threads(os.cpu_count() or 4)
h = load_model(a.model_id, device="cpu", dtype="float32")
lens = JLens(h.adapter, cache_layers=1)
t0 = time.time()
for L in range(h.adapter.n_layers):
    out = os.path.join(lens.path, f"readouts_L{L:02d}.npz")
    if os.path.exists(out):
        continue
    D = lens.directions(L)
    Wd = h.adapter.mlp(L).down_proj.weight.detach().float()
    Wd = Wd / Wd.norm(dim=0, keepdim=True).clamp_min(1e-12)
    res = {k: [] for k in ("pos_ids", "pos_cos", "neg_ids", "neg_cos")}
    with torch.no_grad():
        for c in range(0, Wd.shape[1], a.chunk):
            C = D @ Wd[:, c:c + a.chunk]
            vp, ip = C.topk(a.k, dim=0)
            vn, inn = (-C).topk(a.k, dim=0)
            res["pos_ids"].append(ip.T.numpy()); res["pos_cos"].append(vp.T.numpy())
            res["neg_ids"].append(inn.T.numpy()); res["neg_cos"].append(vn.T.numpy())
    np.savez_compressed(out, **{k: np.concatenate(v) for k, v in res.items()})
    print(f"L{L} done, {time.time() - t0:.0f}s", flush=True)
print("all done", flush=True)
