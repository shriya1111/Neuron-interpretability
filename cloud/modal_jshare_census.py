"""J-share census of every MLP neuron (paper Appendix A.1, Figure 2a/b), for a model with a fitted J-lens.

Same algorithm as backend/interpretability/j_share.py (nonnegative OMP over unit J-lens token directions, exact NNLS
refits with scipy, <= 25 directions), batched: correlations for thousands of vectors per step in one GPU matmul,
NNLS refits on a CPU process pool.

Inputs (Modal volumes): the J-lens (neuronlab-jlens/<lens_tag>/jacobian_lens.pt) and the Atlas
(neuronlab-atlas/<atlas_tag>/LXX.npz, used to decide the sign each neuron uses: the sign with the larger
open-gate activation on Pile-10k, as in Appendix C.2).

Per layer: J-share of +w and of -w for every neuron, and of 2,048 random unit directions (seeded).
Classification of the used sign: high > p99 of random, low < p1, middle otherwise.

    NEURONLAB_GPU=A10G modal deploy cloud/modal_jshare_census.py
    python cloud/modal_jshare_census.py spawn
    modal volume get neuronlab-jlens qwen3-0.6b-wikitext-n1000/census data/jlens/qwen3-0.6b-wikitext-n1000/
"""

from __future__ import annotations

import json
import os
import time

import modal

GPU = os.environ.get("NEURONLAB_GPU", "A10G")
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
app = modal.App("neuronlab-jshare")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "transformers>=5.5", "numpy", "scipy", "huggingface_hub")
    .pip_install(f"git+https://github.com/anthropics/jacobian-lens@{JLENS_COMMIT}")
    .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_XET": "1"})
)
jl_vol = modal.Volume.from_name("neuronlab-jlens", create_if_missing=True)
at_vol = modal.Volume.from_name("neuronlab-atlas", create_if_missing=True)
hf_vol = modal.Volume.from_name("neuronlab-hf-cache", create_if_missing=True)


def _nnls_job(args):
    import numpy as np
    from scipy.optimize import nnls
    A, w = args
    c, _ = nnls(A, w)
    return c


@app.function(image=image, gpu=GPU, cpu=16.0, memory=65536, timeout=6 * 3600,
              volumes={"/jl": jl_vol, "/at": at_vol, "/hf": hf_vol})
def census(model_id: str, lens_tag: str, atlas_tag: str, k: int = 25, n_random: int = 2048, seed: int = 0,
           batch: int = 2048, layers: list[int] | None = None, out_name: str = "census") -> dict:
    import multiprocessing as mp

    import jlens
    import numpy as np
    import torch
    import transformers

    t0 = time.time()
    dev = "cuda"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    lens = jlens.JacobianLens.load(f"/jl/{lens_tag}/jacobian_lens.pt")
    J = {int(a): b.float() for a, b in lens.jacobians.items()}
    nL = len(model.model.layers)
    WUg = (model.lm_head.weight.detach().float() * model.model.norm.weight.detach().float()).to(dev)
    n_valid = len(transformers.AutoTokenizer.from_pretrained(model_id))  # rows beyond the tokenizer are padding
    out_dir = f"/jl/{lens_tag}/{out_name}"
    os.makedirs(out_dir, exist_ok=True)
    pool = mp.get_context("fork").Pool(16)

    def pursuit(D: torch.Tensor, Dc: np.ndarray, W: np.ndarray) -> np.ndarray:
        """D: [V, d] unit directions on GPU. W: [d, n] float64 unit columns. Returns J-shares [n]."""
        d, n = W.shape
        supports = [[] for _ in range(n)]
        R = W.copy()
        active = np.ones(n, bool)
        for _ in range(k):
            idx = np.where(active)[0]
            if idx.size == 0:
                break
            C = D @ torch.from_numpy(R[:, idx]).float().to(dev)  # [V, |idx|]
            for col, i in enumerate(idx):
                if supports[i]:
                    C[supports[i], col] = -float("inf")
            best_val, best = C.max(0)
            best, best_val = best.cpu().numpy(), best_val.cpu().numpy()
            jobs, cols = [], []
            for col, i in enumerate(idx):
                if best_val[col] <= 1e-12:
                    active[i] = False
                    continue
                cand = supports[i] + [int(best[col])]
                jobs.append((Dc[cand].astype(np.float64).T, W[:, i]))
                cols.append((i, cand))
            coefs = pool.map(_nnls_job, jobs, chunksize=64)
            for (i, cand), c in zip(cols, coefs):
                keep = c > 0
                new_sup = [s for s, kk in zip(cand, keep) if kk]
                A = Dc[new_sup].astype(np.float64).T if new_sup else np.zeros((d, 0))
                R[:, i] = W[:, i] - (A @ c[keep] if new_sup else 0)
                if cand[-1] not in new_sup:
                    active[i] = False
                supports[i] = new_sup
        return 1.0 - (R * R).sum(0)

    summary = {"model_id": model_id, "lens": lens_tag, "atlas": atlas_tag, "k": k, "n_random": n_random, "seed": seed,
               "dictionary": f"{n_valid} tokenizer tokens (embedding padding rows excluded)",
               "algorithm": "nonnegative OMP + scipy NNLS refit (same as backend/interpretability/j_share.py)", "layers": []}
    todo = layers if layers is not None else list(range(nL))
    for L in todo:
        f_out = f"{out_dir}/L{L:02d}.npz"
        if os.path.exists(f_out):
            continue
        D = WUg if L == nL - 1 else WUg @ J[L].to(dev)
        D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
        D[n_valid:] = 0.0  # padding rows are not tokens: never selected (same as the backend)
        Dc = D.cpu().numpy()
        Wd = model.model.layers[L].mlp.down_proj.weight.detach().double().numpy()
        Wn = Wd / np.linalg.norm(Wd, axis=0, keepdims=True)
        g = torch.Generator().manual_seed(seed + 1000 * L)
        Rnd = torch.randn(Wd.shape[0], n_random, generator=g, dtype=torch.float64).numpy()
        Rnd = Rnd / np.linalg.norm(Rnd, axis=0, keepdims=True)
        allv = np.concatenate([Wn, -Wn, Rnd], axis=1)
        shares = np.concatenate([pursuit(D, Dc, allv[:, s:s + batch]) for s in range(0, allv.shape[1], batch)])
        m = Wd.shape[1]
        js_pos, js_neg, js_rnd = shares[:m], shares[m:2 * m], shares[2 * m:]
        z = np.load(f"/at/{atlas_tag}/L{L:02d}.npz")
        sign_pos = z["top_pos_val"][:, 0] >= -z["top_neg_val"][:, 0]
        used = np.where(sign_pos, js_pos, js_neg)
        p1, p99 = np.percentile(js_rnd, [1, 99])
        cls = np.where(used > p99, 2, np.where(used < p1, 0, 1))
        np.savez_compressed(f_out, js_pos=js_pos, js_neg=js_neg, js_random=js_rnd, sign_used_pos=sign_pos,
                            js_used=used, cls=cls, p1=p1, p99=p99)
        row = {"layer": L, "frac_high": float((cls == 2).mean()), "frac_middle": float((cls == 1).mean()),
               "frac_low": float((cls == 0).mean()), "median_used": float(np.median(used)),
               "median_random": float(np.median(js_rnd)), "ratio_median": float(np.median(used) / np.median(js_rnd)),
               "p1_random": float(p1), "p99_random": float(p99), "seconds": round(time.time() - t0, 1)}
        summary["layers"].append(row)
        print(f"[census] L{L}: high {row['frac_high']:.3f} middle {row['frac_middle']:.3f} low {row['frac_low']:.3f} "
              f"median ratio {row['ratio_median']:.3f}  ({row['seconds']}s)", flush=True)
        jl_vol.commit()
    pool.close()
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    jl_vol.commit()
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["spawn"])
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lens-tag", default="qwen3-0.6b-wikitext-n1000")
    ap.add_argument("--atlas-tag", default="qwen3-0.6b-pile10k")
    ap.add_argument("--n-random", type=int, default=2048)
    ap.add_argument("--layers", default="")
    ap.add_argument("--out-name", default="census")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else None
    call = modal.Function.from_name("neuronlab-jshare", "census").spawn(a.model_id, a.lens_tag, a.atlas_tag,
                                                                        n_random=a.n_random, layers=layers,
                                                                        out_name=a.out_name)
    print("spawned function call:", call.object_id)
