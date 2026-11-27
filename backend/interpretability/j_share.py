"""J-share (paper, Appendix A.1).

For a unit vector w at layer l and the unit J-lens directions {d_t} of that layer:

    v = sum_j c_j d_{t_j},   c_j >= 0,   at most K = 25 directions
    J-share(w) = 1 - ||w - v||^2           (fraction of ||w||^2 = 1 that v explains)

Fitting: nonnegative orthogonal matching pursuit with coefficient refitting. At each step the token whose direction
has the largest positive correlation with the current residual joins the support, then *all* coefficients on the
support are refit by nonnegative least squares (scipy.optimize.nnls); tokens whose coefficient becomes 0 drop out.
Stops at K directions or when no direction correlates positively with the residual.

The paper (and the J-lens paper) name "nonnegative pursuit" / "gradient pursuit" without giving hyperparameters;
this NN-OMP is our explicit choice. With refitting it returns the exact NNLS optimum on the selected support.

Random baseline: J-shares of N random unit directions at the same layer (paper: N = 2,048).
High = above the 99th percentile of the random J-shares, low = below the 1st, middle otherwise.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from scipy.optimize import nnls

K_DEFAULT = 25


def j_share(D: torch.Tensor, w: torch.Tensor, k: int = K_DEFAULT) -> dict:
    """D: [V, d] unit token directions. w: [d] (normalized internally)."""
    w = (w.double() / w.double().norm()).cpu()
    support: list[int] = []
    coef = np.zeros(0)
    r = w.clone()
    for _ in range(k):
        corr = (D @ r.float()).double()
        if support:
            corr[support] = -np.inf
        j = int(corr.argmax())
        if corr[j] <= 1e-12:
            break
        cand = support + [j]
        A = D[cand].double().T.numpy()  # [d, |S|]
        c, _ = nnls(A, w.numpy())
        keep = c > 0
        support = [s for s, kk in zip(cand, keep) if kk]
        coef = c[keep]
        v = torch.from_numpy(D[support].double().T.numpy() @ coef) if support else torch.zeros_like(w)
        r = w - v
        if j not in support:  # the newly added direction got coefficient 0 -> no further progress possible
            break
    v = torch.from_numpy(D[support].double().T.numpy() @ coef) if support else torch.zeros_like(w)
    resid = w - v
    share = 1.0 - float(resid @ resid)
    order = np.argsort(-coef)
    return {"j_share": share, "k_used": len(support), "token_ids": [int(support[i]) for i in order],
            "coefficients": [float(coef[i]) for i in order], "verbalizable": v.float(), "unexplained": resid.float(),
            "residual_norm_sq": float(resid @ resid), "all_coefficients_nonnegative": bool((coef >= 0).all())}


def j_share_batch(D: torch.Tensor, W: torch.Tensor, k: int = K_DEFAULT) -> np.ndarray:
    """J-shares for many vectors (columns of W: [d, n]); same algorithm, correlations computed in one matmul."""
    W = W.double() / W.double().norm(dim=0, keepdim=True)
    n = W.shape[1]
    supports: list[list[int]] = [[] for _ in range(n)]
    coefs: list[np.ndarray] = [np.zeros(0) for _ in range(n)]
    R = W.clone()
    active = np.ones(n, bool)
    for _ in range(k):
        if not active.any():
            break
        idx = np.where(active)[0]
        C = (D @ R[:, idx].float()).double()  # [V, |active|]
        for col, i in enumerate(idx):
            c = C[:, col]
            if supports[i]:
                c[supports[i]] = -np.inf
            j = int(c.argmax())
            if c[j] <= 1e-12:
                active[i] = False
                continue
            cand = supports[i] + [j]
            cf, _ = nnls(D[cand].double().T.numpy(), W[:, i].numpy())
            keep = cf > 0
            supports[i] = [s for s, kk in zip(cand, keep) if kk]
            coefs[i] = cf[keep]
            v = D[supports[i]].double().T.numpy() @ coefs[i] if supports[i] else np.zeros(W.shape[0])
            R[:, i] = W[:, i] - torch.from_numpy(v)
            if j not in supports[i]:
                active[i] = False
    return np.array([1.0 - float(R[:, i] @ R[:, i]) for i in range(n)])


def random_baseline(lens, layer: int, n: int = 2048, k: int = K_DEFAULT, seed: int = 0, cache: bool = True) -> np.ndarray:
    f = os.path.join(lens.path, f"jshare_random_L{layer:02d}_n{n}_k{k}_s{seed}.npy")
    if cache and os.path.exists(f):
        return np.load(f)
    g = torch.Generator().manual_seed(seed + 1000 * layer)
    R = torch.randn(lens.adapter.d_model, n, generator=g, dtype=torch.float64)
    D = lens.directions(layer)
    out = np.concatenate([j_share_batch(D, R[:, s:s + 256], k) for s in range(0, n, 256)])
    if cache:
        np.save(f, out)
    return out


def classify(share: float, baseline: np.ndarray) -> dict:
    p1, p99 = np.percentile(baseline, [1, 99])
    pct = float((baseline < share).mean() * 100)
    label = "high" if share > p99 else ("low" if share < p1 else "middle")
    return {"percentile_vs_random": pct, "random_p1": float(p1), "random_p99": float(p99),
            "random_median": float(np.median(baseline)), "n_random": int(len(baseline)), "class": label}
