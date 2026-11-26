"""Matched random controls."""

from __future__ import annotations

import torch


def random_directions(n: int, d: int, norm: float, seed: int) -> torch.Tensor:
    """n isotropic Gaussian directions in R^d, each rescaled to exactly ``norm``. Shape [n, d]."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(n, d, generator=g, dtype=torch.float64)
    r = r / r.norm(dim=-1, keepdim=True) * norm
    return r.float()


def random_neurons(n: int, d_mlp: int, exclude: set[int], seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    out: list[int] = []
    for i in torch.randperm(d_mlp, generator=g).tolist():
        if i not in exclude:
            out.append(i)
        if len(out) == n:
            break
    return out
