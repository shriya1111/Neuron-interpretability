"""Activation-level interventions on MLP neurons.

An intervention edits the neuron activation vector ``a = silu(gate) * up``
right before it enters ``down_proj`` (a forward pre-hook on ``down_proj``).
This changes exactly the neuron's residual contribution ``a_i * w_i`` and
nothing else at that layer; downstream layers then see the change through a
real forward pass. Model weights are never modified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import torch

Op = Literal["zero", "scale", "negate", "set", "add"]
OPS = ("zero", "scale", "negate", "set", "add")


@dataclass
class NeuronIntervention:
    layer: int
    neurons: list[int]
    op: Op
    value: float = 0.0  # factor for "scale", target for "set", delta for "add"
    positions: list[int] | None = None  # None = all positions
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_positions(positions: list[int] | None, seq_len: int) -> list[int]:
    if positions is None:
        return list(range(seq_len))
    out = []
    for p in positions:
        q = p + seq_len if p < 0 else p
        if not (0 <= q < seq_len):
            raise ValueError(f"position {p} out of range for sequence of length {seq_len}")
        out.append(q)
    return sorted(set(out))


def validate(iv: NeuronIntervention, adapter, seq_len: int) -> None:
    if iv.op not in OPS:
        raise ValueError(f"op must be one of {OPS}, got {iv.op!r}")
    if not iv.neurons:
        raise ValueError("intervention needs at least one neuron")
    for n in iv.neurons:
        adapter.check_neuron(iv.layer, n)
    resolve_positions(iv.positions, seq_len)


def apply_op(a: torch.Tensor, iv: NeuronIntervention, seq_len: int) -> torch.Tensor:
    """Return a modified copy of activations ``a`` with shape [batch, seq, d_mlp]."""
    pos = torch.tensor(resolve_positions(iv.positions, seq_len), device=a.device)
    neu = torch.tensor(iv.neurons, device=a.device)
    out = a.clone()
    sub = out[:, pos][:, :, neu]
    if iv.op == "zero":
        new = torch.zeros_like(sub)
    elif iv.op == "scale":
        new = sub * iv.value
    elif iv.op == "negate":
        new = -sub
    elif iv.op == "set":
        new = torch.full_like(sub, iv.value)
    elif iv.op == "add":
        new = sub + iv.value
    else:  # pragma: no cover
        raise ValueError(iv.op)
    idx_p = pos.view(-1, 1).expand(-1, len(neu))
    idx_n = neu.view(1, -1).expand(len(pos), -1)
    out[:, idx_p, idx_n] = new
    return out


@dataclass
class VectorAdd:
    """Add coeff[t] * vector to the output of a site at layer L, for positions t.

    site="mlp_out":   added to the MLP output of layer L (same place a neuron writes a*w).
    site="resid_post": added to the residual stream after block L (used for steering).
    ``coeffs`` is either one float for all selected positions, or one per selected position.
    """
    layer: int
    vector: list[float] | torch.Tensor
    site: Literal["mlp_out", "resid_post"] = "mlp_out"
    positions: list[int] | None = None
    coeffs: float | list[float] = 1.0
    label: str = ""

    def to_dict(self) -> dict:
        v = self.vector.tolist() if isinstance(self.vector, torch.Tensor) else self.vector
        return {"layer": self.layer, "site": self.site, "positions": self.positions, "label": self.label,
                "vector_norm": float(torch.as_tensor(v).float().norm()), "coeffs": self.coeffs}

    def delta(self, seq_len: int, d_model: int, dtype, device) -> torch.Tensor:
        v = torch.as_tensor(self.vector, dtype=torch.float32)
        if v.shape != (d_model,):
            raise ValueError(f"vector must have shape [{d_model}], got {list(v.shape)}")
        pos = resolve_positions(self.positions, seq_len)
        c = torch.as_tensor(self.coeffs, dtype=torch.float32)
        if c.dim() == 0:
            c = c.expand(len(pos))
        if c.numel() != len(pos):
            raise ValueError(f"{c.numel()} coeffs for {len(pos)} positions")
        d = torch.zeros(seq_len, d_model, dtype=torch.float32)
        d[pos] = c.view(-1, 1) * v.view(1, -1)
        return d.to(device=device, dtype=dtype)


@dataclass
class InterventionPlan:
    neuron_interventions: list[NeuronIntervention] = field(default_factory=list)
    vector_adds: list[VectorAdd] = field(default_factory=list)

    def by_layer(self) -> dict[int, list[NeuronIntervention]]:
        d: dict[int, list[NeuronIntervention]] = {}
        for iv in self.neuron_interventions:
            d.setdefault(iv.layer, []).append(iv)
        return d

    def __bool__(self) -> bool:
        return bool(self.neuron_interventions or self.vector_adds)
