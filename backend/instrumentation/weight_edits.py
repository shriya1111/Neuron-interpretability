"""Permanent (until reverted) weight edits on MLP neurons.

Unlike activation interventions (which exist only during one forward pass), these change the
model's parameters in memory, so *every* later request sees the edited model. To keep this
explicit and safe:

* every edit stores the exact original values of the slices it touches, so reverting restores
  the weights bit-for-bit;
* the list of active edits is attached to the provenance of every API response;
* ``original_weights()`` temporarily swaps the originals back in (under the model lock) so the
  edited and original models can be compared on the same inputs;
* edits never touch the files on disk; ``export`` writes a new copy only when asked.

Edit kinds (neuron i in layer L; g = gate_proj row, u = up_proj row, w = down_proj column):

  prune           g_i, u_i, w_i (and their biases) ← 0          → a_i ≡ 0 and contribution ≡ 0
  scale_output    w_i ← s · w_i                                    → contribution scaled by s
  scale_input     g_i ← s · g_i  and/or  u_i ← s · u_i             → changes when / how much it fires
  set_output      w_i ← v                                          → writes a different direction
  add_direction   w_i ← w_i + α · unit(d)                          → d = token direction u_t, another
                                                                     neuron's w, or a raw vector
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

KINDS = ("prune", "scale_output", "scale_input", "set_output", "add_direction")


@dataclass
class _Slice:
    param: torch.nn.Parameter
    name: str
    index: tuple  # (slice/int, slice/int)
    original: torch.Tensor


@dataclass
class WeightEdit:
    id: int
    kind: str
    layer: int
    neurons: list[int]
    params: dict
    created_utc: str
    slices: list[_Slice] = field(default_factory=list, repr=False)
    checksum_before: str = ""
    checksum_after: str = ""

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "layer": self.layer, "neurons": self.neurons,
                "params": self.params, "created_utc": self.created_utc,
                "touches": sorted({s.name for s in self.slices}),
                "checksum_before": self.checksum_before, "checksum_after": self.checksum_after}


def _checksum(slices: list[_Slice]) -> str:
    h = hashlib.sha256()
    for s in slices:
        h.update(s.param.data[s.index].detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class WeightEditor:
    def __init__(self, handle):
        self.handle = handle
        self.edits: list[WeightEdit] = []
        self._next = 1

    # ------------------------------------------------------------------ helpers
    def _slices_for(self, layer: int, neurons: list[int], which: tuple[str, ...]) -> list[_Slice]:
        mlp = self.handle.adapter.mlp(layer)
        out = []
        for n in neurons:
            for nm in which:
                if nm == "gate":
                    out.append((mlp.gate_proj.weight, f"L{layer}.gate_proj.weight[{n},:]", (n, slice(None))))
                    if mlp.gate_proj.bias is not None:
                        out.append((mlp.gate_proj.bias, f"L{layer}.gate_proj.bias[{n}]", (n,)))
                elif nm == "up":
                    out.append((mlp.up_proj.weight, f"L{layer}.up_proj.weight[{n},:]", (n, slice(None))))
                    if mlp.up_proj.bias is not None:
                        out.append((mlp.up_proj.bias, f"L{layer}.up_proj.bias[{n}]", (n,)))
                elif nm == "down":
                    out.append((mlp.down_proj.weight, f"L{layer}.down_proj.weight[:,{n}]", (slice(None), n)))
        return [_Slice(p, name, idx, p.data[idx].detach().clone()) for p, name, idx in out]

    def _direction(self, spec: dict) -> torch.Tensor:
        ad = self.handle.adapter
        src = spec.get("source")
        if src == "token":
            from interpretability.logit_lens import token_direction
            ids = self.handle.tokenizer.encode(spec["token"], add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"direction token {spec['token']!r} must be a single token, got {len(ids)}")
            return token_direction(ad, ids[0])
        if src == "neuron":
            return ad.output_vector(int(spec["layer"]), int(spec["neuron"])).detach().float().cpu()
        if src == "vector":
            v = torch.tensor(spec["vector"], dtype=torch.float32)
            if v.shape != (ad.d_model,):
                raise ValueError(f"vector must have length {ad.d_model}")
            return v
        raise ValueError("direction.source must be 'token', 'neuron' or 'vector'")

    def _mutate(self, kind, mlp, neurons, which, params, slices, dt) -> None:
        ad = self.handle.adapter
        for n in neurons:
            if kind == "prune":
                for s in slices:
                    s.param.data[s.index] = 0
            elif kind == "scale_output":
                mlp.down_proj.weight.data[:, n] *= float(params["factor"])
            elif kind == "scale_input":
                if "gate" in which:
                    mlp.gate_proj.weight.data[n, :] *= float(params["factor"])
                if "up" in which:
                    mlp.up_proj.weight.data[n, :] *= float(params["factor"])
            elif kind == "set_output":
                v = torch.tensor(params["vector"], dtype=torch.float32)
                if v.shape != (ad.d_model,):
                    raise ValueError(f"set_output.vector must have length {ad.d_model}")
                mlp.down_proj.weight.data[:, n] = v.to(dt)
            elif kind == "add_direction":
                d = self._direction(params.get("direction", {}))
                d = d / d.norm().clamp_min(1e-12)
                alpha = float(params.get("alpha", 1.0))
                if params.get("alpha_relative_to_norm", False):
                    alpha = alpha * float(mlp.down_proj.weight.data[:, n].float().norm())
                mlp.down_proj.weight.data[:, n] += (alpha * d).to(dt)

    # ------------------------------------------------------------------ apply / revert
    def apply(self, kind: str, layer: int, neurons: list[int], params: dict | None = None) -> WeightEdit:
        params = dict(params or {})
        ad = self.handle.adapter
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not neurons:
            raise ValueError("at least one neuron required")
        neurons = sorted(set(int(n) for n in neurons))
        for n in neurons:
            ad.check_neuron(layer, n)

        if kind == "prune":
            which = ("gate", "up", "down")
        elif kind == "scale_input":
            which = tuple(params.get("which", ["gate", "up"]))
            if not set(which) <= {"gate", "up"} or not which:
                raise ValueError("scale_input.which must be a subset of ['gate', 'up']")
        else:
            which = ("down",)
        if kind in ("scale_output", "scale_input") and "factor" not in params:
            raise ValueError(f"{kind} needs params.factor")

        with self.handle.lock, torch.no_grad():
            slices = self._slices_for(layer, neurons, which)
            before = _checksum(slices)
            mlp = ad.mlp(layer)
            dt = mlp.down_proj.weight.dtype
            try:
                self._mutate(kind, mlp, neurons, which, params, slices, dt)
            except Exception:
                for s in reversed(slices):  # all-or-nothing
                    s.param.data[s.index] = s.original
                raise
            stored_params = {k: v for k, v in params.items() if k != "vector"}
            if "vector" in params:
                stored_params["vector_norm"] = float(torch.tensor(params["vector"]).float().norm())
            e = WeightEdit(self._next, kind, layer, neurons, stored_params,
                           _dt.datetime.now(_dt.timezone.utc).isoformat(), slices, before, _checksum(slices))
            self._next += 1
            self.edits.append(e)
        return e

    def _overlaps_later(self, e: WeightEdit) -> list[int]:
        names = {s.name for s in e.slices}
        return [x.id for x in self.edits if x.id > e.id and names & {s.name for s in x.slices}]

    def revert(self, edit_id: int) -> WeightEdit:
        e = next((x for x in self.edits if x.id == edit_id), None)
        if e is None:
            raise KeyError(f"no active edit with id {edit_id}")
        later = self._overlaps_later(e)
        if later:
            raise ValueError(f"edit {edit_id} touches weights also changed by later edits {later}; revert those first")
        with self.handle.lock, torch.no_grad():
            for s in reversed(e.slices):
                s.param.data[s.index] = s.original
            self.edits.remove(e)
        return e

    def revert_all(self) -> int:
        n = len(self.edits)
        with self.handle.lock, torch.no_grad():
            for e in reversed(self.edits):
                for s in reversed(e.slices):
                    s.param.data[s.index] = s.original
            self.edits.clear()
        return n

    @contextmanager
    def original_weights(self):
        """Temporarily restore the original weights (caller must already hold the model lock or not need it)."""
        with self.handle.lock, torch.no_grad():
            current = []
            for e in reversed(self.edits):
                for s in reversed(e.slices):
                    current.append((s, s.param.data[s.index].clone()))
                    s.param.data[s.index] = s.original
            try:
                yield
            finally:
                for s, val in reversed(current):
                    s.param.data[s.index] = val

    @property
    def modified(self) -> bool:
        return bool(self.edits)

    def summary(self) -> list[dict]:
        return [e.summary() for e in self.edits]


def get_editor(handle) -> WeightEditor:
    ed = getattr(handle, "_weight_editor", None)
    if ed is None:
        ed = WeightEditor(handle)
        handle._weight_editor = ed
    return ed
