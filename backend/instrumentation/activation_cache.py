"""Instrumented forward passes.

``run_forward`` performs one real forward pass with optional capture hooks and
optional neuron interventions, and always removes every hook afterwards.

Capture points (per layer L, all for batch element 0, stored float32 on CPU):

* ``resid_pre``  input to block L (residual stream before the block)
* ``attn_out``   output of self-attention
* ``resid_mid``  resid_pre + attn_out (computed)
* ``mlp_in``     actual MLP input x = RMSNorm_post(resid_mid)  (pre-hook on mlp)
* ``gate_pre``   gate_proj(x)
* ``up``         up_proj(x)
* ``act``        activation entering down_proj, *before* any intervention
* ``act_mod``    activation entering down_proj, *after* interventions (only if intervened)
* ``mlp_out``    MLP output
* ``resid_post`` output of block L
Plus ``final_norm`` (normalized final residual) once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .hooks import hooks
from .interventions import InterventionPlan, apply_op, validate

ALL_POINTS = ("resid_pre", "attn_out", "resid_mid", "mlp_in", "gate_pre", "up", "act",
              "mlp_out", "resid_post")


@dataclass
class CaptureSpec:
    layers: list[int] | None = None  # None = all layers
    points: tuple[str, ...] = ALL_POINTS
    final: bool = True


@dataclass
class ForwardResult:
    input_ids: torch.Tensor  # [T]
    logits: torch.Tensor  # [T, V] float32 cpu
    cache: dict[tuple[str, int], torch.Tensor] = field(default_factory=dict)
    final_norm: torch.Tensor | None = None
    seconds: float = 0.0

    def get(self, point: str, layer: int) -> torch.Tensor:
        return self.cache[(point, layer)]


def _first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def _store(t: torch.Tensor) -> torch.Tensor:
    return t.detach()[0].to(torch.float32).cpu().clone()


def run_forward(handle, input_ids: torch.Tensor, capture: CaptureSpec | None = None,
                plan: InterventionPlan | None = None) -> ForwardResult:
    """One real forward pass. ``input_ids`` has shape [T] or [1, T]."""
    adapter = handle.adapter
    model = handle.model
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[0] != 1:
        raise ValueError("batch size must be 1")
    seq_len = input_ids.shape[1]
    plan = plan or InterventionPlan()
    for iv in plan.neuron_interventions:
        validate(iv, adapter, seq_len)
    by_layer = plan.by_layer()
    cache: dict[tuple[str, int], torch.Tensor] = {}
    holder: dict[str, torch.Tensor] = {}
    layers = range(adapter.n_layers) if capture is None or capture.layers is None else capture.layers
    points = set(capture.points) if capture else set()
    t0 = time.time()

    with handle.lock, torch.no_grad(), hooks() as hs:
        if capture:
            for L in layers:
                adapter.check_layer(L)
                blk, attn, mlp = adapter.block(L), adapter.attn(L), adapter.mlp(L)

                if points & {"resid_pre", "resid_mid"}:
                    def blk_pre(mod, args, kwargs, L=L):
                        h = args[0] if args else kwargs["hidden_states"]
                        cache[("resid_pre", L)] = _store(h)
                    hs.pre(blk, blk_pre, with_kwargs=True)
                if points & {"attn_out", "resid_mid"}:
                    hs.forward(attn, lambda m, i, o, L=L: cache.__setitem__(("attn_out", L), _store(_first(o))))
                if "mlp_in" in points:
                    hs.pre(mlp.module, lambda m, a, L=L: cache.__setitem__(("mlp_in", L), _store(a[0])))
                if "gate_pre" in points:
                    hs.forward(mlp.gate_proj, lambda m, i, o, L=L: cache.__setitem__(("gate_pre", L), _store(o)))
                if "up" in points:
                    hs.forward(mlp.up_proj, lambda m, i, o, L=L: cache.__setitem__(("up", L), _store(o)))
                if "mlp_out" in points:
                    hs.forward(mlp.module, lambda m, i, o, L=L: cache.__setitem__(("mlp_out", L), _store(_first(o))))
                if "resid_post" in points:
                    hs.forward(blk, lambda m, i, o, L=L: cache.__setitem__(("resid_post", L), _store(_first(o))))
            if capture.final:
                hs.forward(adapter.final_norm, lambda m, i, o: holder.__setitem__("final", _store(_first(o))))

        # down_proj pre-hook: capture original activation and apply interventions.
        act_layers = set(by_layer)
        if capture and "act" in points:
            act_layers |= set(layers)
        for L in act_layers:
            ivs = by_layer.get(L, [])
            want = capture is not None and "act" in points and L in set(layers)

            def down_pre(mod, args, L=L, ivs=ivs, want=want):
                a = args[0]
                if want:
                    cache[("act", L)] = _store(a)
                if not ivs:
                    return None
                new = a
                for iv in ivs:
                    new = apply_op(new, iv, seq_len)
                if want:
                    cache[("act_mod", L)] = _store(new)
                return (new,) + tuple(args[1:])

            hs.pre(adapter.mlp(L).down_proj, down_pre)

        out = model(input_ids=input_ids.to(handle.device), use_cache=False)
        logits = out.logits[0].to(torch.float32).cpu()

    for L in layers:
        if ("resid_pre", L) in cache and ("attn_out", L) in cache and "resid_mid" in points:
            cache[("resid_mid", L)] = cache[("resid_pre", L)] + cache[("attn_out", L)]
    if capture and "resid_pre" not in points:
        for L in layers:
            cache.pop(("resid_pre", L), None)
    if capture and "attn_out" not in points:
        for L in layers:
            cache.pop(("attn_out", L), None)
    return ForwardResult(input_ids=input_ids[0].cpu(), logits=logits, cache=cache,
                         final_norm=holder.get("final"), seconds=round(time.time() - t0, 4))


def run_plain(handle, input_ids: torch.Tensor) -> torch.Tensor:
    """Uninstrumented reference forward pass (no hooks at all)."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    with handle.lock, torch.no_grad():
        return handle.model(input_ids=input_ids.to(handle.device), use_cache=False).logits[0].to(torch.float32).cpu()


def run_attention(handle, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    """Attention probabilities of one layer, shape [n_heads, T, T] (eager attention)."""
    handle.adapter.check_layer(layer)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    with handle.lock, torch.no_grad():
        out = handle.model(input_ids=input_ids.to(handle.device), use_cache=False, output_attentions=True)
    if not out.attentions or out.attentions[layer] is None:
        raise ValueError("model did not return attention weights (needs eager attention)")
    return out.attentions[layer][0].to(torch.float32).cpu()
