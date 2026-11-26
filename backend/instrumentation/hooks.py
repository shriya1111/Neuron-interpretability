"""Hook registration with guaranteed cleanup.

Every hook NeuronLab registers is tagged, so leak checks count only our hooks.
Transformers >= 5 installs its own persistent "output capturing" hooks the first
time ``output_attentions`` is requested; those are reported separately by
``foreign_hooks`` and are verified by tests not to change model outputs.
"""

from __future__ import annotations

from contextlib import contextmanager

from torch import nn

_TAG = "_neuronlab_hook"


def _tag(fn):
    def wrapped(*a, **k):
        return fn(*a, **k)
    setattr(wrapped, _TAG, True)
    return wrapped


class HookSet:
    """Collects hook handles and removes all of them on exit (also on exceptions)."""

    def __init__(self) -> None:
        self._handles = []

    def forward(self, module: nn.Module, fn) -> None:
        self._handles.append(module.register_forward_hook(_tag(fn)))

    def pre(self, module: nn.Module, fn, with_kwargs: bool = False) -> None:
        self._handles.append(module.register_forward_pre_hook(_tag(fn), with_kwargs=with_kwargs))

    def remove(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    def __len__(self) -> int:
        return len(self._handles)


@contextmanager
def hooks():
    hs = HookSet()
    try:
        yield hs
    finally:
        hs.remove()


def _all_hooks(model: nn.Module):
    for name, m in model.named_modules():
        for d in (m._forward_hooks, m._forward_pre_hooks, m._backward_hooks):
            for fn in d.values():
                yield name, fn


def count_hooks(model: nn.Module) -> int:
    """Number of NeuronLab-registered hooks still attached (must be 0 between operations)."""
    return sum(1 for _, fn in _all_hooks(model) if getattr(fn, _TAG, False))


def foreign_hooks(model: nn.Module) -> list[str]:
    """Hooks registered by other libraries (e.g. Transformers output capturing)."""
    return sorted({f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', fn)}"
                   for _, fn in _all_hooks(model) if not getattr(fn, _TAG, False)})
