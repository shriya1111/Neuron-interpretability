"""J-lens (Gurnee et al. 2026) as used by the neuron-basis paper (Appendix A.1).

Definitions (layer l = residual stream at the *output* of block l, as in the reference `jlens` code):

    J_l      = E_{prompts, t, t'>=t} [ d h_final,t' / d h_l,t ]          [d, d]   (fitted; loaded from file)
    D_l      = (W_U diag(gamma_final)) J_l                                  [V, d]
    d_{l,t}  = D_l[t] / ||D_l[t]||                                          J-lens direction of token t
    readout(v) at layer l = tokens ranked by cos(d_{l,t}, v)

A neuron in block l writes a*w into the output of block l, so its output vector is read with J_l.
The last block's output *is* the final residual, so the last layer uses the unembedding (J = I), as the paper says.

Provenance kinds that must never be confused:
    "fitted_corpus_mean"  — this file: J averaged over a corpus (exact estimator of the reference code)
    "context_jacobian"    — d h_final / d h_l for one specific prompt (see `context_logit_derivative`)
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict

import torch

DEFAULT_LENS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "jlens",
                                            "qwen3-0.6b-wikitext-n1000"))


class JLens:
    def __init__(self, adapter, path: str = DEFAULT_LENS, cache_layers: int = 3, jacobians: dict | None = None,
                 n_valid_tokens: int | None = None):
        self.adapter = adapter
        # Rows of W_U beyond the tokenizer's vocabulary are unused padding, not tokens: excluded everywhere.
        self.n_valid = n_valid_tokens if n_valid_tokens is not None else adapter.vocab_size
        self.path = path
        if jacobians is None:
            f = os.path.join(path, "jacobian_lens.pt")
            if not os.path.exists(f):
                raise FileNotFoundError(f"no J-lens at {f}")
            import jlens  # Anthropic's reference implementation defines the file format
            jacobians = jlens.JacobianLens.load(f).jacobians
        self.jacobians = {int(k): v.float() for k, v in jacobians.items()}
        d = adapter.d_model
        for k, v in self.jacobians.items():
            if tuple(v.shape) != (d, d):
                raise ValueError(f"lens layer {k} has shape {tuple(v.shape)}, model d_model={d}")
        cfg = os.path.join(path or "", "config.json")
        self.config = json.load(open(cfg)) if os.path.exists(cfg) else {}
        self.final_layer = adapter.n_layers - 1
        self._dirs: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._cache_layers = cache_layers
        W = adapter.unembed.weight.detach().float().cpu()
        g = adapter.final_norm.weight.detach().float().cpu()
        self.WUg = W * g  # [V, d]  (W_U with the final-norm gain folded in)

    @property
    def provenance(self) -> dict:
        return {"kind": "fitted_corpus_mean", "path": self.path, **{k: self.config.get(k) for k in
                ("model_id", "n_prompts_fitted", "dataset", "max_seq_len", "skip_first_positions", "jlens_commit", "dtype")}}

    def check_layer(self, layer: int) -> None:
        if layer != self.final_layer and layer not in self.jacobians:
            raise ValueError(f"no J-lens for layer {layer}; fitted layers {min(self.jacobians)}–{max(self.jacobians)} "
                             f"(+ {self.final_layer} uses the unembedding)")

    def directions(self, layer: int) -> torch.Tensor:
        """Unit J-lens token directions at `layer`: [V, d] (cached for a few layers; ~0.6 GB each in fp32)."""
        self.check_layer(layer)
        if layer in self._dirs:
            self._dirs.move_to_end(layer)
            return self._dirs[layer]
        D = self.WUg if layer == self.final_layer else self.WUg @ self.jacobians[layer]
        D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if self.n_valid < D.shape[0]:
            D = D.clone()
            D[self.n_valid:] = 0.0  # padding rows: never selected (cosine 0, excluded from readouts below)
        self._dirs[layer] = D
        while len(self._dirs) > self._cache_layers:
            self._dirs.popitem(last=False)
        return D

    def readout(self, tokenizer, layer: int, v: torch.Tensor, k: int = 20, targets: list[int] | None = None) -> dict:
        from evaluation import metrics
        D = self.directions(layer)
        v = v.detach().float().cpu()
        cos = D @ (v / v.norm().clamp_min(1e-12))

        def side(sg: float) -> dict:
            c = sg * cos
            c[self.n_valid:] = -float("inf")
            top = c.topk(k)
            return {"top": [dict(metrics.token_info(tokenizer, i), cosine=float(c[i]), rank=r + 1)
                            for r, i in enumerate(top.indices.tolist())],
                    "targets": [dict(metrics.token_info(tokenizer, t), cosine=float(c[t]),
                                     rank=int((c > c[t]).sum()) + 1) for t in (targets or [])]}
        return {"method": "cos(d_{l,t}, v), d = rows of (W_U·diag(γ_final))·J_l normalized" +
                          (" (last layer: J = I)" if layer == self.final_layer else ""),
                "layer": layer, "plus": side(1.0), "minus": side(-1.0), "lens": self.provenance}

    def transport_logits(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Lens logits for a full residual state: W_U · RMSNorm_final(J_l h) (the reference `apply`)."""
        ad = self.adapter
        x = h.float() if layer == self.final_layer else self.jacobians[layer] @ h.float()
        p = next(ad.model.parameters())
        with torch.no_grad():
            return ad.unembed(ad.final_norm(x.to(p.device, p.dtype).unsqueeze(0)))[0].float().cpu()


_LENSES: dict[tuple[int, str], JLens] = {}


def get_lens(handle, path: str | None = None) -> JLens:
    path = path or os.environ.get("NEURONLAB_JLENS", DEFAULT_LENS)
    key = (id(handle.model), path)
    if key not in _LENSES:
        lens = JLens(handle.adapter, path, n_valid_tokens=len(handle.tokenizer))
        mid = lens.config.get("model_id")
        if mid and mid != handle.model_id:
            raise ValueError(f"J-lens was fitted for {mid}, but {handle.model_id} is loaded")
        _LENSES.clear()
        _LENSES[key] = lens
    return _LENSES[key]
