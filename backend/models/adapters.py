"""Architecture adapters.

An adapter resolves the submodules NeuronLab instruments (embeddings, blocks,
norms, attention projections, SwiGLU MLP projections, unembedding) for a loaded
Hugging Face causal LM, and validates that the MLP really is a SwiGLU MLP of the
form ``down(silu(gate(x)) * up(x))``. Dimensions are always read from the
loaded modules, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# model_type -> notes. All of these use model.model.layers[i].{input_layernorm,
# self_attn, post_attention_layernorm, mlp.{gate,up,down}_proj}, model.model.norm
# and model.lm_head in Hugging Face Transformers.
SUPPORTED_MODEL_TYPES = {
    "llama": "Llama / Llama-3.x (optional mlp_bias)",
    "qwen2": "Qwen2 / Qwen2.5 (q/k/v bias, tied embeddings on small sizes)",
    "qwen3": "Qwen3 dense (q/k RMSNorm)",
    "mistral": "Mistral dense",
}


class UnsupportedModelError(ValueError):
    pass


@dataclass
class MLPHandles:
    module: nn.Module
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear
    act_fn: nn.Module


class ModelAdapter:
    """Uniform access to the parts of a SwiGLU decoder-only transformer."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.config = model.config
        self.model_type = getattr(self.config, "model_type", None)
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            raise UnsupportedModelError(
                f"model_type={self.model_type!r} is not supported; supported: "
                f"{sorted(SUPPORTED_MODEL_TYPES)}"
            )
        self.base = model.model
        self.layers = list(self.base.layers)
        self._validate()

    # ---- validation -------------------------------------------------------
    def _validate(self) -> None:
        for i, layer in enumerate(self.layers):
            mlp = layer.mlp
            for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
                if not hasattr(mlp, name):
                    raise UnsupportedModelError(f"layer {i} MLP has no {name}; not a SwiGLU MLP")
            act = mlp.act_fn
            # Verify numerically that act_fn is SiLU, rather than trusting class names.
            probe = torch.linspace(-6, 6, 97, dtype=torch.float32)
            with torch.no_grad():
                got = act(probe)
            if not torch.allclose(got, torch.nn.functional.silu(probe), atol=1e-6):
                raise UnsupportedModelError(f"layer {i} MLP activation is not SiLU")
            if mlp.gate_proj.weight.shape != mlp.up_proj.weight.shape:
                raise UnsupportedModelError(f"layer {i}: gate/up projection shapes differ")
            if mlp.down_proj.weight.shape[::-1] != mlp.gate_proj.weight.shape:
                raise UnsupportedModelError(f"layer {i}: down projection shape mismatch")

    # ---- dimensions -------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_model(self) -> int:
        return self.layers[0].mlp.gate_proj.weight.shape[1]

    def d_mlp(self, layer: int) -> int:
        return self.mlp(layer).gate_proj.weight.shape[0]

    @property
    def vocab_size(self) -> int:
        return self.unembed.weight.shape[0]

    @property
    def n_heads(self) -> int:
        return int(self.config.num_attention_heads)

    @property
    def n_kv_heads(self) -> int:
        return int(getattr(self.config, "num_key_value_heads", None) or self.n_heads)

    @property
    def head_dim(self) -> int:
        hd = getattr(self.config, "head_dim", None)
        if hd:
            return int(hd)
        return self.attn(0).q_proj.weight.shape[0] // self.n_heads

    # ---- modules ----------------------------------------------------------
    def block(self, layer: int) -> nn.Module:
        return self.layers[layer]

    def mlp(self, layer: int) -> MLPHandles:
        m = self.layers[layer].mlp
        return MLPHandles(m, m.gate_proj, m.up_proj, m.down_proj, m.act_fn)

    def attn(self, layer: int) -> nn.Module:
        return self.layers[layer].self_attn

    def input_norm(self, layer: int) -> nn.Module:
        return self.layers[layer].input_layernorm

    def mlp_norm(self, layer: int) -> nn.Module:
        return self.layers[layer].post_attention_layernorm

    @property
    def final_norm(self) -> nn.Module:
        return self.base.norm

    @property
    def embed(self) -> nn.Embedding:
        return self.base.embed_tokens

    @property
    def unembed(self) -> nn.Linear:
        return self.model.lm_head

    @property
    def tied_embeddings(self) -> bool:
        return self.unembed.weight.data_ptr() == self.embed.weight.data_ptr()

    def has_mlp_bias(self, layer: int) -> bool:
        return self.mlp(layer).down_proj.bias is not None

    # ---- validation helpers ----------------------------------------------
    def check_layer(self, layer: int) -> int:
        if not isinstance(layer, int) or not (0 <= layer < self.n_layers):
            raise ValueError(f"layer must be an int in [0, {self.n_layers - 1}], got {layer!r}")
        return layer

    def check_neuron(self, layer: int, neuron: int) -> int:
        self.check_layer(layer)
        n = self.d_mlp(layer)
        if not isinstance(neuron, int) or not (0 <= neuron < n):
            raise ValueError(f"neuron must be an int in [0, {n - 1}] for layer {layer}, got {neuron!r}")
        return neuron

    def output_vector(self, layer: int, neuron: int) -> torch.Tensor:
        """w = down_proj.weight[:, neuron] (shape [d_model])."""
        self.check_neuron(layer, neuron)
        return self.mlp(layer).down_proj.weight[:, neuron]
