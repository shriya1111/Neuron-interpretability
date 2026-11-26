from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoadRequest(BaseModel):
    model_id: str = "Qwen/Qwen2.5-0.5B"
    device: str | None = Field(None, description="cuda | mps | cpu | auto")
    dtype: Literal["float32", "bfloat16", "float16"] | None = None
    revision: str | None = None
    seed: int = 0


class PromptInput(BaseModel):
    prompt: str = Field(..., min_length=1)
    add_special_tokens: bool = True


class InferenceRequest(PromptInput):
    position: int = -1
    top_k: int = Field(10, ge=1, le=200)
    layer_summary: bool = True
    top_neurons_per_layer: int = Field(8, ge=0, le=64)
    attention_layer: int | None = Field(None, description="if set, return that layer's per-head attention pattern")


class CaptureRequest(PromptInput):
    layer: int
    point: Literal["act", "gate_pre", "up"] = "act"
    neuron_start: int = 0
    neuron_count: int = Field(64, ge=1, le=2048)
    order: Literal["index", "max_abs"] = "index"


class InspectRequest(PromptInput):
    layer: int
    neuron: int
    position: int = -1
    top_n: int = Field(20, ge=1, le=200)
    include_weights: bool = True


class NeuronInterventionIn(BaseModel):
    layer: int
    neurons: list[int] = Field(..., min_length=1)
    op: Literal["zero", "scale", "negate", "set", "add"]
    value: float = 0.0
    positions: list[int] | None = None
    label: str = ""


class InterventionRequest(PromptInput):
    interventions: list[NeuronInterventionIn] = Field(..., min_length=1)
    position: int = -1
    top_k: int = Field(10, ge=1, le=200)
    target_tokens: list[str] = Field(default_factory=list,
                                     description="exact strings; each must encode to a single token (e.g. ' John')")
    seed: int = 0
