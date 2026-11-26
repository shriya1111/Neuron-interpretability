from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoadRequest(BaseModel):
    model_id: str = "Qwen/Qwen3-0.6B"
    device: str | None = Field(None, description="cuda | mps | cpu | auto")
    dtype: Literal["float32", "bfloat16", "float16"] | None = None
    revision: str | None = None
    seed: int = 0


class PromptInput(BaseModel):
    prompt: str = Field("", description="text prompt (ignored when token_ids is given)")
    token_ids: list[int] | None = Field(None, description="exact token ids, e.g. an Atlas firing context")
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


class LogitLensRequest(BaseModel):
    kind: Literal["neuron", "residual", "vector"]
    layer: int | None = None
    neuron: int | None = None
    prompt: str | None = None
    position: int = -1
    vector: list[float] | None = None
    top_k: int = Field(20, ge=1, le=200)
    target_tokens: list[str] = Field(default_factory=list)


class NeuronSearchRequest(BaseModel):
    target: str = Field(..., description="exact single-token string, e.g. ' John'")
    sign: Literal["+", "-"] = "+"
    top: int = Field(30, ge=1, le=500)
    layers: list[int] | None = None


class ContextIn(BaseModel):
    label: str = ""
    prompt: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)


class Type1Request(BaseModel):
    layer: int
    neuron: int
    sign: Literal["+", "-"]
    contexts: list[ContextIn] = Field(..., min_length=2)
    n_controls: int = Field(20, ge=2, le=500)
    seed: int = 0


class Type1CandidatesRequest(BaseModel):
    contexts: list[ContextIn] = Field(..., min_length=2)
    sign: Literal["+", "-"] | None = None
    top: int = Field(25, ge=1, le=200)


class WeightEditRequest(BaseModel):
    kind: Literal["prune", "scale_output", "scale_input", "set_output", "add_direction"]
    layer: int
    neurons: list[int] = Field(..., min_length=1)
    params: dict = Field(default_factory=dict, description=(
        "scale_output/scale_input: {factor, which:[gate,up]}; set_output: {vector}; "
        "add_direction: {alpha, alpha_relative_to_norm, direction:{source:'token'|'neuron'|'vector', token|layer,neuron|vector}}"))


class WeightRevertRequest(BaseModel):
    edit_id: int | None = None
    all: bool = False


class ComparePrompt(BaseModel):
    prompt: str = Field(..., min_length=1)
    targets: list[str] = Field(default_factory=list)


class WeightCompareRequest(BaseModel):
    prompts: list[ComparePrompt] = Field(default_factory=list)
    top_k: int = Field(10, ge=1, le=100)
    n_heldout: int = Field(0, ge=0, le=40)


class WeightExportRequest(BaseModel):
    name: str = Field(..., pattern=r"^[A-Za-z0-9_.-]{1,64}$")
