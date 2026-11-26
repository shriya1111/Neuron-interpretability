"""Model presets and paper-reported neuron IDs.

Neuron IDs from the paper are only meaningful for the exact checkpoint they were
reported on. ``paper_neuron`` refuses to hand one out for any other model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

MODEL_PRESETS = {
    "Qwen/Qwen2.5-0.5B": {"mode": "development", "notes": "Default dev model. Tied embeddings."},
    "Qwen/Qwen3-0.6B": {"mode": "development", "notes": "Dev model. Tied embeddings, q/k norm."},
    "meta-llama/Llama-3.1-8B-Instruct": {
        "mode": "paper_reproduction",
        "notes": "Paper's model. Gated: needs an HF token with the Llama 3.1 license accepted. ~16 GB in bf16.",
    },
    "Qwen/Qwen3-8B": {"mode": "paper_reproduction", "notes": "Also studied in the paper (later phase)."},
}


@dataclass(frozen=True)
class PaperNeuron:
    model_id: str
    layer: int
    neuron: int
    sign: str  # "+", "-", or "unstated" when the paper does not give it
    label: str
    source: str


PAPER_NEURONS: dict[str, PaperNeuron] = {
    "john": PaperNeuron("meta-llama/Llama-3.1-8B-Instruct", 28, 2760, "-",
                        "John (John the Ripper; author of the fourth Gospel) — Type 1", "Figure 1"),
    "musk_protein": PaperNeuron("meta-llama/Llama-3.1-8B-Instruct", 20, 2287, "-",
                                "Elon Musk + protein translation — Type 3", "Table 1 / B1"),
    "nineteenth_century": PaperNeuron("meta-llama/Llama-3.1-8B-Instruct", 22, 1766, "unstated",
                                      "19th century (readout Karl, Marx, Victorian...)", "Main text"),
    "calls": PaperNeuron("meta-llama/Llama-3.1-8B-Instruct", 22, 5208, "unstated",
                         "call/calls/_call", "Appendix A.2"),
    "salads": PaperNeuron("meta-llama/Llama-3.1-8B-Instruct", 4, 4934, "unstated",
                          "salads/lettuce", "Appendix A.2"),
    "hearing_shoes": PaperNeuron("meta-llama/Llama-3.1-8B-Instruct", 16, 60, "-", "hearing + shoes — Type 3", "Table B1"),
    "hip_eyeglasses": PaperNeuron("meta-llama/Llama-3.1-8B-Instruct", 21, 2095, "-", "hip replacement + eyeglasses", "Table B1"),
    "liverpool_tampa": PaperNeuron("meta-llama/Llama-3.1-8B-Instruct", 20, 7696, "-", "Liverpool + Tampa", "Table B1"),
    "singapore_flutter": PaperNeuron("Qwen/Qwen3-8B", 25, 5728, "-", "Singapore + Flutter — Type 3", "Table 1 / B1"),
}


class CheckpointMismatchError(ValueError):
    pass


def paper_neuron(key: str, loaded_model_id: str) -> PaperNeuron:
    if key not in PAPER_NEURONS:
        raise KeyError(f"unknown paper neuron {key!r}; known: {sorted(PAPER_NEURONS)}")
    pn = PAPER_NEURONS[key]
    if loaded_model_id != pn.model_id:
        raise CheckpointMismatchError(
            f"paper neuron {key!r} (L{pn.layer} n{pn.neuron}) was reported for {pn.model_id}; "
            f"the loaded model is {loaded_model_id}. Neuron IDs do not transfer between checkpoints."
        )
    return pn


def list_paper_neurons() -> list[dict]:
    return [dict(key=k, **asdict(v)) for k, v in PAPER_NEURONS.items()]
