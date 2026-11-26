import os

import pytest
import torch
import transformers

from models.loader import wrap_model

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
REAL_MODEL = os.environ.get("NEURONLAB_TEST_MODEL", "Qwen/Qwen2.5-0.5B")


def _tiny(kind: str, **extra):
    torch.manual_seed(0)
    common = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=2, vocab_size=512, max_position_embeddings=128, rms_norm_eps=1e-6,
                  hidden_act="silu", attn_implementation="eager")
    common.update(extra)
    if kind == "llama":
        cfg = transformers.LlamaConfig(**common)
    elif kind == "qwen2":
        cfg = transformers.Qwen2Config(**common)
    elif kind == "qwen3":
        cfg = transformers.Qwen3Config(head_dim=16, **common)
    else:
        raise ValueError(kind)
    model = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    # Random init gives tiny activations; scale weights up so nonlinearity regimes are exercised.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 2:
                p.normal_(0, 0.2)
            elif "bias" in n:
                p.normal_(0, 0.1)
    return wrap_model(f"tiny-{kind}", model, None)


@pytest.fixture(params=["llama", "qwen2", "qwen3"])
def tiny(request):
    return _tiny(request.param)


@pytest.fixture
def tiny_llama_bias():
    return _tiny("llama", mlp_bias=True)


@pytest.fixture
def ids():
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 512, (12,), generator=g)


def _real_available() -> bool:
    try:
        transformers.AutoConfig.from_pretrained(REAL_MODEL, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def real():
    if not _real_available():
        pytest.skip(f"{REAL_MODEL} not cached")
    from models.loader import load_model
    return load_model(REAL_MODEL, device="cpu", dtype="float32")
