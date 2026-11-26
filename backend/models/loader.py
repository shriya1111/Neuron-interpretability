"""Model loading with full provenance recording.

Only one model is resident at a time. All access that registers hooks or runs
forward passes must hold ``ModelHandle.lock``: hooks are module-level state and
must never leak between concurrent requests.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import torch
import transformers

from .adapters import ModelAdapter

# Plain HTTPS downloads; some networks block the Xet CAS transfer host.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def pick_device(requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> str:
    # fp32 on CPU keeps the reconstruction checks tight; bf16 on accelerators.
    return "float32" if device == "cpu" else "bfloat16"


def set_deterministic(seed: int = 0) -> None:
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass
class ModelHandle:
    model_id: str
    model: torch.nn.Module
    tokenizer: transformers.PreTrainedTokenizerBase
    adapter: ModelAdapter
    device: str
    dtype: str
    revision: str | None
    load_seconds: float
    lock: threading.RLock = field(default_factory=threading.RLock)

    def provenance(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_revision": self.revision,
            "model_type": self.adapter.model_type,
            "dtype": self.dtype,
            "quantization": None,
            "device": self.device,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_vocab_size": len(self.tokenizer),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }


def load_model(model_id: str, *, device: str | None = None, dtype: str | None = None,
               revision: str | None = None, token: str | None = None, seed: int = 0) -> ModelHandle:
    set_deterministic(seed)
    device = pick_device(device)
    dtype = dtype or default_dtype(device)
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    token = token or os.environ.get("HF_TOKEN")
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, token=token, dtype=_DTYPES[dtype],
        attn_implementation="eager",  # deterministic, hook-friendly reference attention
    )
    return wrap_model(model_id, model, tok, device=device, dtype=dtype, t0=t0,
                      revision=resolve_revision(model_id, revision, model))


def resolve_revision(model_id: str, revision: str | None, model) -> str | None:
    """Commit hash of the checkpoint actually loaded (from the HF cache snapshot path)."""
    h = getattr(model.config, "_commit_hash", None)
    if h:
        return h
    if os.path.isdir(model_id):
        return f"local:{os.path.abspath(model_id)}"
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(model_id, "config.json", revision=revision or "main")
        if isinstance(p, str) and "/snapshots/" in p:
            return p.split("/snapshots/")[1].split("/")[0]
    except Exception:
        pass
    return revision


def wrap_model(model_id: str, model: torch.nn.Module, tokenizer, *, device: str = "cpu",
               dtype: str = "float32", t0: float | None = None, revision: str | None = None) -> ModelHandle:
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    adapter = ModelAdapter(model)
    return ModelHandle(model_id=model_id, model=model, tokenizer=tokenizer, adapter=adapter,
                       device=device, dtype=dtype, revision=revision,
                       load_seconds=round(time.time() - (t0 or time.time()), 3))


class ModelManager:
    """Holds the currently loaded model."""

    def __init__(self) -> None:
        self._handle: ModelHandle | None = None
        self._lock = threading.Lock()

    def load(self, model_id: str, **kw) -> ModelHandle:
        with self._lock:
            old = self._handle
            self._handle = None
            del old
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._handle = load_model(model_id, **kw)
            return self._handle

    def set(self, handle: ModelHandle) -> None:
        with self._lock:
            self._handle = handle

    def get(self) -> ModelHandle:
        if self._handle is None:
            raise RuntimeError("no model loaded; POST /models/load first")
        return self._handle

    @property
    def loaded(self) -> bool:
        return self._handle is not None


manager = ModelManager()
