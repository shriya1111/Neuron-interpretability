"""Fit a J-lens on CPU with Anthropic's reference implementation (same recipe as cloud/modal_jlens.py).

    python cloud/cpu_fit_jlens.py --n-prompts 200 --tag qwen3-0.6b-wikitext-cpu

Writes data/jlens/<tag>/checkpoint.pt every --checkpoint-every prompts (resumable: rerun the same
command to continue, or raise --n-prompts to extend), then jacobian_lens.pt and config.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import jlens  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


def wikitext_prompts(n: int, max_chars: int = 2000, min_chars: int = 500) -> list[str]:
    """Deterministic: the first n training rows with >= min_chars characters, truncated to max_chars."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    out = []
    for t in ds["text"]:
        if len(t.strip()) >= min_chars:
            out.append(t.strip()[:max_chars])
            if len(out) == n:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--tag", default="qwen3-0.6b-wikitext-cpu")
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out-root", default=os.path.join(os.path.dirname(__file__), "..", "data", "jlens"))
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    out = os.path.abspath(os.path.join(a.out_root, a.tag))
    os.makedirs(out, exist_ok=True)

    hf = transformers.AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.float32).eval()
    tok = transformers.AutoTokenizer.from_pretrained(a.model_id)
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(a.model_id, "config.json")
        revision = p.split("/snapshots/")[1].split("/")[0] if isinstance(p, str) else None
    except Exception:
        revision = None
    model = jlens.from_hf(hf, tok)
    prompts = wikitext_prompts(a.n_prompts)

    t0 = time.time()
    lens = jlens.fit(model, prompts, dim_batch=a.dim_batch, max_seq_len=a.max_seq_len,
                     checkpoint_path=os.path.join(out, "checkpoint.pt"), checkpoint_every=a.checkpoint_every, resume=True)
    elapsed = time.time() - t0
    lens.save(os.path.join(out, "jacobian_lens.pt"), dtype=torch.float32)
    meta = {
        "model_id": a.model_id, "model_revision": revision, "n_prompts_requested": a.n_prompts,
        "n_prompts_fitted": lens.n_prompts, "dataset": "Salesforce/wikitext wikitext-103-raw-v1 train",
        "prompt_rule": "first rows with >=500 chars (stripped), truncated to 2000 chars",
        "max_seq_len": a.max_seq_len, "skip_first_positions": 16, "dim_batch": a.dim_batch, "dtype": "float32",
        "device": f"cpu ({a.threads} threads)",
        "estimator": "jlens.fit (sum over target positions t'>=t, mean over source positions and prompts)",
        "jlens_commit": JLENS_COMMIT, "torch": torch.__version__, "transformers": transformers.__version__,
        "fit_seconds_this_session": round(elapsed, 1),
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
