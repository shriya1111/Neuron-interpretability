"""Fit a J-lens on Modal with Anthropic's reference implementation (github.com/anthropics/jacobian-lens).

Recipe matches the published Neuronpedia fits: Salesforce/wikitext wikitext-103-raw-v1 (train), texts truncated
to 2,000 characters, max_seq_len 128, skip the first 16 positions and the last, mean over prompts.

Usage (from the repo root, with a Modal token configured):

    NEURONLAB_GPU=A10G modal run cloud/modal_jlens.py --n-prompts 4 --tag timing          # quick cost/timing check
    # Long runs: deploy, then spawn. A spawned call runs independently of the local client
    # (killing a `modal run --detach` client can still cancel the input).
    NEURONLAB_GPU=A10G modal deploy cloud/modal_jlens.py
    python cloud/modal_jlens.py spawn --n-prompts 1000 --tag qwen3-0.6b-wikitext-n1000
    modal volume get neuronlab-jlens qwen3-0.6b-wikitext-n1000 data/jlens/

Progress is checkpointed to the Modal volume after every chunk, so a rerun with the same --tag resumes.
"""

from __future__ import annotations

import json
import os
import time

import modal

GPU = os.environ.get("NEURONLAB_GPU", "A10G")  # set before `modal run`, e.g. NEURONLAB_GPU=L40S
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
app = modal.App("neuronlab-jlens")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "transformers>=5.5", "datasets", "huggingface_hub", "numpy")
    .pip_install(f"git+https://github.com/anthropics/jacobian-lens@{JLENS_COMMIT}")
    .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_XET": "1"})
)
out_vol = modal.Volume.from_name("neuronlab-jlens", create_if_missing=True)
hf_vol = modal.Volume.from_name("neuronlab-hf-cache", create_if_missing=True)


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


@app.function(image=image, gpu=GPU, volumes={"/out": out_vol, "/hf": hf_vol}, timeout=10 * 3600)
def fit(model_id: str, n_prompts: int, tag: str, dim_batch: int, max_seq_len: int, dtype: str, chunk: int,
        gpu: str = GPU) -> dict:
    import logging
    import os

    import jlens
    import torch
    import transformers

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)
    os.makedirs(f"/out/{tag}", exist_ok=True)
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dt).to("cuda").eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    revision = getattr(hf.config, "_commit_hash", None)
    model = jlens.from_hf(hf, tok)
    prompts = wikitext_prompts(n_prompts)
    hf_vol.commit()

    ckpt = f"/out/{tag}/checkpoint.pt"
    t0 = time.time()
    lens = None
    for end in range(chunk, len(prompts) + chunk, chunk):
        lens = jlens.fit(model, prompts[: min(end, len(prompts))], dim_batch=dim_batch, max_seq_len=max_seq_len,
                         checkpoint_path=ckpt, checkpoint_every=chunk, resume=True)
        out_vol.commit()
        print(f"[neuronlab] {min(end, len(prompts))}/{len(prompts)} prompts, {time.time() - t0:.0f}s", flush=True)
    elapsed = time.time() - t0

    lens.save(f"/out/{tag}/jacobian_lens.pt", dtype=torch.float32)
    meta = {
        "model_id": model_id, "model_revision": revision, "n_prompts_requested": n_prompts,
        "n_prompts_fitted": lens.n_prompts, "dataset": "Salesforce/wikitext wikitext-103-raw-v1 train",
        "prompt_rule": "first rows with >=500 chars (stripped), truncated to 2000 chars", "max_seq_len": max_seq_len,
        "skip_first_positions": 16, "dim_batch": dim_batch, "dtype": dtype, "gpu": gpu,
        "estimator": "jlens.fit (sum over target positions t'>=t, mean over source positions and prompts)",
        "jlens_commit": JLENS_COMMIT, "torch": torch.__version__, "transformers": transformers.__version__,
        "fit_seconds": round(elapsed, 1), "seconds_per_prompt": round(elapsed / max(1, lens.n_prompts), 2),
    }
    with open(f"/out/{tag}/config.json", "w") as f:
        json.dump(meta, f, indent=2)
    out_vol.commit()
    return meta


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen3-0.6B", n_prompts: int = 1000, tag: str = "qwen3-0.6b-wikitext-n1000",
         dim_batch: int = 64, max_seq_len: int = 128, dtype: str = "float32", chunk: int = 25):
    meta = fit.remote(model_id, n_prompts, tag, dim_batch, max_seq_len, dtype, chunk, GPU)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    # python cloud/modal_jlens.py spawn [--n-prompts N] [--tag T] [--dim-batch B] [--chunk C]
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["spawn"])
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--tag", default="qwen3-0.6b-wikitext-n1000")
    ap.add_argument("--dim-batch", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--chunk", type=int, default=25)
    a = ap.parse_args()
    f = modal.Function.from_name("neuronlab-jlens", "fit")
    call = f.spawn(a.model_id, a.n_prompts, a.tag, a.dim_batch, a.max_seq_len, a.dtype, a.chunk, GPU)
    print("spawned function call:", call.object_id)
