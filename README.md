# NeuronLab

An interactive research instrument for reproducing, testing and extending
[*Returning to the Neuron Basis for Interpretability, Part 1*](https://alignment.anthropic.com/2026/neuron-basis-theory/)
(Anthropic, 2026). Every number shown comes from a real forward pass of a real model. There are
no mock activations and no hard-coded probabilities.

**Status: Phase 1 of 7 (neuron microscope) is built and tested.** The paper experiments
(logit lens / J-lens / J-share, polysemanticity, cross-layer steering, factual recall) are
planned in [`docs/PLAN.md`](docs/PLAN.md) and appear in the UI as clearly marked "not implemented" pages.

| Doc | Contents |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | Technical plan, with every implementation-relevant detail from the paper and Appendices A–D |
| [`docs/TEST_RESULTS.md`](docs/TEST_RESULTS.md) | Automated correctness test results per phase |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | Differences from the paper, approximations, open issues |

## What Phase 1 does

* **Model Explorer:** the actual loaded architecture, with dimensions read from the checkpoint:
  * tokenization with visible whitespace
  * a clickable block graph annotated with real residual, attention and MLP norms
  * a layer × token heatmap
  * block internals (q/k/v/o, gate/up/down shapes)
  * per-head attention patterns
  * next-token probabilities
* **Neuron Microscope** for any layer, neuron and token:
  * the actual normalized MLP input `x`, plus `g`, `u` and `w = down_proj.weight[:, i]`
  * `gate = g·x`, `up = u·x`, `silu`, signed activation `a` and contribution `a·w`, all recomputed
    in float64 and checked against the hooked values
  * gate-open vs. closed-gate leakage
  * an MLP reconstruction check `MLP_out = Σ a_i w_i (+ b)` with its error
  * a virtualized all-neurons × tokens grid
  * interventions: zero, reverse sign, scale, override slider, restore, on one token or all positions
  * baseline vs. modified forward passes kept separate, with full-vocabulary largest changes,
    tracked target tokens, KL, per-layer residual change, and integrity checks
* **Reproducibility:** every response includes the model ID and commit hash, dtype, device,
  library versions, code git SHA, input text and token IDs, parameters and a timestamp.

## Setup

### Backend (Python ≥ 3.10)

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r backend/requirements.txt
```

```bash
cd backend && NEURONLAB_PRELOAD=Qwen/Qwen2.5-0.5B uvicorn main:app --host 0.0.0.0 --port 8000
```

Environment variables:

| Variable | Purpose |
|---|---|
| `NEURONLAB_PRELOAD` | Model to load at startup (optional; you can also load from the UI) |
| `NEURONLAB_CORS_ORIGINS` | Comma-separated frontend origins (default `http://localhost:3000`) |
| `HF_TOKEN` | Needed for gated models (Llama-3.1-8B-Instruct: accept the license on Hugging Face first) |
| `HF_HUB_DISABLE_XET` | Set to `1` by default; uses plain HTTPS downloads |

### Frontend (Node ≥ 20)

```bash
cd frontend && npm install
```

```bash
npm run dev
```

Open http://localhost:3000. To use a remote GPU backend, click the backend URL in the header
bar and enter, for example, `http://<gpu-host>:8000`. Add the frontend origin to `NEURONLAB_CORS_ORIGINS`
on the backend.

### Tests

```bash
cd backend && pytest -v
```

The tests use tiny random Llama/Qwen2/Qwen3 models, plus the real Qwen2.5-0.5B when it is cached.

### Docker

```bash
docker compose up --build
```

For a CPU-only host, remove the `deploy:` block in `docker-compose.yml`. On a remote GPU, run only the
backend image:

```bash
docker run --gpus all -p 8000:8000 -e HF_TOKEN -v hf-cache:/root/.cache/huggingface neuronlab-backend
```

The Dockerfiles have not yet been built in CI. The development container had no Docker daemon.

## Models and compute

| Mode | Model | Memory (weights) | Hardware |
|---|---|---|---|
| Development | Qwen2.5-0.5B (default), Qwen3-0.6B | ~2 GB fp32 | CPU fine (~0.3 s per forward pass on 4 cores) |
| Paper reproduction | meta-llama/Llama-3.1-8B-Instruct | ~16 GB bf16 | GPU ≥ 24 GB (L4, A10, 4090); 40–80 GB for gradient-heavy phases |
| Later | Qwen3-8B | ~16 GB bf16 | same |

* **J-lens:** pre-fitted lenses exist for Llama-3.1-8B-Instruct and Qwen3-8B
  (`neuronpedia/jacobian-lens`, ~1 GB each), so a fresh calibration is optional.
* **Own calibration:** costs one forward pass plus `d_model / 128` backward passes per prompt.
  For Llama-8B with 1,000 × 128-token prompts, that's several GPU-hours on an A100/H100.

Supported architectures: Llama, Qwen2/2.5, Qwen3 (dense), Mistral, meaning any SwiGLU decoder whose
MLP is `down(silu(gate(x)) * up(x))`. The activation function is verified numerically at load time.

## How to reproduce the paper

This becomes runnable in Phases 2–6; see `docs/PLAN.md` §2. Paper neuron IDs (e.g. Llama L28 n2760,
L20 n2287) are stored in `backend/models/registry.py`, and the backend refuses to use them with any other
checkpoint. A result is labelled a reproduction only when the checkpoint, dataset and sample sizes match
the paper. Everything else is labelled `exploratory`.
