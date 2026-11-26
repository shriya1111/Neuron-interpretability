# NeuronLab — Technical Implementation Plan

This plan is based on a full reading of *Returning to the Neuron Basis for
Interpretability, Part 1: Answering Three Objections* (Shone & Dunlap, Anthropic,
Nov 2026), including Appendices A–D, and on the J-lens paper it builds on
(Gurnee et al. 2026, *Verbalizable Representations Form a Global Workspace in
Language Models*, arXiv:2607.15495) and its reference code
(`github.com/anthropics/jacobian-lens`).

Status legend: **[P1]** = built in Phase 1. Everything else is planned, not built.

---

## 0. What the paper actually specifies (implementation-relevant facts)

### Neuron model (main text)
* SwiGLU neuron: `a = silu(g·x) (u·x)`, adds `a·w` where `w = down_proj.weight[:, i]`.
* `x` is the **normalized** MLP input (post-attention RMSNorm output), not the raw residual.
* The two signs (+w, −w) are treated separately. A sign counts as "used" only when
  activations of that sign occur **with the gate open (g·x > 0)**. SiLU min ≈ −0.28,
  so closed-gate activations "leak", often with the opposite sign.

### A.1 J-lens / J-share
* J-lens direction of token t at layer L = row t of `W_U J_L`, normalized to unit length.
  `W_U` = unembedding **with the final-norm gain folded in**; `J_L` = Jacobian of the final
  residual stream w.r.t. the residual stream at layer L, **averaged over inputs**.
* The last layer uses the unembedding directly.
* Paper used "the J-lens files released with Gurnee et al. (2026)".
* J-lens estimator (Gurnee et al. §2.1 + `jlens/fitting.py`): for each output dim inject a
  one-hot cotangent at every valid target position simultaneously, backprop; the gradient at
  source position p is `Σ_{p'≥p} ∂h_final[p']/∂h_L[p]`; average over source positions p and
  prompts. Skip the first 16 positions and the last position. Reference fits: 1,000 prompts ×
  128 tokens (quality saturates; ~100 usable). Cost: `ceil(d_model/dim_batch)` backward
  passes per prompt.
* J-share of unit output vector w with verbalizable part v: `1 − ‖w − v‖²`, v = nonnegative
  combination of ≤ 25 J-lens token directions found by **nonnegative (gradient) pursuit**.
  Compare against **2,048 random unit directions at the same layer**; high > 99th pct,
  low < 1st pct.
* Published artifacts found: `huggingface.co/neuronpedia/jacobian-lens` contains
  `llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt`
  (fit by Neuronpedia with the official library, WikiText-103, 461 prompts × 128 tokens,
  bf16, early stopping), plus Qwen3-8B, Qwen3-14B, Qwen3-1.7B, Qwen2.5-1.5B-Instruct, etc.
  **It is not confirmed that these are byte-identical to the files the paper used**; they will
  be labelled "published third-party J-lens (Neuronpedia fit, official jlens code)".

### Appendix B (Type 3 audit) — exact thresholds
* Corpus: Pile-10k (first 10k Pile docs) minus Books3, BookCorpus2, StackExchange,
  OpenWebText2 → 7,070 docs.
* Topic docs: Pile-10k docs matching a topic term list; ≤150 docs; 1,500-char window around
  first match; first 256 tokens.
* "Activates strongly": activation on the tested sign > 99.9th percentile over all positions
  of 200 random Pile-10k docs.
* Check 1: strong activation in ≥5 docs of each topic, and share of docs ≥ 0.07 above random docs.
* Check 2: zero the neuron at every position of ≤40 such docs; drop in log-prob of first tokens
  of the *other* topic's words, minus drop for 200 random words, at strongly-active positions;
  must be ≥ 0.03 nats and ≥ 3σ above drops for 20 random directions with the output vector's norm.
* Check 3: related-category control words must not drop by ≥ ¼ as much with z ≥ 3.
* Check 4: of the 15 tokens with largest drop, at most half of the positive change may be
  word continuations (no leading space, lowercase first letter).
* Recall calibration with synthetic two-concept neurons (sum of two output vectors).
* Census: sample middle-layer neurons uniformly until ≥1,000 signed directions have a
  measurable effect.

### Appendix C
* C.1: average linear map of MLP `M = W_down(diag(E[s SiLU'(g) u])W_gate + diag(E[s SiLU(g)])W_up)diag(γ)`
  on 120 docs; singular values of `I + M` < 0.1 = strongly shrunk.
* C.2: single-token lowercase WordNet concept nouns (4–12 letters, leading-space single token);
  layers at 10–82% depth; top-50 neurons by |cos(w, J-lens dir)|; sign/gate filters; direct
  neuron: word ≤6th in readout and ≤6th in unembedding; indirect: ≤10th in readout and outside
  top-500 in unembedding. Inject `f · REF · unit(w)` at the output of one layer at **every
  position**, `REF` = mean norm of final-layer residual on the prompt excluding position 0,
  `f ∈ {0.1, 0.25, 1}`; prompts "I looked up and saw a" and "Let me tell you about"; read the
  word's logit at last position; control = mean of 3 random unit directions; keep concepts where
  both neurons beat control by >1 logit at their own layer for all strengths/prompts.
* C.3: 10 concepts, 4 same-category controls, 16 sentences ending at the word; per layer the
  neuron with largest mean activation difference (gate pre-activation positive on target);
  facets by hand; pairwise |cos| vs. random-neuron baseline (20 draws); steering at norm ½ mean
  residual norm on 8 neutral prompts vs. 6 random directions; causal arrow if z > 8.

### Appendix D (fact traces)
* Athletes: rank layer 0–13 neurons at the name by Δactivation(made-up names → athlete) ×
  alignment of output vector with sport direction at layer 13; top 300 grouped by hand.
* Books/packages: rank answer-position neurons in layers 10–31 by Δactivation(made-up subject)
  × gradient of (answer logit − fixed wrong-answer logit).
* Removal = **resample ablation** with values from made-up subjects; fact removed only when the
  answer's **first token is no longer top-1**. 12 made-up names per athlete (same token count,
  sport prob < 0.05). Random same-layer neuron sets as control.
* MLP-range patching and residual-copy experiments for where information arrives.

### Things the paper does *not* fully specify (will be disclosed as our choices)
* Exact nonnegative pursuit variant/step sizes, Claude-judge prompts and term lists, the hand
  groupings of neurons, the random-word lists, the exact made-up-name pools. NeuronLab will make
  these explicit, editable inputs and record them with every run. LLM judges are never presented
  as experimental evidence; any human/LLM labels are stored as *annotations*.

---

## 1. Architecture

```
frontend/  (Next.js, TS, Tailwind)  ──HTTP/WS──►  backend/ (FastAPI, PyTorch, HF Transformers)
                                                      │
                                                      ├── models/          load, adapters, architecture, registry
                                                      ├── instrumentation/ hooks, cache, inspector, interventions
                                                      ├── evaluation/      metrics (top-k, diffs, KL)
                                                      ├── interpretability/ (P2+) logit lens, J-lens, J-share, attribution
                                                      ├── experiments/     (P4+) polysemanticity, steering, facets, recall
                                                      ├── datasets/        (P4+) calibration, topics, factual prompts
                                                      ├── storage/         (P7)  SQLite + safetensors caches
                                                      └── api/             routes, schemas, (P7) websocket jobs
```

* The backend is a standalone service (Dockerfile, CORS configurable) so the frontend can
  run locally against a remote GPU host.
* A single global model lock serializes requests that touch the model: hooks are module-level
  state and must never leak between concurrent requests.
* All hooks are registered through one context manager that guarantees removal (also on
  exceptions); tests assert zero hooks remain after each operation.
* Model access is through an **adapter** that resolves embeddings, blocks, norms, attention
  and MLP projections by architecture (Llama, Qwen2, Qwen3, Mistral). Dimensions are read from
  the loaded config; nothing is hard-coded.
* Paper neuron IDs live in a registry keyed by exact checkpoint; using them with another
  model raises an error.

## 2. Phases

| Phase | Scope | Compute |
|---|---|---|
| **1** | Neuron microscope: loader/adapters, hooks, activation capture, neuron decomposition, reconstruction check, interventions (zero/scale/negate/set/restore), baseline vs. modified predictions, architecture explorer, correctness tests, frontend | CPU, Qwen2.5-0.5B / Qwen3-0.6B |
| 2 | Llama-3.1-8B-Instruct, logit lens (direction projection with folded final-norm gain vs. full-state lens), John neuron (L28 n2760, −) reproduction | GPU ≥24 GB, HF token |
| 3 | J-lens: load published `.pt` (Neuronpedia/official jlens format), own calibration pipeline (batched VJPs, chunking, checkpointing, merge), J-share via nonnegative pursuit with refitting, random-direction baselines (quick + 2,048), JVP/finite-difference validation plots | load: CPU/GPU; fit: GPU |
| 4 | Polysemanticity Types 1–3, Appendix B four checks with exact thresholds, Pile-10k filtering, random-direction controls, recall calibration with synthetic neurons, randomized census, exploratory scanner kept separate from confirmation | GPU |
| 5 | Cross-layer injection (C.2 selection, `f·REF·unit(w)` at every position, 3 random controls), C.1 shrink analysis, concept facets (C.3) with intervention-backed causal edges only | GPU |
| 6 | Factual recall lab: baseline/first-token vs full-answer, attribution (Appendix D), resample ablation default, progressive removal with random-neuron controls, group restore, MLP-range patching | GPU |
| 7 | Custom experiments, SQLite persistence, background jobs + WebSocket progress + cancel, exports (JSON/CSV/script/notebook), comparison | any |

Each phase ends with its automated tests run and the results recorded in `docs/TEST_RESULTS.md`.

## 3. Reproducibility record (stored with every result; Phase 1 returns it per response)

model id, resolved commit hash, tokenizer class/vocab size, dtype, device, quantization (none
unless stated), torch / transformers versions, code git SHA, input text, token IDs, seed,
intervention parameters, timings. Each experiment config carries a `mode` of
`exploratory` or `paper_reproduction`; a run is only labelled as a reproduction when the
checkpoint, dataset and sample sizes match the paper's specification, and deviations are
listed explicitly.

## 4. Phase 1 correctness tests

1. Hooked baseline logits == uninstrumented logits (bitwise).
2. Σ_i a_i w_i (+ bias) reconstructs the MLP output (tight tolerance in fp32).
3. Recomputed gate/up/SiLU/activation from captured x and weights match hooked tensors.
4. Zero ablation changes that layer's MLP output by exactly −a·w at the chosen positions only.
5. Scaling by s changes it by (s−1)·a·w; negation by −2·a·w; override by (v−a)·w.
6. Activation zero-ablation via hook == forward pass with the down-proj column and activation
   removed by a weight edit on a deep copy (independent implementation); original weights unchanged.
7. No hooks remain after runs, including runs that raise.
8. Repeated baseline runs give identical logits.
9. Token IDs ↔ decoded strings, including leading whitespace.
10. Neuron/layer/position validation and paper-registry checkpoint checks.
11. MLP-bias handling (tiny Llama with `mlp_bias=True`).

Tests run on tiny randomly-initialized Llama/Qwen2/Qwen3 models (fast, no download) and on the
real Qwen2.5-0.5B checkpoint when it is cached.
