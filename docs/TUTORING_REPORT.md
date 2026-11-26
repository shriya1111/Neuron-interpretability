# Tutoring Report: How NeuronLab Was Built, Step by Step

This is a **how-to-build-it** walkthrough. It records, in the order they actually happened, every step taken to
build NeuronLab: what was done, why, the exact commands, what you should see, and what went wrong and how it was fixed.
Follow it from top to bottom and you can rebuild the project yourself.

> **How this differs from the other docs**
> * `docs/guide/` explains **what** the app computes and the maths behind it (concepts).
> * `docs/TEST_RESULTS.md` and `docs/guide/08-lab-notebook.md` record **results**.
> * **This report** explains **how the project was made**, as a sequence of steps you can repeat. It's a tutorial
>   for the builder, not a reference for the user.

Each step has the same parts:
* **Goal**: what the step achieves.
* **Why**: the reasoning behind it.
* **Do**: commands or code.
* **Check**: how you know it worked.
* **Pitfalls**, where relevant: what went wrong and the fix.

---

## Part A — Research before writing any code

### Step 1. Read the paper, including the appendices

**Goal.** Know exactly what to build, especially the details hidden in Appendices A–D.

**Why.** The appendices contain the exact numbers that make a reproduction faithful: thresholds, sample sizes,
injection scales. Building from the abstract alone would produce something that only *looks* like the paper.

**Do.**
1. Fetch the page and strip the HTML to plain text so it can be searched:
   ```bash
   curl -sSL -o paper.html https://alignment.anthropic.com/2026/neuron-basis-theory/
   ```
   Then remove `<script>`/`<style>`, turn block tags into newlines, strip the remaining tags and unescape the HTML
   entities. A short Python script does this.
2. Read everything. Make a checklist of every implementation-relevant fact. The key ones:
   * Neuron: $a = \mathrm{silu}(g\cdot x)(u\cdot x)$; output vector $w$ = column of `down_proj`; a sign counts only with an open gate.
   * J-lens: rows of $W_U J_L$ with the final-norm gain folded in, $J_L$ averaged over inputs.
   * J-share: $1-\lVert w-v\rVert^2$, $v$ = nonnegative combination of at most 25 J-lens directions; 2,048 random directions as a baseline.
   * Appendix B thresholds (99.9th percentile, ≥5 docs, 0.07, 0.03 nats, 3σ, 20 random directions, …).
   * Appendix C injection: $f\cdot\mathrm{REF}\cdot\hat w$, $f\in\{0.1,0.25,1\}$, two fixed prompts, 3 random controls.
   * Appendix D: resample ablation with made-up subjects; "removed" means first token no longer top-1.
3. Write these into `docs/PLAN.md` §0 *before* coding.

**Check.** You can answer "what threshold does check 2 use?" without reopening the paper.

**Pitfall.** The cloud container's network allowlist blocked `alignment.anthropic.com` at first (`curl: (56) CONNECT
tunnel failed, response 403`). **Fix:** add the host to the environment's allowed hosts, then retry.

### Step 2. Read the referenced method (J-lens) and its reference code

**Goal.** Get the exact estimator for the J-lens, which the paper only summarizes.

**Do.**
1. Fetch the arXiv HTML of Gurnee et al. 2026 (arXiv:2607.15495) and search it for the definition (§2.1) and the ablations (§A.7).
2. Clone the reference implementation linked from the paper:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/anthropics/jacobian-lens
   ```
3. Read `jlens/fitting.py`'s docstring. It states the estimator precisely:
   * one-hot cotangent at every valid target position at once;
   * gradient at source $p$ = $\sum_{p'\ge p}\partial h_{\text{final}}[p']/\partial h_L[p]$;
   * mean over source positions and prompts;
   * skip the first 16 positions and the last one;
   * 1,000 prompts × 128 tokens.

**Check.** You can write the formula for $J_L$ and say how many backward passes one prompt costs ($d_{\text{model}}$/`dim_batch`).

### Step 3. Look for pre-computed artifacts before computing anything

**Goal.** Avoid paying for compute that someone has already done.

**Do.** List the Hugging Face repo mentioned in the J-lens README:
```bash
curl -s "https://huggingface.co/api/models/neuronpedia/jacobian-lens/tree/main"
```
Then read each lens's `config.yaml` to learn its provenance (who fitted it, how many prompts, which dataset).

**Check / result.**
* Lenses exist for Llama-3.1-8B-Instruct, Qwen3-8B, Qwen3-4B, Qwen3-1.7B and others.
* There is **none for Qwen3-0.6B or Qwen2.5-0.5B**.
* The Llama lens was fitted by Neuronpedia with 461 prompts, not 1,000, so it was recorded as "third-party, not
  confirmed identical to the paper's".

**Lesson.** Always record provenance. "A J-lens exists" is not the same as "the paper's J-lens".

### Step 4. Inventory your compute and network

**Goal.** Know what can run where before planning.

**Do.**
```bash
nvidia-smi; nproc; free -g; df -h
```
```bash
curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B
```
**Result.** No GPU, 4 CPU cores, 15 GB RAM, about 61 GB disk; Hugging Face, PyPI and npm reachable; Llama gated (401).

**Decision.** Build and verify everything on small models on CPU. Leave GPU-only work for later.

---

## Part B — Environment

### Step 5. Python environment

```bash
python3 -m venv .venv
```
```bash
. .venv/bin/activate
```
```bash
pip install transformers accelerate safetensors fastapi "uvicorn[standard]" pydantic pytest httpx numpy
```

**Pitfall.** The PyTorch CPU-only index (`download.pytorch.org`) was blocked. **Fix:** the standard PyPI `torch` wheel was
installed as a dependency and runs fine on CPU.

### Step 6. Download the model

```python
transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32)
```

**Pitfall.** The download failed with `CAS Client Error … cas-server.xethub.hf.co 403`. Hugging Face's "Xet" transfer
host was not allowed. **Fix:** `export HF_HUB_DISABLE_XET=1` forces plain HTTPS downloads. NeuronLab now sets this by
default in `models/loader.py`.

### Step 7. Read the model's actual source code

**Goal.** Never assume how the model computes. Read it.

**Do.**
```python
import inspect
from transformers.models.qwen2 import modeling_qwen2 as m
print(inspect.getsource(m.Qwen2MLP))
print(inspect.getsource(m.Qwen2RMSNorm))
print(inspect.getsource(m.Qwen2DecoderLayer.forward))
```

**What you learn.**
* MLP = `down_proj(act_fn(gate_proj(x)) * up_proj(x))`.
* The MLP input is `post_attention_layernorm(resid_mid)`, **not** the raw residual stream.
* Qwen2 has no MLP bias; Llama has an optional `mlp_bias`.
* The decoder layer is called with `hidden_states` as its first positional argument.

All of the hook design follows from these lines.

---

## Part C — Plan and test design

### Step 8. Write the plan and the list of tests first

**Goal.** Decide what "correct" means before writing code.

**Do.** `docs/PLAN.md` contains:
* the facts extracted from the paper;
* an architecture diagram;
* a 7-phase table;
* a list of correctness properties, which became the tests in Step 15:
  * hooked == unhooked;
  * $\sum a_i w_i$ = MLP output;
  * ablation changes exactly $-a\,w$;
  * hooks removed;
  * determinism;
  * tokens round-trip;
  * ID validation;
  * bias handling.

**Why.** Writing tests from the maths, not from the code, catches bugs that tests written after the fact would encode.

---

## Part D — Backend core (Phase 1)

### Step 9. A model adapter (`backend/models/adapters.py`)

**Goal.** One place that knows where each part of each architecture lives.

**Key ideas.**
* Resolve modules by name: `model.model.layers[i].mlp.{gate,up,down}_proj`, `input_layernorm`,
  `post_attention_layernorm`, `model.model.norm`, `lm_head`.
* **Verify SiLU numerically**: compare `act_fn(z)` with `torch.nn.functional.silu(z)` on a grid. Don't trust class names.
* Read all dimensions from weight shapes, never from constants.
* `output_vector(L, i)` returns `down_proj.weight[:, i]`.

### Step 10. Loader with provenance (`backend/models/loader.py`)

* `model.eval()`, `requires_grad_(False)`, `torch.use_deterministic_algorithms(True, warn_only=True)`, seed.
* `attn_implementation="eager"`, so attention weights can be captured and behaviour is exact.
* Default dtype: fp32 on CPU (tight checks), bf16 on GPU.
* Record the commit hash.

**Pitfall.** Transformers 5 no longer sets `config._commit_hash`. **Fix:** read it from the HF cache path
(`…/snapshots/<hash>/config.json`) with `huggingface_hub.try_to_load_from_cache`.

### Step 11. Hooks that always get removed (`backend/instrumentation/hooks.py`)

* A context manager collects every handle and removes all of them in `finally`, so cleanup happens even on errors.
* Every NeuronLab hook is **tagged**, so leak checks count only our own hooks (see Step 20 for why).

### Step 12. The instrumented forward pass (`backend/instrumentation/activation_cache.py`)

**Where to hook, and why:**

| Quantity | Hook |
|---|---|
| `resid_pre` | forward **pre**-hook on the block (with kwargs) |
| `attn_out` | forward hook on `self_attn` |
| `mlp_in` = $x$ | forward **pre**-hook on the MLP module |
| `gate_pre`, `up` | forward hooks on `gate_proj`, `up_proj` |
| activation $a$ | forward **pre**-hook on `down_proj` (its input is exactly $a$) |
| `mlp_out`, `resid_post` | forward hooks on the MLP and the block |

**Rules.**
* Store copies: detach, convert to float32, move to CPU.
* Hold a lock for the whole pass.
* Intervention edits happen in the same `down_proj` pre-hook, which records both the original and the edited activation.

### Step 13. The neuron inspector (`backend/instrumentation/neuron_inspector.py`)

* Recompute gate, up, σ, SiLU, $a$ and $a\,w$ in **float64** from the captured $x$ and the weights. Report the
  difference from the hooked values.
* Recompute $x$ from `resid_mid` via RMSNorm, to show that $x$ is not the raw residual.
* Reconstruction: `act @ W_down.T (+ bias)` vs. the hooked `mlp_out`, giving the max absolute and relative error.

### Step 14. Interventions and comparisons

* `NeuronIntervention(layer, neurons, op, value, positions)` with ops `zero | scale | negate | set | add`.
* `compare_runs` does two separate passes (baseline and modified) and returns:
  * top-k for each pass, and the largest changes over the full vocabulary;
  * KL divergence;
  * per-layer residual change;
  * **integrity checks**: observed $\Delta$`mlp_out` vs. expected $\sum(a'-a)w$.

### Step 15. Test on tiny random models (`backend/tests/`)

**Trick.** Build tiny Llama, Qwen2 and Qwen3 models with `AutoModelForCausalLM.from_config(...)` (3 layers, $d=64$).
They're instant and need no download.

**Pitfall.** With default init, activations are tiny and never reach the interesting regimes. **Fix:** re-initialize
weights with std 0.2, so closed gates and negative SiLU actually occur.

**Strongest test.** Hook ablation == zeroing that `down_proj` column in a `copy.deepcopy` of the model. These are two
independent implementations, and the test also checks that the original weights are untouched.

```bash
cd backend && pytest -q
```

**Pitfall.** One test assumed `" Ripper"` is one token. It's two (`" Rip"`, `"per"`). The API's validation correctly
rejected it, so the test was wrong, not the code. **Lesson:** test with tokens you've checked.

### Step 16. The API (`backend/api/`, `backend/main.py`)

* FastAPI with Pydantic schemas. `ValueError` and `KeyError` become HTTP 422 with the message.
* Every response carries `provenance`: model, revision, dtype, device, versions, git SHA, token ids, params, time.
* After each request, the API **refuses to return results** if any NeuronLab hook remains.

```bash
cd backend && NEURONLAB_PRELOAD=Qwen/Qwen3-0.6B uvicorn main:app --port 8000
```

---

## Part E — Frontend (Phase 1)

### Step 17. Scaffold

```bash
npx create-next-app@15 frontend --ts --tailwind --eslint --app --src-dir --import-alias "@/*" --use-npm --no-turbopack --yes
```
```bash
cd frontend && npm install plotly.js-dist-min @xyflow/react @tanstack/react-virtual
```

**Pitfall.** `react-plotly.js` has peer-dependency problems with React 19. **Fix:** a 40-line wrapper
(`components/Plot.tsx`) that dynamically imports `plotly.js-dist-min`, calls `Plotly.react`, and forwards click events.

### Step 18. Structure

* `lib/api.ts`: typed client. The backend URL can be changed at runtime, so it works with a remote GPU.
* `lib/state.tsx`: shared selection state (prompt, layer, neuron, position, mode, theme), persisted in `localStorage`
  with try/catch.
* `components/Shell.tsx`: header, sidebar, and the **three-panel `Workspace`** (controls | visuals | numbers).
* Pages: `explorer`, `microscope`, and later `lenses` and `polysemanticity`. Unbuilt pages say so and show no data.
* `NeuronGrid.tsx`: a virtualized neurons × tokens grid that loads 256 neurons per request as you scroll.

### Step 19. Build, then test in a real browser

```bash
npx tsc --noEmit
```
```bash
npm run lint
```
```bash
npm run build
```

Then a Playwright script loads each page against the running backend, clicks controls, reads the displayed numbers,
records console errors, and takes screenshots. **Looking at the screenshots** found three problems that tests didn't:
* **Architecture graph unreadable:** 28 blocks in one column. Fixed by splitting into columns of ≤ 12.
* **Heatmap washed out:** the first token's huge norm ("attention sink") dominated the color scale. Fixed with a log scale by default.
* **Comparison tables cramped:** two tables side by side in the narrow right panel. Fixed by stacking them.

**Pitfall.** After a rebuild, an old `next-server` still held port 3000 and served stale chunks. Fix: `pkill -f next-server` before `next start`.

---

## Part F — Bugs found by running the real app

### Step 20. Hooks that weren't ours
After requesting attention weights, the API refused to answer: "48 hooks still registered". Inspecting
`module._forward_hooks` showed `transformers.utils.output_capturing` hooks. Transformers 5 installs these permanently
the first time `output_attentions=True` is used.
* **Fix:** tag our hooks and count only those; report the others under `/health → foreign_hooks`.
* **Test:** those hooks don't change logits.

### Step 21. A race condition
The microscope page fires two requests at once. The leak check counted hooks *without* holding the model lock, so it
sometimes saw the other request's hooks and returned an error.
* **Fix:** count under the lock.
* **Test:** `test_api_concurrent_requests` sends 16 mixed requests across 8 threads.

### Step 22. Commit on a branch and push
```bash
git checkout -b phase-1-neuron-microscope
```
```bash
git add -A && git commit
```
```bash
git push -u origin phase-1-neuron-microscope
```
Check before committing that no `node_modules`, `.next`, `.venv` or model files are staged.

---

## Part G — Phase 2: logit lens and the Type 1 experiment

### Step 23. Choose the model by checking what it knows
**Decision:** Qwen3-0.6B (same family as the paper's Qwen3-8B/14B), with Qwen2.5-0.5B as the cross-check.

Before designing experiments, **probe the model's knowledge** with real forward passes. Qwen3-0.6B knows Gospel → John
but not John the Ripper → John, and not Jágr → hockey. This shaped which contexts are usable. The results are recorded
in the lab notebook.

### Step 24. Logit lens (`backend/interpretability/logit_lens.py`)
* **Direction readout:** `scores = W_U @ (gamma_final * w)`. The readout of $-w$ is `-scores`. Also offer cosine with unit rows of $W_U\,\mathrm{diag}(\gamma)$.
* **Residual lens:** apply the model's own `final_norm` module to each layer's state, then $W_U$.
  **Verify** that the last layer equals the model's logits (measured error 1.9e-6).
* **Cache** the row norms on the adapter object, not in a global dict keyed by `id(model)`. A freed model's id can be
  reused by a new one, which would return a stale cache.

### Step 25. Direct logit attribution and neuron search
* $\mathrm{DLA}=a\,(u_y\cdot w)/\mathrm{rms}(h_{\text{final}})$. Tested against removing $a\,w$ from the final state by hand.
* Neuron search: one matrix-vector product per layer covers all neurons, then exact vocab ranks for the top
  candidates. Tested against a brute-force loop.

### Step 26. Matched random controls
* `VectorAdd` adds $c_t\,v$ to a layer's `mlp_out` or `resid_post`. It's registered *before* the capture hooks, so the
  captures see the change.
* **Key identity, tested:** zero-ablating neuron $i$ ≡ adding $-a_t w_i$. That makes the control "add $-a_t r$ with
  $\lVert r\rVert=\lVert w\rVert$" a perfectly matched comparison.

### Step 27. The Type 1 experiment (`backend/experiments/polysemanticity.py`)
* Per context: a baseline pass, an ablation pass and N control passes, giving $\Delta\log p$, a z-score, a percentile and a status.
* Thresholds borrowed from Appendix B (0.03 nats, 3σ), and stated in every result as our choice.
* The **candidate scan** is kept separate and labelled "exploratory".
* Running it for real on "John" gave **inconclusive**: random directions of the same size hurt as much. Report it as it is.

### Step 28. UI pages and docs
* Vocabulary Lenses and Polysemanticity Lab pages, then the browser check again.
* `docs/guide/` (8 chapters), lab notebook entries, and updated `TEST_RESULTS`, `LIMITATIONS` and `PLAN`.

---

## Part H — Fitting the J-lens

### Step 29. Measure before you spend
Time one forward pass and a few backward passes, then extrapolate. Qwen3-0.6B needs 1,024 backward passes per prompt.
On this CPU that's about 6 minutes per prompt (measured: 360 s and 316 s for the first two prompts).

### Step 30. Running it on Modal (blocked, documented for next time)
1. **Never paste API keys into a chat.** Store them through the environment's secret settings. If a key does get
   pasted, revoke it afterwards.
2. Add `api.modal.com` to the network allowlist.
3. **Pitfall:** "Could not connect to the Modal server" although the host was allowed. Debug logging
   (`MODAL_LOGLEVEL=DEBUG`) showed the real cause: Modal needs an optional package to use an HTTP proxy.
   **Fix:** `pip install "modal[api-proxy-support]"`. An earlier attempt with a hand-made TCP tunnel was unnecessary
   and was removed. **Lesson:** read the debug log before building workarounds.
4. `modal token set …` succeeded.
5. **Pitfall:** every GPU (L40S, A10G, T4) was refused: "Please add a payment method". Also, the GPU type must be set in
   the `@app.function` decorator. `with_options` didn't change the GPU that Modal's billing check sees, so the script
   reads `NEURONLAB_GPU` when it is imported.
6. The script `cloud/modal_jlens.py` is ready. It runs in chunks that save progress, and writes a `config.json` with full provenance.
7. After a card was added, a new error appeared: `403 … no rule allows this host` for **`input-plane.us-east.modal.com`**.
   Modal sends function calls to this second host. `MODAL_LOGLEVEL=DEBUG` named it. **Fix:** allow that exact host.
8. **Measure first:** a 4-prompt run on an A10G took about 12.5 s/prompt. The first prompts' Jacobian norms (8.907, 8.926)
   matched the CPU run exactly, a free cross-check that GPU and CPU compute the same thing.
9. **Launch detached** so the job survives the session pausing:
   ```bash
   NEURONLAB_GPU=A10G modal run --detach cloud/modal_jlens.py --n-prompts 1000 --tag qwen3-0.6b-wikitext-n1000 --chunk 25 --dim-batch 32
   ```
   Estimate: about 1,000 × 15 s ≈ 4 h on an A10G, roughly $4–5. Download when done:
   ```bash
   modal volume get neuronlab-jlens qwen3-0.6b-wikitext-n1000 data/jlens/
   ```

### Step 31. Running it on CPU instead (in progress)
```bash
python cloud/cpu_fit_jlens.py --n-prompts 200 --tag qwen3-0.6b-wikitext-cpu
```
* It runs Anthropic's own `jlens.fit`, with the same recipe as the published lenses.
* It checkpoints every 5 prompts, and rerunning the command resumes.
* **What happened:** the container was paused while the session was idle. The process died after 2 prompts, before
  the first checkpoint at prompt 5. **Lesson:** long jobs don't belong in an interactive cloud container. Use a
  detached cloud job (Step 30.9), or checkpoint after every unit of work.

---

## Part I — Habits that made this work

1. **Read the source, not the docs.** The MLP input and the hook points came from `inspect.getsource`.
2. **Write tests from the maths.** Then test the same thing in two independent ways (hook vs. weight edit, ablation vs. vector add).
3. **Verify lenses against ground truth.** The last-layer lens must equal the model's prediction.
4. **Always keep a control.** A drop in probability means nothing without matched random perturbations.
5. **Record provenance everywhere.** Model revision, versions, token ids, seeds.
6. **Look at the app.** Screenshots found layout and scaling problems that tests can't.
7. **Report what happened.** An "inconclusive" or "the model doesn't know this fact" result is a result.

## Appendix — Command cheat sheet

| Task | Command |
|---|---|
| Run backend | `cd backend && NEURONLAB_PRELOAD=Qwen/Qwen3-0.6B uvicorn main:app --port 8000` |
| Run frontend | `cd frontend && npm run dev` |
| All tests | `cd backend && pytest -v` |
| Cross-check model tests | `NEURONLAB_TEST_MODEL=Qwen/Qwen2.5-0.5B pytest tests/test_real_model.py` |
| Hook status | `curl localhost:8000/health` |
| CPU J-lens fit | `python cloud/cpu_fit_jlens.py --n-prompts 200` |
| Modal J-lens fit | `NEURONLAB_GPU=A10G modal run cloud/modal_jlens.py --n-prompts 4 --tag timing` |
| Fit progress | `grep "prompt " data/jlens/qwen3-0.6b-wikitext-cpu/fit.log \| tail` |

## Changelog
| Date | Added |
|---|---|
| 2026-11-26 | Steps 1–31 (Phases 1–2, J-lens fitting setup) |
