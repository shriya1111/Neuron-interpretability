# 6. Features and API

## 6.1 Layout and modes

Every page uses three panels:
* **left** for inputs (model, prompt, selectors, interventions);
* **center** for visualizations;
* **right** for numbers, formulas and conclusions.

The header has the backend URL (click to point it at a remote GPU), **learning / research** mode, and **dark / light**.
Learning mode adds explanatory notes (blue left border). Research mode hides them.

**Hover cards.** Hover any neuron to see a summary card next to the cursor, without leaving the page. This works on
Atlas pixels, layer-table rows, search results, scanner and candidate rows, the top-neuron chips in the Explorer and
Microscope, the Microscope's grid rows, and the neuron-search rows. The card shows, for each sign:
* firing rate and census J-share class;
* J-lens and logit-lens readouts;
* the strongest firing snippet, with the firing token highlighted.

Clicking still opens the full page. Cards come from `GET /atlas/summary/{L}/{n}` (under 1 ms) and are cached in the browser.

Sidebar badges: no badge = built. `◐P3` = partly built, completed in Phase 3. `P5` = planned for Phase 5.
Planned pages show no data.

## 6.2 Pages

| Page | Status | What you can do |
|---|---|---|
| Model Explorer | built | load a model, run a prompt, see the architecture with real norms, layer × token heatmap, block internals, per-head attention, next-token probabilities |
| Neuron Microscope | built | inspect any neuron at any token (chapter 2); zero / reverse / scale / override with a slider; baseline vs. modified predictions; integrity checks |
| Vocabulary Lenses | logit lens built; J-lens Phase 3 | readout of $\pm w$ for any neuron; logit lens of the residual stream across depth; find neurons that promote a token |
| Polysemanticity Lab | Type 1 built; Types 2/3 Phase 4 | candidate scan; Type 1 experiment with matched random controls and verdicts |
| Cross-Layer Steering | Phase 5 | — |
| Factual Recall Lab | Phase 6 | — |
| Custom Experiments, History | Phase 7 | — |

### Typical workflow (Type 1)
1. **Model Explorer:** check that the model predicts your target after each prompt.
2. **Polysemanticity Lab:** enter the contexts, then click *scan all neurons*.
3. Click a candidate, then *run experiment*.
4. Open the neuron in the **Microscope** to see its activation per token, and in **Vocabulary Lenses** to see its readout.

## 6.3 Backend API

All endpoints return a `provenance` record:
* model id and commit hash, dtype, device;
* torch/transformers versions and the code git SHA;
* input text and token ids, seed, parameters, timestamp.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, CUDA, attached NeuronLab hooks (must be 0), foreign hooks |
| GET | `/models/presets` | model presets and paper neuron registry |
| POST | `/models/load` | `{model_id, device, dtype, revision}` |
| GET | `/models/config` | architecture description + HF config |
| POST | `/inference` | tokens, top-k, per-layer summary, optional attention pattern of one layer |
| POST | `/activations/capture` | one layer's activation/gate/up matrix for a neuron range, or the top neurons by max \|a\| |
| POST | `/neurons/inspect` | full decomposition of one neuron at one token (chapter 2) |
| POST | `/interventions/run` | baseline vs. modified with any number of neuron edits |
| POST | `/lens/logit` | `kind`: `neuron` (direction readout), `vector` (any $d$-vector), `residual` (lens across depth) |
| POST | `/lens/neuron-search` | neurons whose $\pm w$ best aligns with a token |
| POST | `/experiments/polysemanticity` | Type 1 experiment |
| POST | `/experiments/polysemanticity/candidates` | exploratory Type 1 candidate scan |

Interactive API docs: `http://localhost:8000/docs` (FastAPI).

Example:

```bash
curl -s localhost:8000/interventions/run -H 'content-type: application/json' -d '{"prompt":"The fourth Gospel was written by the apostle","target_tokens":[" John"],"interventions":[{"layer":27,"neurons":[81],"op":"zero","value":0,"positions":null}]}'
```

## 6.4 Running it

```bash
cd backend && NEURONLAB_PRELOAD=Qwen/Qwen3-0.6B uvicorn main:app --port 8000
```

```bash
cd frontend && npm install && npm run dev
```
