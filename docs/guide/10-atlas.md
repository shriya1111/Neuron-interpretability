# 10. Neuron Atlas

The Atlas is a browsable catalog of every MLP neuron (86,016 in Qwen3-0.6B). Each neuron has its own page and URL
(`/atlas/qwen3-0.6b/L6/n100`), with evidence from **firings** (what it does on real text) and **weights** (what its output
vector writes), plus one-click causal tests. Its layout was inspired by neuron-lens.com. It adds the paper's corpus and
thresholds, causal testing, and weight editing.

## 10.1 How the data is made (`cloud/modal_atlas.py`)

**Corpus.** `NeelNanda/pile-10k` minus Books3, BookCorpus2, StackExchange and OpenWebText2 (paper, Appendix B.1).
**7,070 documents remain, exactly the paper's count.** The first 256 tokens of each are scanned, about 1.8M tokens.

**Pass 1: thresholds.** From 200 random documents (seed 0):
$$\text{thr}_\pm = q_{99.9}(\pm a)\quad\text{over all positions (the paper's definition)}$$
The scan also stores the same threshold excluding position 0. Qwen has no BOS token, and the first token often has outsized activations.

**Pass 2: every document.** For each neuron and each sign it records:
* the **top 24 activations with an open gate** ($g\cdot x>0$), as (document, position, activation), position 0 excluded;
* the **open-gate firing count**;
* per Pile domain, how many positions exceed $\text{thr}_\pm$;
* the maximum $|a|$.

**Readouts.** Logit lens of $\pm w$: the top 12 tokens of $W_U(\gamma_{\text{final}}\odot \pm w)$.
J-lens readouts are added once the fitted lens is downloaded (Phase 3).

**Verification.** The recorded top activations were recomputed locally, one document at a time on CPU, and matched the
batched GPU scan to a relative error of $6\times10^{-6}$ across early, middle and late layers and both signs.

## 10.2 The pages

* **Overview:** one column per layer, one pixel per neuron (32 × 96 grid), colored by firing rate on a log scale.
  Click a pixel to open that neuron, or a layer label to open the layer table.
* **Search:** a word (finds neurons whose ±w readout contains a token that decodes to it), or an ID like `L6 n100`.
  Keys: `/` search, `r` random neuron, `j`/`k` next/previous neuron.
* **Layer table:** sortable by max |a|, firing rate, strongest +, strongest −, or index, showing each sign's readout.
* **Neuron page:** a **+ side** card and a **− side** card. Each has:
  * **Fires on:** the strongest firing contexts, with the firing token highlighted and its activation and domain;
  * **Writes:** logit-lens tokens, with J-lens to come;
  * firing rate, q99.9 threshold, and strong-firing rate by domain.
* **Test it:** re-runs the model on a firing context (the document up to the firing token, as exact token ids), zeroes the
  neuron at every position, and shows the largest drops and gains in next-token log-probability. Also links to the
  Microscope, Vocabulary Lenses and **prune** (a weight edit).

## 10.3 Reading it carefully

* Firings and readouts are **hypotheses**. Agreement between "fires on" and "writes" is suggestive, not proof.
  Causal claims need the controlled experiments (Polysemanticity Lab).
* "Fires on" lists only firings **with the gate open**, following the paper's rule for which sign a neuron uses.
* The logit-lens "writes" only sees the direct path, so for early and middle layers the J-lens readout will be more informative.
* NeuronLab doesn't generate natural-language labels. If added later, they will be marked as annotations, not evidence.

## 10.4 API

| Endpoint | Returns |
|---|---|
| `GET /atlas/list` | available atlases |
| `GET /atlas/overview` | per-layer arrays for the strip |
| `GET /atlas/layer/{L}?sort=&offset=&limit=` | layer table |
| `GET /atlas/neuron/{L}/{n}` | full neuron page data |
| `GET /atlas/search?q=` | word or ID search |

All Atlas endpoints refuse to serve a catalog computed for a different checkpoint than the loaded model.
