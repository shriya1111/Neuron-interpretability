# 9. Weight editing (pruning and rewriting neurons)

## 9.1 Interventions vs. weight edits

| | Activation intervention (chapter 3) | Weight edit (this chapter) |
|---|---|---|
| What changes | one forward pass | the model's parameters in memory |
| Lifetime | that request only | every request, every page, until reverted |
| Where | Microscope, experiments | **Weight Editor** page, `/weights/*` API |
| Disk | never | never (export writes a *new* copy only on request) |
| Visibility | listed in that result | red **WEIGHTS MODIFIED** banner + `weight_edits` in every result's provenance |

Both are real changes to the computation. The difference is scope and permanence. The spec forbids silently
altering weights, so edits are always visible, logged and reversible.

## 9.2 The five edits

For neuron $i$ in layer $L$, with $g_i$ (gate row), $u_i$ (up row) and $w_i$ (output vector):

| Edit | Change | Effect |
|---|---|---|
| **prune** | $g_i, u_i, w_i \leftarrow 0$ (and biases) | $a_i \equiv 0$ and contribution $\equiv 0$: the permanent version of zero-ablation at every position |
| **scale_output** | $w_i \leftarrow s\,w_i$ | contribution $\times s$ wherever it fires; firing pattern unchanged |
| **scale_input** | $u_i \leftarrow s\,u_i$ and/or $g_i \leftarrow s\,g_i$ | scaling $u$ scales $a$ exactly by $s$ ($a$ is linear in $u$). Scaling $g$ changes *when* it fires (SiLU is nonlinear); $s<0$ on $g$ flips which inputs open the gate |
| **add_direction** | $w_i \leftarrow w_i + \alpha\,\hat d$ | makes the neuron also write $d$: a token direction $u_t = W_U[t]\odot\gamma_{\text{final}}$, another neuron's $w$, or any vector. With "× ‖w‖", $\alpha$ is relative to the current norm |
| **set_output** | $w_i \leftarrow w_j$ | the neuron keeps its firing pattern but writes neuron $j$'s concept |

Every edit accepts a list of neurons (`5, 81, 100-120`), so you can prune whole groups (Appendix D style).

## 9.3 Exactness and reversibility

Before modifying anything, the editor saves an exact copy of every slice it will touch: rows of `gate_proj` and
`up_proj`, columns of `down_proj`, and biases.
* **revert(id)** writes those saved values back, so the weights are bit-for-bit identical to before. An edit that
  overlaps a later edit must be reverted after it, and the API refuses otherwise.
* **Atomicity.** If an edit fails halfway (for example, a vector of the wrong length), everything it touched is
  restored and no edit is recorded.
* **Checksums** (SHA-256 of the touched slices) are recorded before and after each edit.

## 9.4 Comparing edited vs. original

`/weights/compare` runs every prompt twice:
* once with the edited weights;
* once inside `original_weights()`, which swaps the saved originals back in under the model lock, then restores the edits.

It reports:
* top-k for both models;
* KL divergence;
* target-token probability and rank;
* **held-out perplexity** on WikiText-103 *validation* texts (first $n$ with ≥ 500 characters, 128 tokens each).
  The J-lens is fit on the *train* split, so these don't overlap. This shows whether an edit damaged the model
  generally or only the targeted behaviour.

$$\Delta\text{NLL} = \overline{\text{NLL}}_{\text{edited}} - \overline{\text{NLL}}_{\text{original}} \quad(\text{nats/token}),\qquad
\text{perplexity ratio} = e^{\Delta\text{NLL}}$$

## 9.5 Export

`/weights/export` saves a new folder `data/edited_models/<name>/` with safetensors, the tokenizer and
`neuronlab_edits.json` (base model, base revision, full edit log). The original checkpoint is not changed.

## 9.6 How to check it

* `test_prune_equals_activation_zero_and_reverts_bitwise`
* `test_scale_output_equals_activation_scale`
* `test_scale_up_row_equals_activation_scale`
* `test_set_and_add_direction`
* `test_original_weights_context`
* `test_overlapping_revert_order_and_atomicity`
* `test_multiple_overlapping_edits_original_weights`
* `test_weight_edit_api`: on the real model, prune → provenance shows the edit → compare (incl. held-out) → revert
  all → logits bit-identical to before.

**Manual check:** apply any edit, then revert all, then run the Explorer. The prediction must be identical to before the edit.
