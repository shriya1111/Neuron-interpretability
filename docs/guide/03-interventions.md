# 3. Interventions and controls

## 3.1 Principle

An intervention changes something *during* a forward pass and lets every later layer react through the model's real
computation. NeuronLab never edits model weights. Every comparison uses **two separate forward passes**:

```
prompt → unmodified model → baseline logits
prompt → model + intervention hooks → modified logits
```

The hooks are removed after every pass, including when it fails. The API refuses to return a result if any
NeuronLab hook is still attached.

## 3.2 Activation edits (one neuron or many)

The edit happens on the activation vector just before `down_proj`, so it changes only the chosen neurons'
contributions at that layer. If the activation changes from $a$ to $a'$:

$$\Delta\,\text{mlp\_out}_t = (a'_t - a_t)\, w \quad\text{at edited positions } t, \qquad 0 \text{ elsewhere}$$

| Operation | $a'$ | Change written to the residual stream |
|---|---|---|
| zero (ablation) | $0$ | $-a\,w$ |
| scale by $s$ | $s\,a$ | $(s-1)\,a\,w$ |
| reverse sign | $-a$ | $-2a\,w$ |
| override to $v$ | $v$ | $(v-a)\,w$ |
| add $\delta$ | $a+\delta$ | $\delta\, w$ |

Positions: one token, a list, or all tokens. Several edits at different layers can run together.

**Integrity checks returned with every run:**
* how many activations changed, and at which positions;
* whether the pre-edit activation equals the baseline (it won't if an earlier layer was also edited);
* $\max\lvert\Delta\text{mlp\_out}_{\text{observed}} - \Delta_{\text{expected}}\rvert$;
* the number of hooks remaining.

## 3.3 Vector additions

`VectorAdd` adds $c_t\,v$ to a layer's MLP output (`site="mlp_out"`) or to the residual stream after the block
(`site="resid_post"`), at chosen positions with per-position coefficients $c_t$. Uses:
* **Random-direction controls** (§3.5).
* **Steering** (Phase 5): $h \leftarrow h + \text{strength}\cdot \hat v$.

A useful identity, tested in `test_zero_ablation_equals_subtracting_aw`:

$$\text{zero-ablating neuron } i \;\equiv\; \text{adding } {-a_t\, w_i} \text{ to that layer's MLP output at every } t$$

These are two independent code paths, and they agree to $10^{-5}$.

## 3.4 What we measure after an intervention

At the prediction position, for a target token $y$:

| Quantity | Formula |
|---|---|
| logit change | $\Delta z_y = z'_y - z_y$ |
| log-prob change (nats) | $\Delta \log p_y = \log p'_y - \log p_y$ |
| probability change | $p'_y - p_y$ |
| rank change | $\mathrm{rank}'_y - \mathrm{rank}_y$ (rank 1 = top prediction) |
| distribution shift | $\mathrm{KL}(p \,\Vert\, p') = \sum_v p_v (\log p_v - \log p'_v)$ |
| largest changes | top-$k$ tokens by $\Delta\log p$ and by $\Delta p$ over **all** 151,936 tokens |
| propagation | $\lVert h'_L - h_L\rVert / \lVert h_L\rVert$ at every layer (zero before the edited layer) |

## 3.5 Matched random controls

A drop in the target's probability after ablation proves little on its own. *Any* perturbation of that size might
cause it. The control used by the paper (Appendix B, check 2), and by NeuronLab:

1. Draw $N$ random directions $r$, uniformly on the sphere, rescaled so that $\lVert r\rVert = \lVert w\rVert$.
2. For each $r$, add $-a_t\, r$ at **the same layer, the same positions, with the same coefficients** as the ablation.
3. Record $\Delta\log p_y$ for each control.

The neuron's own effect is then compared with that distribution:

$$z = \frac{\Delta_{\text{neuron}} - \mathrm{mean}(\Delta_{\text{controls}})}{\mathrm{sd}(\Delta_{\text{controls}})}$$

We also report the empirical percentile and a 95% *t*-interval for the control mean. Controls are seeded
(`seed` parameter), so they are reproducible.

## 3.6 How to check it

* `test_intervention_delta`: zero, scale (including ×0), negate → exact $(a'-a)w$ at chosen positions, 0 elsewhere.
* `test_set_override`, `test_scaling_is_proportional`.
* `test_ablation_matches_weight_edit`: the hook ablation equals zeroing that `down_proj` column in a **deep copy**
  of the model; the original weights are verified unchanged.
* `test_zero_ablation_equals_subtracting_aw`, `test_vector_add_resid_post_and_positions`.
* `test_random_directions_matched_norm`: exact norm, and the same seed gives the same directions.
* `test_hooks_removed_on_error`, `test_api_concurrent_requests`.
