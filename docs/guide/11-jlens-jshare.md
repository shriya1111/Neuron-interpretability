# 11. J-lens, J-share and validating linear predictions

## 11.1 The lens we use

* **Fit:** Qwen3-0.6B, Anthropic's reference `jlens` code (commit 581d398), 1,000 WikiText-103 train prompts × 128
  tokens, fp32, Modal A10G, 3.3 h. It is stored in `data/jlens/qwen3-0.6b-wikitext-n1000/` with a `config.json`
  recording everything.
* **Provenance labels** (never mixed up in the app):
  * **fitted corpus-mean J-lens:** this file;
  * **context Jacobian:** the exact derivative for one prompt (§11.5);
  * **finite-difference intervention:** a real forward pass with a small perturbation.

  Neither of the last two is called "the J-lens".
* **Not the paper's file.** The paper used the files released with Gurnee et al. for its own, larger models.

## 11.2 Definitions

Layer $l$ means the residual stream at the **output of block $l$** (as in the reference code); the target is the output of the last block.

$$J_l = \mathbb{E}_{\text{prompts},\,t,\,t'\ge t}\!\left[\frac{\partial h_{\text{final},t'}}{\partial h_{l,t}}\right]\in\mathbb{R}^{d\times d}$$

The reference estimator puts a one-hot cotangent on every valid target position at once, so it **sums** over
$t'\ge t$, then averages over source positions $t$ (skipping the first 16 and the last) and over prompts.

$$D_l = (W_U\,\mathrm{diag}(\gamma_{\text{final}}))\,J_l,\qquad d_{l,y} = D_l[y]/\lVert D_l[y]\rVert$$

* **Readout** of a vector $w$ at layer $l$: tokens ranked by $\cos(d_{l,y}, w)$.
* **Which layer:** a neuron in block $l$ writes $a\,w$ into the output of block $l$, so it is read with $J_l$. The last
  layer (27) *is* the final residual, so it uses the unembedding ($J=I$), as the paper says.
* **Residual readout across depth:** $\mathrm{softmax}(W_U\,\mathrm{RMSNorm}_{\text{final}}(J_l h_l))$, the reference
  `apply`. Compare it with the logit lens, which uses $J=I$ at every layer.

## 11.3 J-share

For unit $\hat w$:

$$v=\sum_{j\le 25} c_j\, d_{l,y_j},\quad c_j\ge 0,\qquad \text{J-share}(\hat w)=1-\lVert \hat w - v\rVert^2$$

**Algorithm (our explicit choice; the paper says "nonnegative pursuit" without parameters):** nonnegative orthogonal matching pursuit.
1. Add the token whose direction has the largest positive correlation with the current residual.
2. Refit **all** coefficients on the support with exact nonnegative least squares (`scipy.optimize.nnls`); drop tokens
   whose coefficient becomes 0.
3. Stop at 25 tokens or when nothing correlates positively.

Because of the refit, the result is the exact NNLS optimum on the chosen support. Tested:
* an independent NNLS solve reproduces it to $10^{-8}$;
* a vector equal to a token direction gets J-share 1 with one direction;
* more directions never lower it;
* the batched and single implementations agree.

**Reading the coefficients.** Directions picked later can get larger coefficients than the first pick, because they
explain what the earlier, correlated directions could not. The table therefore shows each direction's coefficient
**and** its cosine with $w$.

**Random baseline.** The J-shares of $N$ random unit directions at the same layer: 256 for a quick check, **2,048 as in
the paper**. Results are cached per layer in the lens folder.
* **high:** above the 99th percentile of the random J-shares;
* **low:** below the 1st;
* **middle:** otherwise.

The UI labels which baseline size was used.

## 11.4 Atlas integration

`backend/scripts/precompute_jlens_readouts.py` computes the top 12 J-lens tokens of $+w$ and $-w$ for all 86,016
neurons. That's one $[V\times d]\cdot[d\times 3072]$ product per layer, about 4 minutes in total on 4 CPU cores. Every
Atlas neuron page now shows **logit-lens** and **J-lens** "writes" side by side, plus on-demand J-share with the
histogram against random directions.

## 11.5 Validation: does the linear picture predict real interventions?

Inject $\varepsilon\,\hat v$ at the output of block $l$, at one position $t$, and read token $y$'s logit at $t$:

| Curve | Formula | Meaning |
|---|---|---|
| observed | $z_y(\varepsilon)-z_y(0)$ | real forward passes (`VectorAdd` at `resid_post`) |
| context Jacobian | $\varepsilon\cdot \partial z_y/\partial\varepsilon\,|_0$ | exact first order for **this prompt** (autograd) |
| corpus J-lens | $\varepsilon\cdot u_y^\top P\,J_l\hat v/\mathrm{rms}(h_f)$ | the averaged lens, with the final RMSNorm linearized at the real final state: $P=I-h_fh_f^\top/(d\,\mathrm{rms}^2)$, $u_y=W_U[y]\odot\gamma$ |

$\varepsilon$ is given as a multiple of $\lVert h_l[t]\rVert$.

**Correctness checks:**
* the autograd derivative matches a symmetric finite difference;
* at the last layer, where $J=I$ and the logit is read at the injection position, the corpus prediction equals the
  exact derivative.

**What to expect, and what we see:**
* The context Jacobian tracks the observed curve for small $\varepsilon$ and diverges for large $\varepsilon$ (the
  model is nonlinear).
* The corpus J-lens is **not** a same-position magnitude predictor, because the estimator sums over all later target
  positions. On the Gospel prompt at layer 8 its slope had the right sign but was about 9× too large. At layer 10 it
  had the **opposite sign** to the true local derivative. Its job is ranking directions (cosines), which the paper uses it for.

## 11.6 API

| Endpoint | Purpose |
|---|---|
| `GET /lens/jacobian/status` | whether a lens is loaded, and its provenance |
| `POST /lens/jacobian` | `kind`: `neuron`, `vector` or `residual` readouts |
| `POST /lens/j-share` | J-share of a signed neuron or a vector, plus a random baseline (`n_random` 256 or 2048) |
| `POST /lens/validate` | observed vs. context-Jacobian vs. corpus-lens curves |
