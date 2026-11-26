# 4. Logit lens

The logit lens asks: *if this vector were fed straight to the unembedding, which tokens would it push up?*
NeuronLab implements two different operations and never mixes them up.

## 4.1 Reading a direction (a neuron's output vector)

$$\text{score} = W_U\,(\gamma_{\text{final}} \odot w) \;\in\; \mathbb{R}^{V}$$

* $\gamma_{\text{final}}$ is the final RMSNorm gain. It is **folded into $W_U$**, as the paper specifies (Appendix A.1).
* The final norm also divides by $\mathrm{rms}(h)$. That is a positive number shared by all tokens, so it does not change
  the *ranking*. It does change magnitudes (see §4.3).
* $\text{score}_t$ is the direct logit effect on token $t$ per unit of activation.
* **The readout of $-w$ is exactly $-\text{score}$**, so its top tokens are $+w$'s bottom tokens. Both are shown,
  because a neuron adds $-w$ whenever $a<0$.

**Cosine version.** Let $u_t = W_U[t] \odot \gamma_{\text{final}}$ (the direction whose dot product gives token $t$'s score):

$$\cos_t = \frac{u_t \cdot w}{\lVert u_t\rVert\,\lVert w\rVert}$$

Ranking by score favors tokens with large embedding rows. Ranking by cosine removes that effect. The paper's
J-lens readout uses cosine (chapter 4.6), so both are offered.

**Important:** a direction readout is **not** a next-token prediction. It describes what a vector pushes toward, not
what the model will say.

## 4.2 Reading a residual state at a layer

$$p_L = \mathrm{softmax}\big(W_U\,\mathrm{RMSNorm}_{\text{final}}(h_L)\big)$$

This uses the model's real final norm, including $1/\mathrm{rms}$ and $\varepsilon$, on the *complete* residual stream at
layer $L$. It asks what the model would predict if every remaining layer were skipped. Only the last row
($L=27$) is the model's actual prediction. NeuronLab verifies that row against the real logits: the measured max error is
$1.9\times10^{-6}$.

Example (Qwen3-0.6B, *"The fourth Gospel was written by the apostle"*): `" Jesus"` leads at layers 20–22,
`" John"` takes over from layer 23, and the final prediction is `" John"` with $p=0.42$.

## 4.3 Direct logit attribution (DLA)

How much does one neuron, through the direct path only, add to token $y$'s final logit at token position $t$?

$$\mathrm{DLA}_i(y) = \frac{a_{i,t}\,(u_y \cdot w_i)}{\mathrm{rms}(h_{\text{final},t})}$$

This is exact for the direct path, holding the final $\mathrm{rms}$ fixed. `test_dla_equals_direct_effect` checks it against
removing $a\,w$ from the final state by hand. It ignores everything that happens *through later layers*.

## 4.4 Searching for neurons that promote a token

Exact scan over all $28 \times 3072$ neurons: rank by $\cos(u_y, \pm w_i)$. For each top candidate, compute the token's
exact rank among all 151,936 tokens of that neuron's readout. This is **exploratory**. Alignment says nothing about
when a neuron fires, or whether it matters causally.

## 4.5 Limits

The logit lens only sees the **direct path** to the output. Neurons in early and middle layers mostly act *through* later
layers, so their logit-lens readouts often look like noise. The paper uses the J-lens for this reason.

## 4.6 Preview: the J-lens (Phase 3)

The J-lens replaces "skip the remaining layers" with "the average linear effect of the remaining layers":

$$J_L = \mathbb{E}_{\text{prompts},\,t,\,t'\ge t}\left[\frac{\partial h_{\text{final},t'}}{\partial h_{L,t}}\right] \in \mathbb{R}^{d\times d}, \qquad
\text{J-lens direction of token } y = \frac{(W_U J_L)[y]}{\lVert (W_U J_L)[y]\rVert}$$

The reference estimator (Anthropic's `jlens` code) does the following:
* Backpropagates a one-hot cotangent on each output dimension, at every valid target position at once.
* This gives $\sum_{t'\ge t}\partial h_{\text{final},t'}/\partial h_{L,t}$, which it averages over source positions $t$ and prompts.
* It skips the first 16 positions and the last one.

No published J-lens exists for Qwen3-0.6B or Qwen2.5-0.5B, so NeuronLab will fit its own. That takes $d$ backward passes
per prompt (1,024 for Qwen3-0.6B). The **J-share** of a unit vector $w$ is $1 - \lVert w - v\rVert^2$, where $v$ is the best
nonnegative combination of at most 25 J-lens directions.

## 4.7 How to check it

* `test_folded_scores_match_manual`: score = $W_U(\gamma\odot w)$ in float64.
* `test_minus_readout_is_negation`.
* `test_residual_lens_last_layer_is_model_prediction`.
* `test_dla_equals_direct_effect`.
* `test_neuron_search_matches_bruteforce`: the fast search returns the same best neuron as a brute-force loop.

**Manual check:** in Vocabulary Lenses → residual state, the last row must match the Model Explorer's prediction.
