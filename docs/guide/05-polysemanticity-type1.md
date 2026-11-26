# 5. Polysemanticity — Type 1

## 5.1 The paper's taxonomy

A neuron is usually called *polysemantic* when it **fires** on unrelated inputs. The paper classifies neurons by what
they **add** as well:

| Type | Fires on | Adds |
|---|---|---|
| **1** | unrelated inputs | **one** concept in all of them (e.g. "John" for the Gospel *and* for John the Ripper) |
| **2** | — | unrelated concepts with its **two signs** ($+w$ = mascara, $-w$ = radio), never both at once |
| **3** | — | **two unrelated concepts with one sign** (e.g. Elon Musk + protein translation); rare, ≈1% or less |

**Definition used:** a neuron *adds* a concept in a context **only if zeroing it lowers the probability of that concept's words.**

## 5.2 NeuronLab's Type 1 protocol

Inputs: a signed neuron (layer, index, sign) and at least two contexts. Each context is a prompt plus its single-token
target, and should be about an unrelated topic.

For each context:
1. **Baseline pass.** Record gate and activation at every token, and whether the tested sign occurs with an open gate.
2. **Direct logit attribution** of the neuron to the target at the prediction position (chapter 4.3).
3. **Ablation pass.** Zero the neuron at every position. Measure Δlog p, Δlogit, Δp and Δrank of the target.
4. **Controls.** $N$ random directions with $\lVert r\rVert=\lVert w\rVert$, each added as $-a_t r$ at every position (chapter 3.5).
5. **Status:**

| Status | Rule |
|---|---|
| supported | $\Delta\log p \le -0.03$ nats **and** $z \le -3$ |
| contradicted | $\Delta\log p \ge 0$ (ablation does not lower the target) |
| inconclusive | anything else |
| insufficient data | the neuron never fires with the tested sign and an open gate in that context |

The thresholds (0.03 nats, 3σ) come from the paper's Appendix B check 2. The paper doesn't give a separate statistical
test for Type 1. **Overall verdict:** *supported* only if every context is supported.

What the protocol **does not** test: whether the contexts are truly unrelated (your judgment), and whether the
neuron's effect goes through the concept the readout names.

## 5.3 Finding candidates (exploratory)

`/experiments/polysemanticity/candidates` ranks every neuron by
$\min_{\text{contexts}} \mathrm{DLA}_i(\text{target})$, i.e. neurons that push the target up directly in *every* context.
This finds candidates; it is not evidence. Last-layer neurons dominate this list, because DLA only sees the direct path.

## 5.4 Reading the results page

Each context gets a panel with:
* the tokens colored by activation (⊘ = closed gate);
* the activation and gate at the prediction position;
* baseline → ablated probability;
* the histogram of control effects, with the neuron's effect as a red line;
* the status and the reason for it.

The right panel shows the overall verdict and the logit-lens readout of the tested sign.

## 5.5 Notes on interpretation

* **Huge activations make strong controls.** Some last-layer neurons have $\lvert a\rvert > 100$. Subtracting $a\,r$ for a
  random $r$ of the same norm is then a large perturbation, and it can lower the target as much as the real ablation does.
  The neuron is then not *specifically* important, and the verdict will be inconclusive. That is the control working.
* **The target must be predicted.** If the model doesn't predict the target to begin with, there is nothing to lower.
  Check the Model Explorer first.

## 5.6 How to check it

* `test_phase2_api` (real model) runs candidates → experiment end to end and checks that:
  * each status is one of the four allowed values;
  * there is one control per requested control;
  * no hooks remain;
  * a single context is rejected.
* Each ingredient has its own test in chapters 3–4.
