# 2. Neurons

## 2.1 What one neuron computes

Neuron $i$ in layer $L$ owns three vectors of length $d$:

| Vector | Where it lives | Role |
|---|---|---|
| $g_i$ | row $i$ of `gate_proj.weight` | decides whether the neuron is "on" |
| $u_i$ | row $i$ of `up_proj.weight` | decides how much and **which sign** |
| $w_i$ | column $i$ of `down_proj.weight` | **what it writes** into the residual stream (its *output vector*) |

At each token, given the normalized MLP input $x$:

$$
\begin{aligned}
\text{gate} &= g_i \cdot x \\
\text{up} &= u_i \cdot x \\
\mathrm{SiLU}(z) &= z\,\sigma(z) = \frac{z}{1+e^{-z}} \\
a_i &= \mathrm{SiLU}(\text{gate}) \times \text{up} \\
\text{contribution}_i &= a_i\, w_i \;\in \mathbb{R}^d
\end{aligned}
$$

So **a neuron is a context-gated steering vector**. The context decides one number, $a_i$, and the neuron adds
$a_i$ times a *fixed* vector $w_i$ to the residual stream. Whatever $w_i$ "means", the neuron adds more or less of it,
with either sign.

Code: `w = model.model.layers[L].mlp.down_proj.weight[:, i]` (`adapter.output_vector(L, i)`).

## 2.2 Two signs

`up` can be positive or negative, so $a_i$ can be too:

* $a_i > 0$: the neuron adds $+w_i$
* $a_i < 0$: the neuron adds $-w_i$

The paper treats $+w$ and $-w$ as two separate "signed directions" that can mean unrelated things. A neuron can't
add both at the same token. Throughout the app, **warm colors mean positive and cool colors mean negative**.

## 2.3 The closed gate and "leakage"

SiLU is not zero for negative inputs. Its minimum is about $-0.278$, at $z \approx -1.278$:

| gate $z$ | $-6$ | $-3$ | $-1.28$ | $0$ | $1$ | $3$ |
|---|---|---|---|---|---|---|
| $\mathrm{SiLU}(z)$ | $-0.015$ | $-0.142$ | $-0.278$ | $0$ | $0.731$ | $2.858$ |

When the gate is closed ($g\cdot x \le 0$), small values still leak through, *often with the opposite sign*, because
the negative SiLU flips the sign of `up`. Following the paper, **a sign only counts as one the neuron uses when it
occurs with the gate open ($g\cdot x > 0$)**. The microscope marks closed-gate tokens with ⊘ and greys their bars,
and the Type 1 experiment reports `sign_used_with_open_gate` for each token.

## 2.4 The MLP output is exactly a sum over neurons

$$\text{mlp\_out} = W_{\text{down}}\, a = \sum_{i=1}^{m} a_i\, w_i \;(+\, b_{\text{down}} \text{ if the model has a bias})$$

This is just matrix multiplication written column by column, but it is the foundation of the neuron basis:
**every MLP output is a sum of neuron contributions**. NeuronLab checks it numerically at every request:

$$\text{relative error} = \frac{\big\lVert \sum_i a_i w_i + b - \text{mlp\_out}_{\text{hooked}} \big\rVert}{\lVert \text{mlp\_out}_{\text{hooked}} \rVert}$$

computed in float64 from the hooked activations. Measured values: $2.9\times10^{-7}$ (Qwen2.5-0.5B) and
$3.3\times10^{-7}$ (Qwen3-0.6B), i.e. float32 rounding. Neither model has an MLP bias; Llama has an optional one, and the
check includes it when present (`test_reconstruction_with_bias`).

## 2.5 A real worked example

Qwen3-0.6B, prompt *"The fourth Gospel was written by the apostle"*, layer 27, neuron 81, last token (`le`):

| Step | Value |
|---|---|
| gate $=g\cdot x$ | $25.409$ (open) |
| $\sigma(\text{gate})$ | $\approx 1.000$, so $\mathrm{SiLU}(\text{gate}) \approx \text{gate} = 25.409$ |
| up $=u\cdot x$ | $-6.386$ |
| activation $a = 25.409 \times (-6.386)$ | $-162.26$ → the neuron adds $-w$ |
| $\lVert w\rVert$ | $0.543$ |
| $\lVert a\,w\rVert$ | $88.0$ |
| $\lVert\text{mlp\_out}\rVert$ | $254.2$ |
| $\cos(a\,w,\ \text{mlp\_out})$ | $0.83$ |
| neurons with open gate at this token | $495$ of $3072$ |
| reconstruction error (relative) | $3.2\times10^{-7}$ |

One neuron writes a vector about a third the size of the whole MLP output. Large last-layer activations like this one
matter when interpreting random controls (chapter 5.5).

## 2.6 Where to see this in the app

**Neuron Microscope**:
* **Left**: choose layer/neuron, or type `L27 n81` in the search box. Click tokens to choose the position.
* **Center**:
  * the activation and gate across tokens;
  * a scrollable grid of all 3,072 neurons × tokens, loaded in pages;
  * plots of $w$, $g$, $u$, $x$ and $a\,w$ with raw values.
* **Right**:
  * each step of §2.1 with real numbers, and hooked vs. recomputed values;
  * $x$ vs. the raw residual (§1.2);
  * the reconstruction check;
  * the top neurons by $\lVert a\,w\rVert$.

## 2.7 How to check it

* `test_inspector_recompute`: gate/up/SiLU/$a$ recomputed in float64 match the hooked tensors.
* `test_output_vector_is_down_proj_column`: $w_i$ really is column $i$ of `down_proj`.
* `test_reconstruction`, `test_reconstruction_with_bias`: §2.4, including that dropping the bias breaks it.
