# 1. The model, layer by layer

This chapter follows one prompt through **Qwen3-0.6B**. Every shape below is what NeuronLab reads from
the loaded checkpoint (`GET /models/config`). Nothing is hard-coded.

## 1.1 The numbers

| Quantity | Config name | Qwen3-0.6B | Qwen2.5-0.5B |
|---|---|---|---|
| Layers (transformer blocks) | `num_hidden_layers` | 28 | 24 |
| Residual width $d$ | `hidden_size` | 1024 | 896 |
| MLP neurons per layer $m$ | `intermediate_size` | 3072 | 4864 |
| Total MLP neurons | $28 \times 3072$ | 86,016 | 116,736 |
| Attention (query) heads | `num_attention_heads` | 16 | 14 |
| Key/value heads (GQA) | `num_key_value_heads` | 8 | 2 |
| Head size | `head_dim` | 128 | 64 |
| Vocabulary $V$ | `vocab_size` | 151,936 | 151,936 |
| Embedding tied to unembedding | `tie_word_embeddings` | yes | yes |
| Extra per-head norm on q and k | | yes ("qk-norm") | no (has q/k/v biases instead) |

## 1.2 The pipeline

```
text ──tokenizer──► token ids (T)
     ──embedding──► h_0  [T × d]
     ──block 0──► h_1 ──block 1──► … ──block 27──► h_28
     ──final RMSNorm──► ──unembedding W_U──► logits [T × V] ──softmax──► next-token probabilities
```

### Tokenization
The tokenizer splits text into pieces from a fixed vocabulary of 151,936 tokens. **Leading spaces are part of
the token**: `" John"` (id 3757) and `"John"` (id 13079) are different tokens. The UI shows spaces as `·` so this
is always visible.

### Embedding
Each token id selects one row of the embedding matrix $W_E \in \mathbb{R}^{V \times d}$. That row is the starting
residual stream $h_0$ at that position.

### The residual stream
The residual stream is the running sum each block reads from and writes to:

$$h_{L+1} = h_L + \underbrace{\mathrm{Attn}_L(\cdot)}_{\text{attention output}} + \underbrace{\mathrm{MLP}_L(\cdot)}_{\text{MLP output}}$$

Because blocks *add* to it, anything a neuron writes stays there unless a later layer writes over it.
This is why the paper can ask whether a neuron's output keeps its meaning at other layers (Phase 5).

### One block, exactly as computed

$$
\begin{aligned}
\text{resid\_pre} &= h_L \\
\text{attn\_out} &= \mathrm{Attn}\big(\mathrm{RMSNorm}_{\text{in}}(h_L)\big) \\
\text{resid\_mid} &= h_L + \text{attn\_out} \\
x &= \mathrm{RMSNorm}_{\text{post}}(\text{resid\_mid}) \quad\text{(the actual MLP input)} \\
\text{mlp\_out} &= W_{\text{down}}\Big(\mathrm{SiLU}(W_{\text{gate}}\,x) \odot (W_{\text{up}}\,x)\Big) \\
h_{L+1} = \text{resid\_post} &= \text{resid\_mid} + \text{mlp\_out}
\end{aligned}
$$

NeuronLab can capture every one of these tensors (`/activations/capture`, `/neurons/inspect`).

### RMSNorm
$$\mathrm{RMSNorm}(v) = \gamma \odot \frac{v}{\sqrt{\tfrac1d\sum_k v_k^2 + \varepsilon}}$$
It rescales the vector to a fixed root-mean-square size, then multiplies each coordinate by a learned gain $\gamma$.
Two consequences matter for interpretability:
1. **The MLP never sees the raw residual stream.** It sees a rescaled, reweighted copy $x$. The microscope recomputes
   $x$ from `resid_mid` and shows the difference.
2. **The final norm's $\gamma$ must be folded into the unembedding** when reading directions as words
   (chapter 4). The paper does this.

### Attention (grouped-query)
Each of the 16 query heads computes $\mathrm{softmax}\big(q\,k^\top/\sqrt{128}\big)$ over earlier positions (causal mask)
and moves a weighted mix of value vectors. Pairs of query heads share one of the 8 key/value heads ("GQA").
Qwen3 normalizes $q$ and $k$ per head before the dot product. Rotary position embeddings (RoPE) encode position.
The Model Explorer shows each head's real attention pattern for any layer.

### MLP (SwiGLU)
Shapes per layer: $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{3072\times1024}$ and $W_{\text{down}} \in \mathbb{R}^{1024\times3072}$.
Each of the 3,072 rows/columns is one **neuron**. Chapter 2 explains this in detail.

### Final norm, unembedding, probabilities
$$\text{logits} = W_U\,\mathrm{RMSNorm}_{\text{final}}(h_{28}), \qquad p = \mathrm{softmax}(\text{logits})$$
In Qwen3-0.6B, $W_U$ *is* the embedding matrix $W_E$ ("tied embeddings"). Llama-3.1-8B, the paper's model,
has a separate $W_U$.

## 1.3 Where to see this in the app

* **Model Explorer → Architecture**: the graph shows all 28 blocks with the actual norms
  $\lVert h\rVert$, $\lVert\text{attn\_out}\rVert$, $\lVert\text{mlp\_out}\rVert$ and the fraction of open gates at the selected token.
  Click a block to see its internals.
* **Layer × token heatmap**: one metric for every (layer, token). The first token usually has norms that are orders of
  magnitude larger (an "attention sink"), so the default color scale is logarithmic.
* **Attention**: tick "capture attention pattern" and choose a head.

## 1.4 How to check it

* The architecture card is built from module shapes at load time; `test_dimensions_read_from_model` checks this.
* `test_hooked_matches_plain` proves that instrumenting the model does not change its output (bit-for-bit).
* `test_attention_capture_and_foreign_hooks` checks that attention rows sum to 1 and respect the causal mask.
