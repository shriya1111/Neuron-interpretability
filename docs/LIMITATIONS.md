# Research limitations and differences from the paper

This file is kept current as phases land. It lists everything that could make a NeuronLab
result differ from the paper's, so that nothing is mislabelled as a reproduction.

## Current state (Phase 1)

* **No paper result has been reproduced yet.** Phase 1 runs on Qwen2.5-0.5B / Qwen3-0.6B,
  which the paper did not study. Paper neuron IDs are refused for any other checkpoint.
* **No lenses yet.** The logit lens (Phase 2), J-lens and J-share (Phase 3) are not built.
  Nothing in the UI shows a vocabulary readout of a neuron yet.
* **Attention heads** can be selected and their real attention probabilities shown, but there are
  no head-level interventions yet.
* **Precision.** CPU runs use float32, so reconstruction errors are about 1e-7 relative. On GPU in bf16
  (Llama-8B), expect errors around 1e-3 relative. Every response reports its actual error.
* **Attention implementation.** Models are loaded with `attn_implementation="eager"` so hooks and
  attention capture are exact and deterministic. It is slower than SDPA/flash attention but gives the
  same function up to floating point.
* **Transformers hooks.** Transformers ≥ 5 permanently installs its own "output capturing" hooks the
  first time attention weights are requested. NeuronLab reports them under `/health → foreign_hooks`,
  and a test verifies that they don't change logits.

## Known differences from the paper (to be addressed in later phases)

* **J-lens artifacts.** The paper says it used "the J-lens files released with Gurnee et al. (2026)".
  The publicly available Llama-3.1-8B-Instruct J-lens (`huggingface.co/neuronpedia/jacobian-lens`) was
  fitted by Neuronpedia with Anthropic's official `jlens` code on WikiText-103: 461 prompts × 128 tokens,
  bf16, stopped early at Δ=0.002. The J-lens paper's own lenses used 1,000 pretraining-like prompts.
  We can't confirm these are the same files the paper used, so results will be labelled
  "published third-party J-lens".
* **Unspecified details.** The paper doesn't specify the exact nonnegative-pursuit hyperparameters,
  the Claude-judge prompts and topic term lists (Appendix B), the hand-assigned neuron groupings
  (Appendix D), the random-word lists, or the pools of made-up names. NeuronLab will make these explicit,
  editable inputs, record them with every run, and treat LLM/human labels as annotations, never as evidence.
* **Pile-10k.** Appendix B and C use Pile-10k minus four subsets (7,070 docs). It needs to be downloadable
  from the GPU host (`NeelNanda/pile-10k` on Hugging Face).
* **Sample sizes.** Interactive runs use reduced sample sizes. They will be labelled `exploratory`, never
  `paper_reproduction`.
