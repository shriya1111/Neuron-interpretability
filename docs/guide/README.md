# NeuronLab Guide

A running, plain-language explanation of everything NeuronLab computes. It covers the model's
layers, the maths behind each feature, how to use the features, and how to check that they are correct.
It is updated with every phase; see the [changelog](#changelog) at the bottom.

> Formulas use LaTeX notation (`$…$`), which GitHub renders as maths.

## Chapters

| # | Chapter | What you will learn |
|---|---|---|
| 1 | [The model, layer by layer](01-model-architecture.md) | What happens to a prompt inside Qwen3-0.6B, with real tensor shapes |
| 2 | [Neurons](02-neurons.md) | What one MLP neuron computes, why it has two signs, the "closed gate", and how its contributions add up |
| 3 | [Interventions and controls](03-interventions.md) | How we switch neurons off or change them, why baseline and modified runs are separate, and how random controls work |
| 4 | [Logit lens](04-logit-lens.md) | Reading a neuron's output vector as words; reading the residual stream at each layer; direct logit attribution |
| 5 | [Polysemanticity, Type 1](05-polysemanticity-type1.md) | The paper's taxonomy, our experimental protocol, statistics and verdicts |
| 6 | [Features and API](06-features.md) | Every page of the app and every backend endpoint |
| 7 | [Testing and verification](07-testing.md) | How to run the tests, what each one proves, and how to check results yourself |
| 8 | [Lab notebook](08-lab-notebook.md) | Dated findings from real runs, including ones that disagree with the paper |
| 9 | [Weight editing](09-weight-editing.md) | Pruning and rewriting neurons for real: exact undo, comparison with the original, held-out perplexity, export |
| 10 | [Neuron Atlas](10-atlas.md) | The per-neuron catalog: corpus scan, firings, readouts, search, and one-click causal tests |

## Notation used throughout

| Symbol | Meaning | Qwen3-0.6B size |
|---|---|---|
| $T$ | number of tokens in the prompt | varies |
| $d$ | residual stream width (`hidden_size`) | 1024 |
| $m$ | MLP neurons per layer (`intermediate_size`) | 3072 |
| $V$ | vocabulary size | 151,936 |
| $L$ | layer index, $0 \dots 27$ | 28 layers |
| $h_L \in \mathbb{R}^d$ | residual stream after block $L$ at one token | |
| $x \in \mathbb{R}^d$ | the **normalized** MLP input (not the raw residual) | |
| $g_i, u_i \in \mathbb{R}^d$ | neuron $i$'s rows of `gate_proj` and `up_proj` | |
| $w_i \in \mathbb{R}^d$ | neuron $i$'s **output vector**, column $i$ of `down_proj` | |
| $a_i \in \mathbb{R}$ | neuron $i$'s activation (can be negative) | |
| $W_U \in \mathbb{R}^{V\times d}$ | unembedding matrix (tied to the embedding in Qwen3-0.6B) | |
| $\gamma$ | learned gain of an RMSNorm | $\mathbb{R}^d$ |

## Which model, and what our results mean

* **Primary model: `Qwen/Qwen3-0.6B`.** It belongs to the same family as the paper's Qwen3-8B and Qwen3-14B.
  **Cross-check model: `Qwen/Qwen2.5-0.5B`**, used to see whether a finding is a quirk of one model.
* The paper's experiments were on 8B–32B models. We apply **the paper's methods to smaller models**,
  so our results are *replications of the method*, not *reproductions of the paper's numbers*. The paper's
  neuron IDs (such as Llama L28 n2760) do not exist in these models, and the code refuses to use them.
* Disagreeing with the paper is an allowed and useful outcome. The app is built to show it.

## Changelog

| Date | Phase | Changes to the guide |
|---|---|---|
| 2026-11-26 | 1 | Chapters 1–3, 6, 7: architecture, neuron maths, interventions, microscope and explorer, tests |
| 2026-11-26 | 2 | Switched primary model to Qwen3-0.6B. Added chapters 4 (logit lens), 5 (Type 1 experiment) and 8 (lab notebook); updated 3, 6 and 7 |
| 2026-11-26 | Atlas + editor | Added chapters 9 (weight editing) and 10 (Neuron Atlas) |
