# 8. Lab notebook

Dated findings from real runs. Each entry says what was run, what came out, and how much it can be trusted.
Entries are never edited after the fact; corrections are added as new entries.

---

### 2026-11-26 — What Qwen3-0.6B knows about the paper's examples

Next-token probabilities, real forward passes, float32 on CPU:

| Prompt | Top prediction | Paper's answer |
|---|---|---|
| The fourth Gospel was written by the apostle | **" John" 0.416** | John ✔ |
| In the Bible, the Gospel according to Saint | **" John" 0.581** | John ✔ |
| The password cracking tool is called John the | **" Rip" 0.718** | Ripper ✔ |
| Popular password cracking tools include Hashcat and | " Rainbow"; " John" only 0.002 (rank 65) | John ✘ |
| Jaromir Jagr plays the sport of | " football" 0.091 | hockey ✘ |
| Wayne Gretzky played the sport of | " hockey" 0.062 (barely top) | hockey ~ |
| The author of One Hundred Years of Solitude is Gabriel | " García" 0.978 | ✔ |
| To use the package beautifulsoup4 … import | " beautiful" 0.769 | bs4 ✘ (the paper's *post-removal* fallback answer) |
| The CEO of SpaceX and Tesla is Elon | " Musk" 0.982 | ✔ |

**Takeaway.** The small model doesn't know several of the paper's facts, including "John" as the answer for a password
cracker, Jágr → hockey, and bs4. Experiments that need those facts can't be run as specified on this model. We use
contexts the model does know.

---

### 2026-11-26 — Type 1 test of the strongest "John" candidate: inconclusive

**Setup.**
* Contexts (target `" John"`):
  * Gospel: "The fourth Gospel was written by the apostle"
  * Elton: "The British singer and pianist Elton"
  * Beatles: "The Beatles were Paul McCartney, George Harrison, Ringo Starr and"
* Candidate scan (direct logit attribution, minimum over contexts): the top candidate is **L27 n81 (−)**,
  with DLA 0.61 / 0.83 / 0.57.
* 20 matched random-direction controls, seed 0.

| Context | p(John) base → ablated | Δ log p | controls mean | z | status |
|---|---|---|---|---|---|
| Gospel | 0.416 → 0.272 | −0.426 | −0.236 | −0.52 | inconclusive |
| Elton | 0.868 → 0.826 | −0.050 | −0.153 | +0.73 | inconclusive |
| Beatles | 0.946 → 0.936 | −0.010 | −0.031 | +0.56 | inconclusive |

**Readout.** The logit-lens readout of $-w$ ("imiters", "postId", "udeau", "Puppy", …) doesn't mention John.

**Interpretation.**
* Ablating the neuron lowers p(John) in every context, but random directions of the same size, applied the same way,
  lower it about as much.
* The neuron's activation is very large ($a\approx-150$ to $-200$), so any perturbation of that size is disruptive.
* This neuron is therefore **not** shown to be a Type 1 "John" neuron.
* This doesn't contradict the paper, which used Llama-8B, but the paper's headline example doesn't carry over to the
  top direct-path candidate in Qwen3-0.6B.

**Open.**
* Candidates in middle layers act indirectly. They need the J-lens (Phase 3) to be found properly.
* The scan should be repeated with sign filtering, more contexts and more controls.

---

### 2026-11-26 — Logit lens across depth (Gospel prompt)

* `" Jesus"` is the top logit-lens token at layers 20–22.
* `" John"` takes over from layer 23 and ends at $p=0.42$.
* Before layer ~18 the lens shows function words (`" of"`, `","`, `" and"`), the usual early-layer incoherence.
* `" John"` ranks: 42,265 (embedding) → 834 (L4) → 30 (L9) → 6 (L21) → 1 (L24 onward).

---

### 2026-11-26 — J-lens fit for Qwen3-0.6B: timings and a CPU/GPU cross-check

* **CPU (4 cores, fp32):** 360 s and 316 s for the first two prompts. The run died when the idle container was paused
  (before its first checkpoint).
* **Modal A10G (fp32, dim_batch 32):** 11–15 s per prompt.
* **Cross-check:** the per-prompt Jacobian norms $\max_L \lVert J_L\rVert/\sqrt d$ for the same first prompts were **identical**
Vert/\sqrt d$ for the same first prompts were **identical**
  on CPU and GPU (8.907, 8.926, …). That's evidence the fit is deterministic and hardware-independent.
* **Full fit:** 1,000 WikiText prompts, Anthropic's `jlens` at commit 581d398. Launched detached on Modal
  (app ap-duPL4moboS6ko65jXgBGx7). It was cancelled after 25 prompts when the local client was killed. It was relaunched
  as a spawned call on the deployed app `neuronlab-jlens` (fc-01M8CH55PVG1DF61M0QJT7QSEM), resuming at prompt 26.

---

### 2026-11-26 — Full Atlas scan of Qwen3-0.6B

* **Corpus:** Pile-10k minus Books3, BookCorpus2, StackExchange and OpenWebText2 left **7,070 documents, exactly the
  paper's count** (Appendix B.1). 1,548,037 positions were scanned, position 0 excluded. The largest domains are Pile-CC
  (548k), PubMed Abstracts (314k), GitHub (179k) and Wikipedia (157k).
* **Cost and time:** 576 s on one A10G. The first attempt stalled in the CPU threshold step because the Modal function
  had no CPU request; it ran normally once given 8 cores.
* **Verification:** 16 recorded top firings from 8 random neurons were recomputed locally, one document at a time on CPU.
  The max relative error was $3.5\times10^{-6}$, and every one had an open gate.

### 2026-11-26 — A readout that says "John" is not a John neuron (L21 n29)

* **Search:** 15 neurons have `" John"` in their ±w logit-lens top 12. The top hit is **L21 n29 (+)**, where `" John"` is
  readout token #1.
* **Firings:** its strongest + firings are code punctuation (`*>(`, `(!`, `[`) in GitHub and HackerNews text, with
  activations around 60. It fires (gate open, +) on **10.3% of all tokens**.
* **Ablation:** zeroing it in its top firing context changes the prediction very little (KL 0.039, top-1 unchanged).
  The largest changes are to tokens that were already very unlikely.
* **Conclusion:** the readout is a hypothesis, not a classification. This neuron is not shown to "add John" in any
  context. This is the paper's own caution, and why the Atlas puts firings, readouts and a causal test side by side.

---

### 2026-11-26 — J-lens fitted for Qwen3-0.6B (1,000 prompts) and a first readout

* **Fit:**
  * code: Anthropic's `jlens` at commit 581d398;
  * data: 1,000 WikiText-103 train prompts × 128 tokens;
  * hardware: fp32 on a Modal A10G;
  * time: 11,741 s (11.7 s/prompt).
* **Convergence:** the relative change of the running mean per added prompt was about 0.2% at the end
  (`max_d_mean` 1.7e-3 to 2.0e-3).
* **Checks:**
  * 27 source layers, each 1024×1024 and all finite;
  * $\lVert J_L - I\rVert/\lVert I\rVert$ falls from 4.45 (L0) to 2.43 (L13) to 0.48 (L26), so late layers transport
    the residual almost unchanged, as expected.
* **Gospel prompt, last position, J-lens readout** (compare the logit-lens entry above):

| Layer | J-lens top tokens | " John" rank |
|---|---|---|
| L6 | Apostle, ,”, apost | 213 |
| L12 | Apostle, apost, 耶稣, Jesus, Christianity | 332 |
| L18 | Jesus, Apostle, Muhammad | 48 |
| L21 | Jesus, 耶稣, Matthew, Apostle | 8 |
| L24 | Jesus, **John**, 耶稣, Paul | 2 |
| L26 | **John**, Paul, Peter, Matthew, Jesus | 1 |

The logit lens showed only function words (" of", ",", " and") before layer ~18. The J-lens surfaces the topic
(apostle, Jesus, Christianity) from layer 6–12 on. That is the advantage the J-lens paper reports, and why the
neuron-basis paper reads neurons with it.

* **Stored:** `data/jlens/qwen3-0.6b-wikitext-n1000/` (not in git; `modal volume get neuronlab-jlens
  qwen3-0.6b-wikitext-n1000/jacobian_lens.pt …` restores it).

---

### 2026-11-27 — J-lens readouts vs. firings, and J-share of L14 n2000

* **L14 n2000:**
  * the **logit lens** of +w is noise ("lice", "geries", "撬");
  * the **J-lens** of +w is coherent: **spells, magical, vampire, magic, mystical, supernatural, Witch**;
  * but its strongest + firings are code and science tokens ("translated", "license", "-trained", "methyl"), and it
    fires (gate open, +) on **28%** of all tokens;
  * its −w J-lens readout is annotation words (注明, 标注, Remarks, 名单).
* **J-share of +w = 0.059.** Against 256 random directions at L14 (p1 0.036, median 0.062, p99 0.093) that is the
  **37.9th percentile: middle**.
* **Interpretation:** a coherent readout alone doesn't establish what a neuron does. Its firings don't match it, and
  its J-share is ordinary for the layer. It needs a causal test.

### 2026-11-27 — Validating linear predictions (Gospel prompt, inject the J-lens direction of " John")

| Layer | context slope ∂z/∂ε | corpus J-lens slope | ratio | notes |
|---|---|---|---|---|
| 8 | 0.0434 | 0.378 | 8.7 | the context Jacobian matches small ε (ε=−0.02‖h‖: predicted −0.026, observed −0.027) |
| 10 | −0.0231 | 0.348 | −15.1 | **opposite sign**: locally, this direction slightly lowers " John" in this prompt |
| 20 | 0.0252 | 0.265 | 10.5 | strong nonlinearity beyond about 0.1‖h‖ |

* The context Jacobian is a correct local predictor; its tests check it against finite differences.
* The corpus J-lens overestimates same-position effects about 9–10×, consistent with its estimator **summing over all
  later target positions**. At L10 it predicts the wrong sign for this single context.
* So the corpus J-lens describes typical directions across contexts. It is not this prompt's Jacobian, and it should
  not be used to predict effect sizes.

---

### 2026-11-27 — Type 3 scanner on Qwen3-0.6B (layers 6–24): phrase halves, not two topics

* **Padding rows.** The first scan was dominated by the 267 embedding padding rows (beyond the tokenizer). They were
  then excluded everywhere (see guide chapter 12).
* **Formatting clusters.** The second scan was dominated by groups of whitespace and punctuation tokens. The scanner now
  requires ≥ 3 word-like tokens per group, leaving 53,106 neuron-signs scored.
* **What the top candidates look like.** Most of the highest-scoring "two-concept" readouts are **two halves of one phrase**:

| Neuron | Group A | Group B | Likely phrase |
|---|---|---|---|
| L24 n2084 (−) | control | Panel | control panel |
| L22 n1669 (+) | mean | sample | sample mean |
| L23 n1680 (+) | replacement | without | without replacement |
| L24 n2733 (−) | boundary | initial | initial/boundary (conditions) |
| L23 n1522 (−) | controller | view | view controller |
| L23 n951 (−) | execution | control | execution control |

  These are not Type 3 in the paper's sense: one concept spread over two tokens is not two unrelated concepts. The
  readout-split heuristic can't tell the difference, which is why the paper confirms candidates with causal checks.
* **A more plausible mixed candidate:** L19 n2780 (−), with months (September, June, April, August) vs. document
  sections (Contents, Introduction). It went to the four checks; see the next entry.

---

### 2026-11-27 — J-share census of all 86,016 Qwen3-0.6B neurons (our Figure 2)

**Setup:**
* our J-lens (1,000 WikiText prompts);
* nonnegative OMP with exact NNLS refits, ≤ 25 directions;
* dictionary = 151,669 tokenizer tokens (the 267 padding rows excluded);
* 2,048 random unit directions per layer (seeded);
* sign used = the sign with the larger open-gate activation in the Atlas (as in Appendix C.2);
* run on Modal (A10G + 16 CPUs).

**Check:** a one-layer smoke test matched the local implementation to 7e-9.

| | Qwen3-0.6B | Paper (Llama-8B, Qwen3-8B/14B, OLMo-32B) |
|---|---|---|
| high (> random p99) | **15.9%** | 19–29% |
| middle | **67.0%** | 56–81% |
| low (< random p1) | **17.1%** | 1–21% |
| median neuron ÷ median random | **0.71–1.20** (by layer) | 1.02–1.26 |

**By depth (share high / low; median ratio):**

| Layers | high | low | median ratio |
|---|---|---|---|
| L0–L8 | 4–9% | 19–29% | 0.71–0.82 |
| L9–L16 | 5–18% | 18–32% | 0.73–0.86 |
| L17–L19 | 15–22% | 15–17% | 0.90–0.95 |
| **L20–L24** | **27–40%** | **1–4%** | **1.11–1.20** |
| L25–L27 | 15–29% | 1–9% | 0.96–1.01 |

**Interpretation:**
* **The late layers (L20–24) match the paper's picture:** many high-J-share neurons, almost no low ones, and neurons
  more verbalizable than random directions.
* **The early and middle layers do not:** the median neuron is *less* verbalizable than a random direction, and a
  quarter are "low".
* **This is a disagreement with the paper's headline (median ratio 1.02–1.26) for this model.** Candidate explanations,
  none tested yet:
  1. small models write more non-verbal, structural directions early;
  2. our lens (1,000 WikiText prompts) differs from the reference fits;
  3. early-layer J-lens directions span a low-rank subspace (noted by the J-lens paper), so random directions and
     neurons are compared in a space the lens barely covers;
  4. the paper notes that low-J-share neurons tend to align with later attention queries (Appendix A.3), which we
     haven't measured.
* **Next test:** query alignment (A.3) for the low-J-share neurons.
