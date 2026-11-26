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
