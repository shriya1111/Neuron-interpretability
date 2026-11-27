# 12. Polysemanticity Types 2 and 3, the scanner, and background jobs

Recap of the taxonomy (chapter 5):

| Type | Description | Paper's finding |
|---|---|---|
| 1 | many unrelated inputs, one concept | likely common |
| **2** | **each sign** adds a different concept | exists |
| **3** | **one sign** adds **two unrelated concepts** | rare: ≤ ~1% of signed directions with an effect |

All three tabs live in **Polysemanticity Lab**. Types 2, 3 and the scanner run as **background jobs** (§12.4).

## 12.1 Type 2: two signs, two concepts

**Contexts.** For each sign, the Atlas's strongest firings of that sign, all with an **open gate**. That rules out
closed-gate leakage by construction. Each context runs as exact token ids up to the firing token.

**Words.** Each sign's words default to its J-lens readout tokens with ≥ 2 letters, in any script; you can override
them. If a sign has no usable words, the test refuses to run rather than measuring a placeholder. Words are measured by
their first token **as they appear mid-text**: " word" for space-separated scripts, and "word" with no space for
Chinese, Japanese and Thai.

> **Bug found by a real run.** The first version only accepted " word" tokens. For L14 n2000, whose −w readout is
> Chinese (注明, 标注…), it found no words and silently measured a placeholder. Both the picker and the CJK tokenization
> were fixed and tested.

**Per sign** (read at the firing position, the neuron zeroed at every position):

$$E_{\text{own}} = \overline{\Delta \log p}(\text{own words}) - \overline{\Delta \log p}(\text{200 random words})$$

Here $\Delta\log p$ = base − ablated, so a positive value means the words were *lowered* by removing the neuron.
$E_{\text{other}}$ is the same for the other sign's words, and should be about 0 for a clean Type 2 neuron.

**Controls.** $N$ random directions with $\lVert r\rVert=\lVert w\rVert$, each adding $-a_t r$ at every position.

**A sign "adds its concept"** when $E_{\text{own}}\ge 0.03$ nats and $z\ge 3$ against the controls (the Appendix B
check 2 thresholds). **Type 2 is supported** when both signs pass. Whether the two concepts are unrelated is your judgment.

**Steering comparison.** On 8 neutral prompts, the activation is set at every position to $+A$ (the largest open-gate
+ activation in Pile-10k), to $-A'$, or to 0, and the change in each sign's words is reported. A clean Type 2 neuron:
+ steering raises + words only, and − steering raises − words only.

## 12.2 Type 3: the four checks (Appendix B.1), implemented as written

| Check | NeuronLab implementation |
|---|---|
| **Corpus** | Pile-10k minus Books3, BookCorpus2, StackExchange, OpenWebText2. **7,070 documents** (tested). |
| **Topic docs** | documents matching any search term (case-insensitive, whole word), ≤ 150, a 1,500-char window around the first match, the first 256 tokens |
| **Strong activation** | $s\cdot a > q_{99.9}$ of $s\cdot a$ over all positions of the 200 reference docs (the same seeded docs the Atlas used; per-layer maxima cached in `data/cache/`) |
| **1** | strong in ≥ 5 docs of each topic **and** topic share ≥ reference share + 0.07 |
| **2** | zero the neuron at every position of ≤ 40 strongly-activating docs of topic X. At strong positions: drop in log-prob of the first tokens of the **other** topic's words − drop for 200 random words. Must be ≥ 0.03 nats **and** z ≥ 3 vs 20 random directions (each subtracting $a_t r$, $\lVert r\rVert=\lVert w\rVert$). Required **both ways**. |
| **3** | category-control words (yours; the paper used a Claude judge) must **not** drop by ≥ ¼ as much as the other topic's words with z ≥ 3 |
| **4** | of the 15 tokens with the largest mean drop, ≤ half of the positive change may fall on word continuations (no leading space, lowercase first letter) |

**Verdict:**
* *supported*: all four checks pass both ways;
* *contradicted*: check 2's effect ≤ 0, or check 3 or 4 fails;
* *inconclusive*: positive but below the thresholds;
* *insufficient data*: fewer than 5 strong documents for a topic.

**Sample sizes.** The paper's sizes are available behind a checkbox. They take about 20–30 minutes on CPU. The
default exploratory run uses 60 topic docs, 12 ablated docs and 10 random directions, and results are labelled
`reduced_sample_sizes (exploratory)`.

**Default topics.** These are the paper's Llama example (Elon Musk + protein translation). That neuron ID does not exist in
Qwen3, so find Qwen3 candidates with the scanner.

## 12.3 Scanner (Appendix B.3 heuristics): candidates only

| Heuristic | How |
|---|---|
| readout split | J-lens top-12 tokens of ±w, cosine matrix of their J-lens directions, split by the leading eigenvector; score = mean within − mean between. Both groups need ≥ 3 tokens and no shared spelling. |
| pair reconstruction | R² of fitting $w$ with its best pair among the 8 most-aligned same-layer neurons |
| firing split | the Atlas's top-firing peak tokens (unit embeddings) split the same way |

The paper's search over Llama used these heuristics to go from 44,000 flagged neurons to 54, then ran the four checks.
**A high scanner score is not evidence.** Load the candidate into the Type 3 tab and run the checks.

**Pitfall found and fixed.** The first scan's top candidates all had a group of empty strings. Qwen3-0.6B's
embedding matrix has 151,936 rows, but the tokenizer only defines 151,669 tokens. The 267 extra rows are unused padding
with near-identical vectors, and they appeared in 0.2% of J-lens readouts. They are now excluded everywhere: readouts,
neuron search, J-share dictionaries, the precomputed readouts and the census. A test guards this.

## 12.4 Background jobs

* `POST /experiments/type2`, `/experiments/type3` and `/experiments/type3/scan` return a `job_id` immediately.
* Track it with `GET /experiments/{id}` (status, progress 0–1, message, result), or with **WebSocket**
  `/experiments/{id}/stream`, which pushes updates every 0.5 s.
* Cancel with `POST /experiments/{id}/cancel`. The job stops at its next progress checkpoint.
* Results carry full provenance and `hooks_remaining`.
