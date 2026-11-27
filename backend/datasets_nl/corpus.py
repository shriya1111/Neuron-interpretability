"""Pile-10k as filtered in the paper (Appendix B.1) and topic documents.

* Pile-10k = NeelNanda/pile-10k (first 10,000 Pile documents) minus the Books3, BookCorpus2, StackExchange and
  OpenWebText2 subsets -> 7,070 documents (paper's count; also what cloud/modal_atlas.py finds).
* Topic documents: Pile-10k documents matching any term of a topic's term list (case-insensitive, whole word),
  at most 150, as a 1,500-character window around the first match, truncated to the first 256 tokens.
* Reference ("random") documents: the same 200 documents the Atlas used for its thresholds
  (torch.randperm(n_docs, seed 0)[:200] over the filtered list).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

EXCLUDED = {"Books3", "BookCorpus2", "StackExchange", "OpenWebText2"}
WINDOW_CHARS = 1500
MAX_TOPIC_DOCS = 150
N_TOKENS = 256


@lru_cache(maxsize=1)
def pile10k() -> tuple[tuple[str, str], ...]:
    """(text, domain) for the 7,070 kept documents, in dataset order."""
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    out = []
    for row in ds:
        name = (row.get("meta") or {}).get("pile_set_name", "unknown")
        if name in EXCLUDED or len(row["text"].strip()) == 0:
            continue
        out.append((row["text"], name))
    return tuple(out)


def topic_docs(terms: list[str], max_docs: int = MAX_TOPIC_DOCS, window: int = WINDOW_CHARS) -> list[dict]:
    terms = [t.strip() for t in terms if t.strip()]
    if not terms:
        raise ValueError("topic needs at least one search term")
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    out = []
    for i, (text, dom) in enumerate(pile10k()):
        m = pat.search(text)
        if not m:
            continue
        s = max(0, m.start() - window // 2)
        out.append({"doc": i, "domain": dom, "match": m.group(0), "text": text[s:s + window]})
        if len(out) >= max_docs:
            break
    return out


def reference_docs(tokenizer, n: int = 200, seed: int = 0) -> list[dict]:
    """The Atlas's reference documents (first 256 tokens)."""
    import torch

    docs = pile10k()
    # the Atlas dropped documents that tokenized to < 2 tokens; mirror that before sampling
    kept = [i for i, (t, _) in enumerate(docs) if len(tokenizer(t[:4000], add_special_tokens=False).input_ids) >= 2]
    g = torch.Generator().manual_seed(seed)
    idx = sorted(torch.randperm(len(kept), generator=g)[:n].tolist())
    return [{"doc": kept[i], "domain": docs[kept[i]][1], "text": docs[kept[i]][0]} for i in idx]


def tokenize(tokenizer, text: str, n_tokens: int = N_TOKENS):
    import torch
    ids = tokenizer(text, add_special_tokens=False).input_ids[:n_tokens]
    return torch.tensor(ids, dtype=torch.long)


def word_first_tokens(tokenizer, words: list[str]) -> list[int]:
    """First token of ' word' for each word (the paper measures the first tokens of a topic's words)."""
    out = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if ids:
            out.append(ids[0])
    return sorted(set(out))


@lru_cache(maxsize=4)
def random_word_tokens(tokenizer, n: int = 200, seed: int = 0) -> tuple[int, ...]:
    """n random whole-word tokens: decode to ' ' + 3+ lowercase letters (seeded)."""
    import random
    rng = random.Random(seed)
    V = len(tokenizer)
    cand = []
    for i in range(V):
        s = tokenizer.decode([i])
        if len(s) >= 4 and s[0] == " " and s[1:].isalpha() and s[1:].islower() and s[1:].isascii():
            cand.append(i)
    return tuple(sorted(rng.sample(cand, min(n, len(cand)))))


def is_word_continuation(tokenizer, tid: int) -> bool:
    """Appendix B check 4: a token without a leading space that starts with a lowercase letter."""
    s = tokenizer.decode([tid])
    return len(s) > 0 and s[0].isalpha() and s[0].islower()
