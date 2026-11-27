"""Phase 4: job runner, corpus helpers, Type 3 verdict rules, scanner split."""

import time

import numpy as np
import pytest

from api.jobs import Cancelled, JobManager
from datasets_nl import corpus
from experiments.type3 import PAPER, verdict
from experiments.type3_scan import _split


def _wait(j, t=5.0):
    end = time.time() + t
    while j.status in ("queued", "running") and time.time() < end:
        time.sleep(0.01)
    return j


def test_jobs_success_failure_cancel():
    jm = JobManager()
    ok = _wait(jm.submit("t", {}, lambda job: (job.report(0.5, "half"), {"x": 1})[1]))
    assert ok.status == "done" and ok.result == {"x": 1} and ok.progress == 1.0
    bad = _wait(jm.submit("t", {}, lambda job: 1 / 0))
    assert bad.status == "failed" and "ZeroDivisionError" in bad.error

    def slow(job):
        for i in range(1000):
            job.report(i / 1000)
            time.sleep(0.005)
        return {}
    s = jm.submit("t", {}, slow)
    time.sleep(0.05)
    jm.cancel(s.id)
    assert _wait(s).status == "cancelled"
    with pytest.raises(KeyError):
        jm.get("nope")


class _Tok:
    vocab = {" Paris": [7], " New York": [11, 12], "x": [3], "注明": [21, 22], " 注明": [30, 21]}
    def encode(self, s, add_special_tokens=False): return self.vocab.get(s, [99])
    def decode(self, ids): return {5: "ing", 6: " walk", 8: "Ing", 9: "."}.get(ids[0], "?")


def test_word_tokens_and_continuations():
    assert corpus.word_first_tokens(_Tok(), ["Paris", "New York", " Paris "]) == [7, 11]
    assert corpus.word_first_tokens(_Tok(), ["注明"]) == [21]  # CJK: no leading space
    t = _Tok()
    assert corpus.is_word_continuation(t, 5) and not corpus.is_word_continuation(t, 6)
    assert not corpus.is_word_continuation(t, 8) and not corpus.is_word_continuation(t, 9)


def _out(c1=(True, True), c2=(0.1, 0.1), z=(5.0, 5.0), c3=(True, True), c4=(True, True), strong=(10, 10)):
    T = {}
    for i, k in enumerate("AB"):
        T[k] = {"check1": {"pass": c1[i], "n_strong_docs": strong[i]},
                "cross": {"n_positions": 30,
                          "check2": {"drop_other_minus_random": c2[i], "z": z[i],
                                     "pass": c2[i] >= PAPER["min_drop_nats"] and z[i] >= PAPER["min_z"]},
                          "check3": {"pass": c3[i]}, "check4": {"pass": c4[i]}}}
    return {"topics": T}


def test_type3_verdict_rules():
    assert verdict(_out())["status"] == "supported"
    assert verdict(_out(c2=(0.1, -0.01)))["status"] == "contradicted"
    assert verdict(_out(c2=(0.1, 0.02)))["status"] == "inconclusive"   # below 0.03 nats
    assert verdict(_out(z=(5.0, 2.0)))["status"] == "inconclusive"      # below z = 3
    assert verdict(_out(c3=(True, False)))["status"] == "contradicted"  # category-level effect
    assert verdict(_out(c4=(False, True)))["status"] == "contradicted"  # word pieces
    assert verdict(_out(c1=(False, True), strong=(2, 10)))["status"] == "insufficient_data"


def test_scanner_split_finds_two_blocks():
    S = np.full((8, 8), 0.05)
    S[:4, :4] = 0.9
    S[4:, 4:] = 0.8
    np.fill_diagonal(S, 1.0)
    score, g = _split(S)
    assert set(np.where(g)[0]) in ({0, 1, 2, 3}, {4, 5, 6, 7}) and score > 0.7
    _split(np.full((4, 4), 0.5))  # degenerate input must not crash


def test_paper_constants():
    assert (PAPER["max_topic_docs"], PAPER["n_ablate_docs"], PAPER["n_random_words"], PAPER["n_random_dirs"]) == (150, 40, 200, 20)
    assert (PAPER["min_strong_docs"], PAPER["min_share_margin"], PAPER["min_drop_nats"], PAPER["min_z"]) == (5, 0.07, 0.03, 3.0)
    assert (PAPER["check3_ratio"], PAPER["check4_top"], PAPER["check4_max_continuation_share"]) == (0.25, 15, 0.5)


def test_pile10k_filter_count():
    try:
        docs = corpus.pile10k()
    except Exception as e:  # dataset unavailable offline
        pytest.skip(f"Pile-10k not available: {e}")
    assert len(docs) == 7070  # paper, Appendix B.1
    assert not ({d for _, d in docs} & corpus.EXCLUDED)
