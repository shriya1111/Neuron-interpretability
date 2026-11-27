"""Atlas reader on a synthetic atlas with known contents."""

import json

import numpy as np
import pytest

from interpretability.atlas import AtlasStore


class _Tok:
    def decode(self, ids): return f"<{ids[0]}>"
    def __len__(self): return 50


@pytest.fixture
def store(tmp_path):
    T, D, N, K = 8, 3, 4, 2
    tokens = np.arange(D * T, dtype=np.int32).reshape(D, T) % 50
    tokens[2, 5:] = -1  # doc 2 has 5 tokens
    np.save(tmp_path / "tokens.npy", tokens)
    (tmp_path / "docs.json").write_text(json.dumps({"n_docs": D, "domains": ["A", "B"], "doc_domain": [0, 1, 1], "lengths": [8, 8, 5]}))
    (tmp_path / "meta.json").write_text(json.dumps({"model_id": "m", "tokens_per_doc": T, "positions_scanned_excl_pos0": 18,
                                                    "positions_per_domain": {"A": 7, "B": 11}, "n_docs": D}))
    for L in range(2):
        np.savez(tmp_path / f"L{L:02d}.npz",
                 top_pos_val=np.array([[5.0, 2.0]] * N, np.float32), top_pos_idx=np.array([[1 * T + 3, 4]] + [[-1, -1]] * (N - 1)),
                 top_neg_val=np.array([[-4.0, -1.0]] * N, np.float32), top_neg_idx=np.array([[2 * T + 4, 5]] * N),
                 n_open_pos=np.array([9, 0, 1, 2]), n_open_neg=np.array([0, 3, 0, 0]),
                 n_strong_pos=np.array([[1, 2]] * N), n_strong_neg=np.array([[0, 1]] * N),
                 thr_pos=np.ones(N, np.float32), thr_neg=np.ones(N, np.float32),
                 thr_pos_nz=np.ones(N, np.float32), thr_neg_nz=np.ones(N, np.float32),
                 max_abs=np.array([5.0, 1.0, 3.0, 2.0], np.float32),
                 read_pos_ids=np.array([[7, 8, 9]] * N), read_pos_score=np.ones((N, 3), np.float32),
                 read_neg_ids=np.array([[10, 11, 12]] * N), read_neg_score=np.ones((N, 3), np.float32))
    return AtlasStore(str(tmp_path)), tokens, T


def test_neuron_snippets_and_rates(store):
    st, tokens, T = store
    n = st.neuron(_Tok(), 0, 0, k=2)
    f = n["plus"]["firings"][0]
    assert (f["doc"], f["position"], f["activation"]) == (1, 3, 5.0)
    assert f["tokens"][f["peak_index"]] == f"<{tokens[1, 3]}>"
    assert f["prefix_token_ids"] == tokens[1, :4].tolist()
    g = n["minus"]["firings"][0]
    assert (g["doc"], g["position"]) == (2, 4) and len(g["tokens"]) == g["peak_index"] + 1  # doc 2 ends at 5 tokens
    assert n["plus"]["fire_rate"] == pytest.approx(9 / 18)
    assert n["plus"]["strong_rate_by_domain"] == pytest.approx({"A": 1 / 7, "B": 2 / 11})
    assert [w["text"] for w in n["minus"]["writes_logit_lens"]] == ["<10>", "<11>", "<12>"]
    # neuron 1 has no positive firings recorded (-1 index) -> skipped, not crashed
    assert st.neuron(_Tok(), 0, 1)["plus"]["firings"] == []


def test_layer_table_sort_and_validation(store):
    st, _, _ = store
    rows = st.layer_table(_Tok(), 1, sort="max_abs")["rows"]
    assert [r["neuron"] for r in rows] == [0, 2, 3, 1]
    assert [r["neuron"] for r in st.layer_table(_Tok(), 1, sort="fire_rate")["rows"]][:2] == [0, 1]
    with pytest.raises(ValueError):
        st.layer(5)
    with pytest.raises(ValueError):
        st.neuron(_Tok(), 0, 99)
    with pytest.raises(ValueError):
        st.layer_table(_Tok(), 0, sort="nope")


def test_search_word(store):
    st, _, _ = store
    hits = st.search_word(_Tok(), "<8>")
    assert {(h["layer"], h["sign"], h["rank_in_readout"]) for h in hits} == {(0, "+", 2), (1, "+", 2)}
    assert st.search_word(_Tok(), "zzz") == []


def test_summary_card(store):
    st, tokens, T = store
    s = st.summary(_Tok(), 0, 0, k=2)
    assert s["plus"]["top_firing"]["activation"] == 5.0 and "prefix_token_ids" not in s["plus"]["top_firing"]
    assert s["plus"]["logit_lens"] == ["<7>", "<8>"] and s["minus"]["fire_rate"] == 0.0
    assert st.summary(_Tok(), 0, 1)["plus"]["top_firing"] is None
    with pytest.raises(ValueError):
        st.summary(_Tok(), 0, 99)
