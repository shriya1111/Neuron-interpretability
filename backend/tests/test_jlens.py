"""J-lens readouts, J-share pursuit and linear validation on tiny models with synthetic Jacobians."""

import numpy as np
import pytest
import torch
from scipy.optimize import nnls

from interpretability.j_lens import JLens
from interpretability.j_share import classify, j_share, j_share_batch, random_baseline
from interpretability.validation import validate


class _Tok:
    def decode(self, x): return f"<{x[0]}>"
    def convert_ids_to_tokens(self, i): return f"t{i}"


@pytest.fixture
def lens(tiny, tmp_path):
    g = torch.Generator().manual_seed(0)
    d = tiny.adapter.d_model
    jac = {L: torch.eye(d) + 0.3 * torch.randn(d, d, generator=g) for L in range(tiny.adapter.n_layers - 1)}
    return JLens(tiny.adapter, path=str(tmp_path), jacobians=jac)


def test_directions_definition(tiny, lens):
    ad = tiny.adapter
    WUg = ad.unembed.weight.detach().float() * ad.final_norm.weight.detach().float()
    D = lens.directions(1)
    ref = WUg @ lens.jacobians[1]
    assert torch.allclose(D, ref / ref.norm(dim=1, keepdim=True), atol=1e-5)
    assert torch.allclose(D.norm(dim=1), torch.ones(D.shape[0]), atol=1e-5)
    last = lens.directions(ad.n_layers - 1)  # last layer uses the unembedding
    assert torch.allclose(last, WUg / WUg.norm(dim=1, keepdim=True), atol=1e-5)
    with pytest.raises(ValueError):
        lens.directions(ad.n_layers)


def test_readout_signs(tiny, lens):
    w = tiny.adapter.output_vector(1, 3)
    r = lens.readout(_Tok(), 1, w, k=5, targets=[7])
    assert r["plus"]["targets"][0]["cosine"] == pytest.approx(-r["minus"]["targets"][0]["cosine"], abs=1e-6)
    top = r["plus"]["top"][0]["id"]
    D = lens.directions(1)
    assert top == int((D @ (w / w.norm())).argmax())


def test_j_share_properties(lens):
    D = lens.directions(0)
    g = torch.Generator().manual_seed(1)
    w = torch.randn(D.shape[1], generator=g)
    r = j_share(D, w, k=25)
    assert r["k_used"] <= 25 and r["all_coefficients_nonnegative"] and 0 <= r["j_share"] <= 1
    # optimality on the chosen support: equals an independent NNLS solve
    A = D[r["token_ids"]].double().T.numpy(); wn = (w / w.norm()).double().numpy()
    c, _ = nnls(A, wn)
    assert 1 - np.sum((wn - A @ c) ** 2) == pytest.approx(r["j_share"], abs=1e-8)
    # a vector that IS a token direction is fully verbalizable with one direction
    r1 = j_share(D, D[5].clone() * 3.0, k=25)
    assert r1["j_share"] == pytest.approx(1.0, abs=1e-6) and r1["token_ids"][0] == 5
    # more directions never explain less
    assert j_share(D, w, k=5)["j_share"] <= r["j_share"] + 1e-12
    # batch implementation agrees with the single-vector one
    W = torch.randn(D.shape[1], 6, generator=g)
    b = j_share_batch(D, W, k=10)
    s = [j_share(D, W[:, i], k=10)["j_share"] for i in range(6)]
    assert np.allclose(b, s, atol=1e-8)


def test_random_baseline_and_classify(lens):
    b1 = random_baseline(lens, 0, n=64, k=10, seed=3, cache=False)
    b2 = random_baseline(lens, 0, n=64, k=10, seed=3, cache=False)
    assert np.array_equal(b1, b2) and len(b1) == 64
    c = classify(float(np.max(b1)) + 1, b1)
    assert c["class"] == "high" and c["percentile_vs_random"] == 100.0
    assert classify(float(np.min(b1)) - 1, b1)["class"] == "low"


def test_context_derivative_matches_finite_difference(tiny, ids, lens):
    tiny.tokenizer = tiny.tokenizer or _Tok()
    g = torch.Generator().manual_seed(2)
    v = torch.randn(tiny.adapter.d_model, generator=g)
    r = validate(tiny, lens, ids, 1, v, target=9, eps_rel=[-1e-3, 1e-3])
    lo, hi = r["rows"]
    fd = (hi["observed"] - lo["observed"]) / (hi["eps"] - lo["eps"])
    assert r["dz_deps_context"] == pytest.approx(fd, rel=1e-2, abs=1e-5)


def test_last_layer_lens_equals_context_derivative(tiny, ids, lens):
    """At the last layer the lens is the identity and the logit is read at the injection position,
    so the corpus prediction and the exact derivative must coincide."""
    tiny.tokenizer = tiny.tokenizer or _Tok()
    v = tiny.adapter.output_vector(tiny.adapter.n_layers - 1, 2)
    r = validate(tiny, lens, ids, tiny.adapter.n_layers - 1, v, target=4, eps_rel=[1e-3])
    assert r["dz_deps_corpus_jlens"] == pytest.approx(r["dz_deps_context"], rel=1e-4, abs=1e-6)
