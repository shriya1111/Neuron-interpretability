"""Phase 2: logit lens, vector-add interventions, random controls, Type 1 experiment."""

import pytest
import torch

from evaluation.controls import random_directions
from instrumentation.activation_cache import CaptureSpec, run_forward, run_plain
from instrumentation.hooks import count_hooks
from instrumentation.interventions import InterventionPlan, NeuronIntervention, VectorAdd
from interpretability import logit_lens


class _Tok:
    def decode(self, x): return f"<{x[0]}>"
    def convert_ids_to_tokens(self, i): return f"t{i}"


def test_folded_scores_match_manual(tiny):
    ad = tiny.adapter
    w = ad.output_vector(1, 4).detach()
    s = logit_lens.direction_scores(ad, w)
    manual = (ad.unembed.weight.double() @ (ad.final_norm.weight.double() * w.double())).float()
    assert torch.allclose(s, manual, atol=1e-4)


def test_minus_readout_is_negation(tiny):
    tiny.tokenizer = tiny.tokenizer or _Tok()
    r = logit_lens.neuron_readout(tiny.tokenizer, tiny.adapter, 2, 3, k=5, targets=[7])
    assert r["plus"]["targets"][0]["score"] == pytest.approx(-r["minus"]["targets"][0]["score"], abs=1e-6)
    top_plus = [x["id"] for x in r["plus"]["top_by_score"]]
    bottom_minus = [x["id"] for x in r["minus"]["bottom_by_score"]]
    assert top_plus == bottom_minus


def test_residual_lens_last_layer_is_model_prediction(tiny, ids):
    tiny.tokenizer = tiny.tokenizer or _Tok()
    r = logit_lens.residual_lens(tiny, ids, -1, k=5)
    assert r["last_layer_matches_model_max_abs_logit_error"] < 1e-4
    assert [x["id"] for x in r["layers"][-1]["top_k"]] == [x["id"] for x in r["model_prediction"]]
    assert len(r["layers"]) == tiny.adapter.n_layers + 1  # embedding + every block
    assert count_hooks(tiny.model) == 0


def test_dla_equals_direct_effect(tiny, ids):
    """a·(u_t·w)/rms equals the change in target logit when a·w is removed only from the final state
    while holding the final rms fixed (the definition of a direct, unembedding-only effect)."""
    ad = tiny.adapter
    L, n, tid = ad.n_layers - 1, 5, 9
    res = run_forward(tiny, ids, CaptureSpec(layers=[L], points=("act", "resid_post")))
    h = res.get("resid_post", L)[-1].double()
    a = float(res.get("act", L)[-1, n])
    w = ad.output_vector(L, n).double()
    eps = ad.final_norm.variance_epsilon
    rms = torch.sqrt(h.pow(2).mean() + eps)
    u = logit_lens.token_direction(ad, tid).double()
    dla = a * float(u @ w) / float(rms)
    W = ad.unembed.weight.double()
    g = ad.final_norm.weight.double()
    full = float(W[tid] @ (g * h / rms))
    without = float(W[tid] @ (g * (h - a * w) / rms))
    assert dla == pytest.approx(full - without, rel=1e-6, abs=1e-9)


def test_zero_ablation_equals_subtracting_aw(tiny, ids):
    """Two independent mechanisms: activation edit vs. adding −a_t·w to the MLP output."""
    L, n = 1, 6
    a = run_forward(tiny, ids, CaptureSpec(layers=[L], points=("act",))).get("act", L)[:, n]
    w = tiny.adapter.output_vector(L, n).detach().float()
    z = run_forward(tiny, ids, None, InterventionPlan([NeuronIntervention(L, [n], "zero")])).logits
    v = run_forward(tiny, ids, None, InterventionPlan(vector_adds=[VectorAdd(L, w, "mlp_out", None, (-a).tolist())])).logits
    assert torch.allclose(z, v, atol=1e-5)
    assert count_hooks(tiny.model) == 0


def test_vector_add_resid_post_and_positions(tiny, ids):
    L = 0
    v = torch.zeros(tiny.adapter.d_model)
    base = run_forward(tiny, ids, CaptureSpec(layers=[L], points=("resid_post",)))
    r0 = run_forward(tiny, ids, CaptureSpec(layers=[L], points=("resid_post",)),
                     InterventionPlan(vector_adds=[VectorAdd(L, v, "resid_post")]))
    assert torch.equal(base.logits, r0.logits)  # adding zero changes nothing
    v[0] = 3.0
    r1 = run_forward(tiny, ids, CaptureSpec(layers=[L], points=("resid_post",)),
                     InterventionPlan(vector_adds=[VectorAdd(L, v, "resid_post", [2, 5], 2.0)]))
    d = r1.get("resid_post", L) - base.get("resid_post", L)
    assert torch.allclose(d[2], 2 * v) and torch.allclose(d[5], 2 * v)
    assert torch.all(d[[0, 1, 3, 4]] == 0)
    with pytest.raises(ValueError):
        run_forward(tiny, ids, None, InterventionPlan(vector_adds=[VectorAdd(L, v[:5], "mlp_out")]))
    assert count_hooks(tiny.model) == 0


def test_random_directions_matched_norm():
    r = random_directions(50, 64, 2.5, seed=3)
    assert torch.allclose(r.norm(dim=-1), torch.full((50,), 2.5), atol=1e-5)
    assert torch.equal(r, random_directions(50, 64, 2.5, seed=3))
    assert not torch.equal(r, random_directions(50, 64, 2.5, seed=4))


def test_neuron_search_matches_bruteforce(tiny):
    tiny.tokenizer = tiny.tokenizer or _Tok()
    ad, tid = tiny.adapter, 11
    out = logit_lens.neuron_search(tiny.tokenizer, ad, tid, "-", top=3)
    u = logit_lens.token_direction(ad, tid)
    best = max(((float(-(u @ ad.output_vector(L, i).float())) / (float(ad.output_vector(L, i).float().norm()) * float(u.norm())), L, i)
                for L in range(ad.n_layers) for i in range(ad.d_mlp(L))))
    assert (out["candidates"][0]["layer"], out["candidates"][0]["neuron"]) == (best[1], best[2])
    assert out["candidates"][0]["cosine"] == pytest.approx(best[0], abs=1e-5)
