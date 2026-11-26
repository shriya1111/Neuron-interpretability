"""Mathematical correctness tests for Phase 1 instrumentation."""

import copy

import pytest
import torch

from instrumentation.activation_cache import CaptureSpec, run_forward, run_plain
from instrumentation.compare import compare_runs
from instrumentation.hooks import count_hooks
from instrumentation.interventions import InterventionPlan, NeuronIntervention
from instrumentation.neuron_inspector import inspect_neuron, reconstruction
from models.registry import CheckpointMismatchError, paper_neuron


def _iv(**kw):
    return InterventionPlan([NeuronIntervention(**kw)])


# 1. hooked == uninstrumented
def test_hooked_matches_plain(tiny, ids):
    plain = run_plain(tiny, ids)
    hooked = run_forward(tiny, ids, CaptureSpec())
    assert torch.equal(plain, hooked.logits), "capture hooks changed the forward pass"
    assert count_hooks(tiny.model) == 0


# 2. sum of neuron contributions == MLP output
def test_reconstruction(tiny, ids):
    res = run_forward(tiny, ids, CaptureSpec())
    for L in range(tiny.adapter.n_layers):
        r = reconstruction(tiny.adapter, L, res.get("act", L), res.get("mlp_out", L))
        assert r["max_rel_error"] < 1e-5, (L, r["max_rel_error"])


def test_reconstruction_with_bias(tiny_llama_bias, ids):
    h = tiny_llama_bias
    assert h.adapter.has_mlp_bias(0)
    res = run_forward(h, ids, CaptureSpec())
    for L in range(h.adapter.n_layers):
        r = reconstruction(h.adapter, L, res.get("act", L), res.get("mlp_out", L))
        assert r["includes_bias"] and r["max_rel_error"] < 1e-5
    # Without the bias term the reconstruction must be wrong — the check is not vacuous.
    W = h.adapter.mlp(0).down_proj.weight.double()
    no_bias = res.get("act", 0).double() @ W.T
    assert (no_bias - res.get("mlp_out", 0).double()).abs().max() > 1e-3


# 3. inspector recomputation matches hooks; residual stream != MLP input
def test_inspector_recompute(tiny, ids):
    for L in range(tiny.adapter.n_layers):
        out = inspect_neuron(tiny, ids, L, 5, -1)
        for k, v in out["recompute_error"].items():
            assert v < 1e-4, (L, k, v)
        assert out["mlp_input_vs_raw_residual"]["max_abs_error_recomputed_norm"] < 1e-4
        s = out["steps"]
        assert abs(s["silu"] - s["gate"] * s["sigmoid_gate"]) < 1e-9
        assert abs(s["activation"] - s["silu"] * s["up"]) < 1e-9
        c = torch.tensor(out["vectors"]["contribution"], dtype=torch.float64)
        w = torch.tensor(out["vectors"]["w"], dtype=torch.float64)
        assert torch.allclose(c, s["activation"] * w, atol=1e-6)


def test_output_vector_is_down_proj_column(tiny):
    L, n = 1, 7
    w = tiny.adapter.output_vector(L, n)
    assert torch.equal(w, tiny.model.model.layers[L].mlp.down_proj.weight[:, n])
    assert w.shape == (tiny.adapter.d_model,)


# 4/5. intervention deltas are exactly (a' - a) w at the chosen positions only
@pytest.mark.parametrize("op,value,factor", [("zero", 0.0, 0.0), ("scale", 2.5, 2.5), ("negate", 0.0, -1.0),
                                             ("scale", 0.0, 0.0)])
def test_intervention_delta(tiny, ids, op, value, factor):
    L, n, pos = 1, 3, [4, 9]
    spec = CaptureSpec(layers=[L], points=("act", "mlp_out"))
    base = run_forward(tiny, ids, spec)
    mod = run_forward(tiny, ids, spec, _iv(layer=L, neurons=[n], op=op, value=value, positions=pos))
    a = base.get("act", L)[:, n].double()
    w = tiny.adapter.output_vector(L, n).double()
    delta = mod.get("mlp_out", L).double() - base.get("mlp_out", L).double()
    for t in range(ids.numel()):
        expected = (factor - 1.0) * a[t] * w if t in pos else torch.zeros_like(w)
        assert torch.allclose(delta[t], expected, atol=1e-5), (t, (delta[t] - expected).abs().max())
    assert count_hooks(tiny.model) == 0


def test_set_override(tiny, ids):
    L, n, v = 2, 11, 3.0
    spec = CaptureSpec(layers=[L], points=("act", "mlp_out"))
    base = run_forward(tiny, ids, spec)
    mod = run_forward(tiny, ids, spec, _iv(layer=L, neurons=[n], op="set", value=v, positions=[-1]))
    a = base.get("act", L)[-1, n].double()
    w = tiny.adapter.output_vector(L, n).double()
    delta = mod.get("mlp_out", L)[-1].double() - base.get("mlp_out", L)[-1].double()
    assert torch.allclose(delta, (v - a) * w, atol=1e-5)
    assert mod.get("act_mod", L)[-1, n] == pytest.approx(v)


def test_scaling_is_proportional(tiny, ids):
    L, n = 0, 2
    spec = CaptureSpec(layers=[L], points=("act", "mlp_out"))
    base = run_forward(tiny, ids, spec).get("mlp_out", L).double()
    deltas = []
    for s in (0.5, 2.0, 3.0):
        m = run_forward(tiny, ids, spec, _iv(layer=L, neurons=[n], op="scale", value=s)).get("mlp_out", L).double()
        deltas.append((m - base) / (s - 1.0))
    assert torch.allclose(deltas[0], deltas[1], atol=1e-5) and torch.allclose(deltas[1], deltas[2], atol=1e-5)


# 6. hook-based zero ablation == independent weight edit on a deep copy
def test_ablation_matches_weight_edit(tiny, ids):
    L, n = 1, 6
    before = {k: v.clone() for k, v in tiny.model.state_dict().items()}
    hooked = run_forward(tiny, ids, None, _iv(layer=L, neurons=[n], op="zero")).logits
    edited = copy.deepcopy(tiny.model)
    with torch.no_grad():
        edited.model.layers[L].mlp.down_proj.weight[:, n] = 0.0
        ref = edited(input_ids=ids.unsqueeze(0)).logits[0]
    assert torch.allclose(hooked, ref, atol=1e-5)
    assert not torch.allclose(hooked, run_plain(tiny, ids)), "ablation had no effect"
    for k, v in tiny.model.state_dict().items():
        assert torch.equal(v, before[k]), f"weights changed: {k}"


def test_compare_runs_layer_checks(tiny, ids):
    plan = InterventionPlan([NeuronIntervention(layer=1, neurons=[1, 2], op="zero", positions=[3]),
                             NeuronIntervention(layer=2, neurons=[4], op="scale", value=-2.0)])
    if tiny.tokenizer is None:
        class _T:
            def decode(self, x): return f"<{x[0]}>"
            def convert_ids_to_tokens(self, i): return f"t{i}"
        tiny.tokenizer = _T()
    out = compare_runs(tiny, ids, plan)
    c1, c2 = out["layer_checks"]
    assert c1["pre_edit_activation_matches_baseline"] and c1["positions_changed"] == [3]
    assert c2["upstream_intervened"]
    for c in (c1, c2):
        assert c["max_abs_error_observed_vs_expected_delta"] < 1e-5
    assert all(r["delta_norm"] == 0 for r in out["residual_delta_by_layer"][:1])  # layer 0 untouched
    assert count_hooks(tiny.model) == 0


# 7. hooks removed even when the forward pass raises
def test_hooks_removed_on_error(tiny, ids):
    with pytest.raises(ValueError):
        run_forward(tiny, ids, CaptureSpec(), _iv(layer=0, neurons=[10**6], op="zero"))
    assert count_hooks(tiny.model) == 0

    orig = tiny.model.model.layers[1].forward

    def boom(*a, **k):
        raise RuntimeError("boom")
    tiny.model.model.layers[1].forward = boom
    try:
        with pytest.raises(RuntimeError):
            run_forward(tiny, ids, CaptureSpec(), _iv(layer=2, neurons=[0], op="zero"))
    finally:
        tiny.model.model.layers[1].forward = orig
    assert count_hooks(tiny.model) == 0


# 8. determinism
def test_repeat_baseline_identical(tiny, ids):
    a = run_forward(tiny, ids, CaptureSpec()).logits
    b = run_forward(tiny, ids, CaptureSpec()).logits
    assert torch.equal(a, b)


# 10. validation
def test_validation(tiny, ids):
    ad = tiny.adapter
    with pytest.raises(ValueError):
        ad.check_neuron(ad.n_layers, 0)
    with pytest.raises(ValueError):
        ad.check_neuron(0, ad.d_mlp(0))
    with pytest.raises(ValueError):
        run_forward(tiny, ids, None, _iv(layer=0, neurons=[0], op="zero", positions=[ids.numel()]))
    with pytest.raises(ValueError):
        inspect_neuron(tiny, ids, 0, 0, 99)


def test_paper_neuron_checkpoint_guard():
    pn = paper_neuron("john", "meta-llama/Llama-3.1-8B-Instruct")
    assert (pn.layer, pn.neuron, pn.sign) == (28, 2760, "-")
    with pytest.raises(CheckpointMismatchError):
        paper_neuron("john", "Qwen/Qwen2.5-0.5B")
    with pytest.raises(CheckpointMismatchError):
        paper_neuron("john", "meta-llama/Llama-3.1-8B")  # base model is a different checkpoint


def test_dimensions_read_from_model(tiny):
    ad = tiny.adapter
    assert ad.n_layers == 3 and ad.d_model == 64 and ad.d_mlp(0) == 96 and ad.vocab_size == 512


# Transformers' own output-capturing hooks (installed on first output_attentions) must not alter outputs.
def test_attention_capture_and_foreign_hooks(tiny, ids):
    from instrumentation.activation_cache import run_attention
    from instrumentation.hooks import foreign_hooks
    before = run_plain(tiny, ids)
    att = run_attention(tiny, ids, 1)
    assert att.shape == (tiny.adapter.n_heads, ids.numel(), ids.numel())
    assert torch.allclose(att.sum(-1), torch.ones(att.shape[:2]), atol=1e-5)
    assert torch.all(att.triu(1) == 0), "causal mask violated"
    assert count_hooks(tiny.model) == 0
    assert all("transformers" in f for f in foreign_hooks(tiny.model))
    assert torch.equal(before, run_plain(tiny, ids))
    assert torch.equal(before, run_forward(tiny, ids, CaptureSpec()).logits)
