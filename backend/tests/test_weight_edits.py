"""Weight editing: exactness, reversibility, equivalence with activation interventions."""

import pytest
import torch

from instrumentation.activation_cache import run_forward, run_plain
from instrumentation.interventions import InterventionPlan, NeuronIntervention
from instrumentation.weight_edits import get_editor


def _state(m):
    return {k: v.clone() for k, v in m.state_dict().items()}


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture
def ed(tiny):
    return get_editor(tiny)


def test_prune_equals_activation_zero_and_reverts_bitwise(tiny, ids, ed):
    before = _state(tiny.model)
    ablated = run_forward(tiny, ids, None, InterventionPlan([NeuronIntervention(1, [3, 7], "zero")])).logits
    e = ed.apply("prune", 1, [3, 7])
    assert torch.allclose(run_plain(tiny, ids), ablated, atol=1e-5)
    mlp = tiny.adapter.mlp(1)
    assert torch.all(mlp.down_proj.weight[:, [3, 7]] == 0) and torch.all(mlp.gate_proj.weight[[3, 7]] == 0)
    assert e.checksum_before != e.checksum_after
    ed.revert(e.id)
    assert _same(before, _state(tiny.model)) and not ed.modified


def test_scale_output_equals_activation_scale(tiny, ids, ed):
    ref = run_forward(tiny, ids, None, InterventionPlan([NeuronIntervention(2, [5], "scale", 3.0)])).logits
    ed.apply("scale_output", 2, [5], {"factor": 3.0})
    assert torch.allclose(run_plain(tiny, ids), ref, atol=1e-4)
    ed.revert_all()


def test_scale_up_row_equals_activation_scale(tiny, ids, ed):
    """a = silu(g·x)(u·x) is linear in u, so scaling the up row by s scales a by s."""
    ref = run_forward(tiny, ids, None, InterventionPlan([NeuronIntervention(0, [9], "scale", -2.0)])).logits
    ed.apply("scale_input", 0, [9], {"factor": -2.0, "which": ["up"]})
    assert torch.allclose(run_plain(tiny, ids), ref, atol=1e-4)
    ed.revert_all()


def test_set_and_add_direction(tiny, ed):
    d = tiny.adapter.d_model
    v = torch.arange(d, dtype=torch.float32) / d
    w0 = tiny.adapter.output_vector(1, 2).detach().clone()
    e1 = ed.apply("set_output", 1, [2], {"vector": v.tolist()})
    assert torch.allclose(tiny.adapter.output_vector(1, 2), v)
    ed.revert(e1.id)
    assert torch.equal(tiny.adapter.output_vector(1, 2), w0)
    src = tiny.adapter.output_vector(0, 4).detach().float()
    ed.apply("add_direction", 1, [2], {"alpha": 0.5, "direction": {"source": "neuron", "layer": 0, "neuron": 4}})
    assert torch.allclose(tiny.adapter.output_vector(1, 2), w0 + 0.5 * src / src.norm(), atol=1e-6)
    ed.revert_all()
    assert torch.equal(tiny.adapter.output_vector(1, 2), w0)


def test_original_weights_context(tiny, ids, ed):
    base = run_plain(tiny, ids)
    ed.apply("prune", 2, list(range(20)))
    edited = run_plain(tiny, ids)
    assert not torch.allclose(base, edited)
    with ed.original_weights():
        assert torch.equal(run_plain(tiny, ids), base)
    assert torch.equal(run_plain(tiny, ids), edited)
    ed.revert_all()
    assert torch.equal(run_plain(tiny, ids), base)


def test_overlapping_revert_order_and_atomicity(tiny, ed):
    before = _state(tiny.model)
    a = ed.apply("scale_output", 1, [1], {"factor": 2.0})
    b = ed.apply("prune", 1, [1])
    with pytest.raises(ValueError):
        ed.revert(a.id)  # b changed the same weights later
    ed.revert(b.id)
    ed.revert(a.id)
    assert _same(before, _state(tiny.model))
    with pytest.raises(ValueError):
        ed.apply("set_output", 1, [1], {"vector": [1.0, 2.0]})  # wrong length
    assert _same(before, _state(tiny.model)) and not ed.modified
    with pytest.raises(ValueError):
        ed.apply("scale_output", 1, [10**6], {"factor": 2.0})
    with pytest.raises(ValueError):
        ed.apply("scale_output", 1, [1], {})


def test_multiple_overlapping_edits_original_weights(tiny, ids, ed):
    base = run_plain(tiny, ids)
    ed.apply("scale_output", 0, [1, 2], {"factor": 0.5})
    ed.apply("prune", 0, [2, 3])
    ed.apply("scale_input", 0, [1], {"factor": 4.0})
    with ed.original_weights():
        assert torch.equal(run_plain(tiny, ids), base)
    ed.revert_all()
    assert torch.equal(run_plain(tiny, ids), base)
