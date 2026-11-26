"""Checks on the real Qwen2.5-0.5B checkpoint (skipped when not cached)."""

import pytest
import torch
from fastapi.testclient import TestClient

from instrumentation.activation_cache import CaptureSpec, run_forward, run_plain
from instrumentation.hooks import count_hooks
from instrumentation.neuron_inspector import reconstruction

pytestmark = pytest.mark.real_model


def test_real_hooked_matches_plain_and_reconstructs(real):
    ids = real.tokenizer("The fourth Gospel was written by", return_tensors="pt").input_ids[0]
    plain = run_plain(real, ids)
    res = run_forward(real, ids, CaptureSpec())
    assert torch.equal(plain, res.logits)
    for L in range(real.adapter.n_layers):
        r = reconstruction(real.adapter, L, res.get("act", L), res.get("mlp_out", L))
        assert r["max_rel_error"] < 1e-4, (L, r)
    assert count_hooks(real.model) == 0


def test_tokens_leading_whitespace(real):
    tok = real.tokenizer
    for s in [" John", "John", " the", "\n", " Ripper"]:
        ids = tok.encode(s, add_special_tokens=False)
        assert tok.decode(ids) == s
    j = tok.encode(" John", add_special_tokens=False)
    assert len(j) == 1 and tok.decode(j) == " John" and tok.decode(tok.encode("John", add_special_tokens=False)) == "John"
    assert j != tok.encode("John", add_special_tokens=False)


def test_api_end_to_end(real):
    from main import app
    from models.loader import manager
    manager.set(real)
    c = TestClient(app)
    p = "The password cracking tool is called John the"
    r = c.post("/inference", json={"prompt": p, "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["layers"]) == real.adapter.n_layers and len(body["top_k"]) == 5
    assert body["provenance"]["input_token_ids"] == real.tokenizer(p).input_ids
    r = c.post("/neurons/inspect", json={"prompt": p, "layer": 10, "neuron": 100, "position": -1})
    assert r.status_code == 200, r.text
    assert r.json()["reconstruction"]["max_rel_error"] < 1e-4
    r = c.post("/interventions/run", json={"prompt": p, "target_tokens": [" John"],
                                           "interventions": [{"layer": 10, "neurons": [100], "op": "zero"}]})
    assert r.status_code == 200, r.text
    assert r.json()["hooks_remaining"] == 0
    r = c.post("/interventions/run", json={"prompt": p, "target_tokens": [" Rippers and"],
                                           "interventions": [{"layer": 10, "neurons": [100], "op": "zero"}]})
    assert r.status_code == 422
    r = c.post("/neurons/inspect", json={"prompt": p, "layer": 10, "neuron": 10**7})
    assert r.status_code == 422
    r = c.post("/activations/capture", json={"prompt": p, "layer": 3, "order": "max_abs", "neuron_count": 16})
    assert r.status_code == 200 and len(r.json()["neurons"]) == 16
    assert count_hooks(real.model) == 0


def test_api_concurrent_requests(real):
    """Concurrent requests must not see each other's hooks (regression: leak check raced)."""
    from concurrent.futures import ThreadPoolExecutor
    from main import app
    from models.loader import manager
    manager.set(real)
    c = TestClient(app)
    p = "The fourth Gospel was written by"
    reqs = [("/neurons/inspect", {"prompt": p, "layer": 5, "neuron": 7}),
            ("/activations/capture", {"prompt": p, "layer": 5, "neuron_count": 256}),
            ("/interventions/run", {"prompt": p, "interventions": [{"layer": 5, "neurons": [7], "op": "zero"}]}),
            ("/inference", {"prompt": p})] * 4
    with ThreadPoolExecutor(8) as ex:
        codes = list(ex.map(lambda r: c.post(r[0], json=r[1]).status_code, reqs))
    assert codes == [200] * len(reqs), codes
    assert count_hooks(real.model) == 0
