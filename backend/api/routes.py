from __future__ import annotations

import time

import torch
from fastapi import APIRouter, HTTPException

from evaluation import metrics
from instrumentation.activation_cache import CaptureSpec, run_attention, run_forward
from instrumentation.compare import compare_runs
from instrumentation.hooks import count_hooks, foreign_hooks
from instrumentation.interventions import InterventionPlan, NeuronIntervention
from instrumentation.neuron_inspector import inspect_neuron
from models.architecture import describe
from models.loader import manager, set_deterministic
from models.registry import MODEL_PRESETS, list_paper_neurons
from reproducibility import provenance

from . import schemas

router = APIRouter()


def _handle():
    try:
        return manager.get()
    except RuntimeError as e:
        raise HTTPException(409, str(e))


def _encode(h, req: schemas.PromptInput) -> torch.Tensor:
    if req.token_ids:
        V = h.adapter.vocab_size
        if any(not 0 <= t < V for t in req.token_ids):
            raise HTTPException(422, f"token ids must be in [0, {V - 1}]")
        return torch.tensor(req.token_ids, dtype=torch.long)
    if not req.prompt:
        raise HTTPException(422, "give a prompt or token_ids")
    ids = h.tokenizer(req.prompt, add_special_tokens=req.add_special_tokens, return_tensors="pt").input_ids[0]
    if ids.numel() == 0:
        raise HTTPException(422, "prompt encodes to zero tokens")
    return ids


def _hooks_clean(h) -> int:
    # Hooks are only ever registered while holding h.lock, so counting under the lock
    # cannot observe another request's in-flight hooks.
    with h.lock:
        n = count_hooks(h.model)
    if n:
        raise HTTPException(500, f"{n} hooks still registered after operation — refusing to return results")
    return n


def _wrap(fn):
    try:
        return fn()
    except HTTPException:
        raise
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e))


@router.get("/health")
def health():
    out = {"ok": True, "model_loaded": manager.loaded, "cuda": torch.cuda.is_available()}
    if manager.loaded:
        m = manager.get().model
        out["neuronlab_hooks_attached"] = count_hooks(m)
        out["foreign_hooks"] = foreign_hooks(m)
    return out


@router.get("/models/presets")
def presets():
    return {"presets": MODEL_PRESETS, "paper_neurons": list_paper_neurons()}


@router.post("/models/load")
def load(req: schemas.LoadRequest):
    try:
        h = manager.load(req.model_id, device=req.device, dtype=req.dtype, revision=req.revision, seed=req.seed)
    except Exception as e:  # surface HF auth / unsupported-arch errors verbatim
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return {"loaded": True, "load_seconds": h.load_seconds, "provenance": provenance(h),
            "architecture": describe(h.adapter)}


@router.get("/models/config")
def config():
    h = _handle()
    return {"provenance": provenance(h), "architecture": describe(h.adapter),
            "hf_config": h.model.config.to_dict()}


@router.post("/inference")
def inference(req: schemas.InferenceRequest):
    h = _handle()

    def go():
        ids = _encode(h, req)
        spec = CaptureSpec(points=("resid_pre", "attn_out", "gate_pre", "act", "mlp_out", "resid_post")) \
            if req.layer_summary else None
        res = run_forward(h, ids, spec)
        T = ids.numel()
        t = req.position + T if req.position < 0 else req.position
        if not 0 <= t < T:
            raise ValueError(f"position {req.position} out of range for {T} tokens")
        out = {
            "tokens": metrics.tokens_of(h.tokenizer, ids),
            "position": t,
            "top_k": metrics.top_k(h.tokenizer, res.logits[t], req.top_k),
            "forward_seconds": res.seconds,
        }
        if req.layer_summary:
            layers = []
            for L in range(h.adapter.n_layers):
                act = res.get("act", L)
                gate = res.get("gate_pre", L)
                top = act[t].abs().topk(min(req.top_neurons_per_layer, act.shape[1])) if req.top_neurons_per_layer else None
                layers.append({
                    "layer": L,
                    "resid_pre_norm": res.get("resid_pre", L).norm(dim=-1).tolist(),
                    "attn_out_norm": res.get("attn_out", L).norm(dim=-1).tolist(),
                    "mlp_out_norm": res.get("mlp_out", L).norm(dim=-1).tolist(),
                    "resid_post_norm": res.get("resid_post", L).norm(dim=-1).tolist(),
                    "gate_open_frac": (gate > 0).float().mean(-1).tolist(),
                    "act_abs_max": act.abs().max(-1).values.tolist(),
                    "top_neurons_at_position": [] if top is None else
                        [{"neuron": int(i), "activation": float(act[t, i])} for i in top.indices],
                })
            out["layers"] = layers
        if req.attention_layer is not None:
            att = run_attention(h, ids, req.attention_layer)
            out["attention"] = {"layer": req.attention_layer, "shape": list(att.shape),
                                "note": "softmax attention probabilities from an eager forward pass",
                                "pattern": att.tolist()}
        _hooks_clean(h)
        out["provenance"] = provenance(h, prompt=req.prompt, input_ids=ids, params=req.model_dump(exclude={"prompt"}))
        return out

    return _wrap(go)


@router.post("/activations/capture")
def capture(req: schemas.CaptureRequest):
    h = _handle()

    def go():
        h.adapter.check_layer(req.layer)
        ids = _encode(h, req)
        res = run_forward(h, ids, CaptureSpec(layers=[req.layer], points=("act", "gate_pre", "up"), final=False))
        m = res.get(req.point, req.layer)  # [T, d_mlp]
        d = m.shape[1]
        col_max = m.abs().max(0).values
        if req.order == "max_abs":
            idx = col_max.topk(min(req.neuron_count, d)).indices
        else:
            if not 0 <= req.neuron_start < d:
                raise ValueError(f"neuron_start must be in [0, {d - 1}]")
            idx = torch.arange(req.neuron_start, min(req.neuron_start + req.neuron_count, d))
        _hooks_clean(h)
        return {
            "layer": req.layer, "point": req.point, "shape_full": list(m.shape),
            "tokens": metrics.tokens_of(h.tokenizer, ids),
            "neurons": idx.tolist(),
            "values": m[:, idx].tolist(),  # [T, len(neurons)]
            "gate_open": (res.get("gate_pre", req.layer)[:, idx] > 0).tolist(),
            "layer_col_max_abs": col_max.tolist(),  # [d_mlp] overview strip
            "provenance": provenance(h, prompt=req.prompt, input_ids=ids, params=req.model_dump(exclude={"prompt"})),
        }

    return _wrap(go)


@router.post("/neurons/inspect")
def inspect(req: schemas.InspectRequest):
    h = _handle()

    def go():
        ids = _encode(h, req)
        out = inspect_neuron(h, ids, req.layer, req.neuron, req.position, req.top_n, req.include_weights)
        out["tokens"] = metrics.tokens_of(h.tokenizer, ids)
        _hooks_clean(h)
        out["provenance"] = provenance(h, prompt=req.prompt, input_ids=ids, params=req.model_dump(exclude={"prompt"}))
        return out

    return _wrap(go)


def _single_token(h, s: str) -> int:
    ids = h.tokenizer.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        pieces = [h.tokenizer.decode([i]) for i in ids]
        raise ValueError(f"target {s!r} is {len(ids)} tokens {pieces}; give a single-token string "
                         f"(mind the leading space)")
    return ids[0]


@router.post("/interventions/run")
def interventions(req: schemas.InterventionRequest):
    h = _handle()

    def go():
        set_deterministic(req.seed)
        ids = _encode(h, req)
        plan = InterventionPlan([NeuronIntervention(**iv.model_dump()) for iv in req.interventions])
        targets = [_single_token(h, s) for s in req.target_tokens]
        t0 = time.time()
        out = compare_runs(h, ids, plan, req.position, req.top_k, targets)
        out["runtime_seconds"] = round(time.time() - t0, 4)
        out["hooks_remaining"] = _hooks_clean(h)
        out["provenance"] = provenance(h, prompt=req.prompt, input_ids=ids, seed=req.seed,
                                       params=req.model_dump(exclude={"prompt"}))
        return out

    return _wrap(go)


# ---------------------------------------------------------------- Phase 2: logit lens

from experiments.polysemanticity import type1, type1_candidates  # noqa: E402
from interpretability import logit_lens  # noqa: E402


@router.post("/lens/logit")
def lens_logit(req: schemas.LogitLensRequest):
    h = _handle()

    def go():
        targets = [_single_token(h, s) for s in req.target_tokens]
        if req.kind == "neuron":
            if req.layer is None or req.neuron is None:
                raise ValueError("kind=neuron needs layer and neuron")
            out = logit_lens.neuron_readout(h.tokenizer, h.adapter, req.layer, req.neuron, req.top_k, targets)
            ids = None
        elif req.kind == "vector":
            if not req.vector or len(req.vector) != h.adapter.d_model:
                raise ValueError(f"kind=vector needs a vector of length d_model={h.adapter.d_model}")
            out = logit_lens.direction_readout(h.tokenizer, h.adapter, torch.tensor(req.vector), req.top_k, targets)
            ids = None
        else:
            if not req.prompt:
                raise ValueError("kind=residual needs a prompt")
            ids = h.tokenizer(req.prompt, return_tensors="pt").input_ids[0]
            out = logit_lens.residual_lens(h, ids, req.position, min(req.top_k, 50), targets)
        _hooks_clean(h)
        out["provenance"] = provenance(h, prompt=req.prompt, input_ids=ids, params=req.model_dump(exclude={"vector"}))
        return out

    return _wrap(go)


@router.post("/lens/neuron-search")
def lens_neuron_search(req: schemas.NeuronSearchRequest):
    h = _handle()

    def go():
        out = logit_lens.neuron_search(h.tokenizer, h.adapter, _single_token(h, req.target), req.sign, req.top, req.layers)
        out["provenance"] = provenance(h, params=req.model_dump())
        return out

    return _wrap(go)


@router.post("/experiments/polysemanticity")
def exp_type1(req: schemas.Type1Request):
    h = _handle()

    def go():
        set_deterministic(req.seed)
        out = type1(h, req.layer, req.neuron, req.sign, [c.model_dump() for c in req.contexts], req.n_controls, req.seed)
        out["hooks_remaining"] = _hooks_clean(h)
        out["provenance"] = provenance(h, seed=req.seed, params=req.model_dump())
        return out

    return _wrap(go)


@router.post("/experiments/polysemanticity/candidates")
def exp_type1_candidates(req: schemas.Type1CandidatesRequest):
    h = _handle()

    def go():
        out = type1_candidates(h, [c.model_dump() for c in req.contexts], req.sign, req.top)
        _hooks_clean(h)
        out["provenance"] = provenance(h, params=req.model_dump())
        return out

    return _wrap(go)


# ---------------------------------------------------------------- weight editing (persistent until reverted)

from evaluation.weight_compare import compare as weights_compare  # noqa: E402
from instrumentation.weight_edits import get_editor  # noqa: E402


@router.get("/weights/edits")
def weights_edits():
    h = _handle()
    return {"modified": get_editor(h).modified, "edits": get_editor(h).summary()}


@router.post("/weights/edit")
def weights_edit(req: schemas.WeightEditRequest):
    h = _handle()
    return _wrap(lambda: {"applied": get_editor(h).apply(req.kind, req.layer, req.neurons, req.params).summary(),
                          "edits": get_editor(h).summary()})


@router.post("/weights/revert")
def weights_revert(req: schemas.WeightRevertRequest):
    h = _handle()

    def go():
        ed = get_editor(h)
        if req.all:
            return {"reverted": ed.revert_all(), "edits": ed.summary()}
        if req.edit_id is None:
            raise ValueError("give edit_id or all=true")
        return {"reverted": ed.revert(req.edit_id).summary(), "edits": ed.summary()}

    return _wrap(go)


@router.post("/weights/compare")
def weights_compare_route(req: schemas.WeightCompareRequest):
    h = _handle()

    def go():
        out = weights_compare(h, [p.model_dump() for p in req.prompts], req.top_k, req.n_heldout)
        _hooks_clean(h)
        out["provenance"] = provenance(h, params=req.model_dump())
        return out

    return _wrap(go)


@router.post("/weights/export")
def weights_export(req: schemas.WeightExportRequest):
    import json
    import os
    h = _handle()

    def go():
        ed = get_editor(h)
        if not ed.modified:
            raise ValueError("no active edits; nothing to export")
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "edited_models", req.name))
        if os.path.exists(root):
            raise ValueError(f"{root} already exists")
        with h.lock:
            h.model.save_pretrained(root)
            h.tokenizer.save_pretrained(root)
        with open(os.path.join(root, "neuronlab_edits.json"), "w") as f:
            json.dump({"base_model": h.model_id, "base_revision": h.revision, "edits": ed.summary()}, f, indent=2)
        return {"path": root, "edits": ed.summary()}

    return _wrap(go)


# ---------------------------------------------------------------- atlas (precomputed corpus scan)

from interpretability import atlas as atlas_mod  # noqa: E402


def _atlas(tag: str | None):
    h = _handle()
    try:
        st = atlas_mod.get_store(tag)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    if st.meta.get("model_id") != h.model_id:
        raise HTTPException(409, f"atlas was computed for {st.meta.get('model_id')}, but {h.model_id} is loaded; "
                                 f"neuron IDs do not transfer between checkpoints")
    return h, st


@router.get("/atlas/list")
def atlas_list():
    return {"atlases": atlas_mod.list_atlases()}


@router.get("/atlas/overview")
def atlas_overview(tag: str | None = None):
    _, st = _atlas(tag)
    return st.overview()


@router.get("/atlas/layer/{layer}")
def atlas_layer(layer: int, sort: str = "max_abs", offset: int = 0, limit: int = 100, tag: str | None = None):
    h, st = _atlas(tag)
    return _wrap(lambda: st.layer_table(h.tokenizer, layer, sort, offset, min(limit, 500)))


@router.get("/atlas/neuron/{layer}/{neuron}")
def atlas_neuron(layer: int, neuron: int, tag: str | None = None):
    h, st = _atlas(tag)
    return _wrap(lambda: st.neuron(h.tokenizer, layer, neuron))


@router.get("/atlas/search")
def atlas_search(q: str, tag: str | None = None):
    import re
    h, st = _atlas(tag)
    m = re.fullmatch(r"\s*L?\s*(\d+)\s*[·./ ]?\s*n?\s*(\d+)\s*", q, re.I)
    if m and q.strip()[:1].upper() == "L":
        return {"kind": "id", "results": [{"layer": int(m.group(1)), "neuron": int(m.group(2))}]}
    return _wrap(lambda: {"kind": "word", "results": st.search_word(h.tokenizer, q)})


# ---------------------------------------------------------------- Phase 3: J-lens, J-share, validation

from interpretability import j_share as js_mod  # noqa: E402
from interpretability.j_lens import get_lens  # noqa: E402
from interpretability.validation import validate as jvalidate  # noqa: E402


def _lens(h):
    try:
        return get_lens(h)
    except FileNotFoundError as e:
        raise HTTPException(404, f"{e}. Fit one with cloud/modal_jlens.py and download it.")


@router.get("/lens/jacobian/status")
def jlens_status():
    h = _handle()
    try:
        lens = get_lens(h)
        return {"available": True, "lens": lens.provenance, "layers": sorted(lens.jacobians) + [lens.final_layer]}
    except (FileNotFoundError, ValueError) as e:
        return {"available": False, "reason": str(e)}


@router.post("/lens/jacobian")
def jlens_route(req: schemas.JLensRequest):
    h = _handle()

    def go():
        lens = _lens(h)
        targets = [_single_token(h, s) for s in req.target_tokens]
        ids = None
        if req.kind == "neuron":
            if req.layer is None or req.neuron is None:
                raise ValueError("kind=neuron needs layer and neuron")
            out = lens.readout(h.tokenizer, req.layer, h.adapter.output_vector(req.layer, req.neuron), req.top_k, targets)
            out.update({"neuron": req.neuron, "vector": "w = down_proj.weight[:, neuron]"})
        elif req.kind == "vector":
            if req.layer is None or not req.vector or len(req.vector) != h.adapter.d_model:
                raise ValueError(f"kind=vector needs layer and a vector of length {h.adapter.d_model}")
            out = lens.readout(h.tokenizer, req.layer, torch.tensor(req.vector), req.top_k, targets)
        else:
            if not req.prompt:
                raise ValueError("kind=residual needs a prompt")
            ids = h.tokenizer(req.prompt, return_tensors="pt").input_ids[0]
            res = run_forward(h, ids, CaptureSpec(points=("resid_post",), final=False))
            T = ids.numel()
            t = req.position + T if req.position < 0 else req.position
            if not 0 <= t < T:
                raise ValueError("position out of range")
            rows = []
            for L in range(h.adapter.n_layers):
                lg = lens.transport_logits(res.get("resid_post", L)[t], L)
                rows.append({"state": f"L{L}", "top_k": metrics.top_k(h.tokenizer, lg, min(req.top_k, 10)),
                             "targets": [dict(metrics.token_info(h.tokenizer, x), **metrics.token_stats(lg, x)) for x in targets]})
            out = {"method": "J-lens of the residual: W_U · RMSNorm_final(J_l h_l) (reference jlens `apply`)",
                   "position": t, "tokens": metrics.tokens_of(h.tokenizer, ids), "layers": rows,
                   "model_prediction": metrics.top_k(h.tokenizer, res.logits[t], 10), "lens": lens.provenance}
        _hooks_clean(h)
        out["provenance"] = provenance(h, prompt=req.prompt, input_ids=ids, params=req.model_dump(exclude={"vector"}))
        return out

    return _wrap(go)


@router.post("/lens/j-share")
def jshare_route(req: schemas.JShareRequest):
    h = _handle()

    def go():
        lens = _lens(h)
        lens.check_layer(req.layer)
        if req.neuron is not None:
            v = h.adapter.output_vector(req.layer, req.neuron).detach().float() * (1 if req.sign == "+" else -1)
        elif req.vector and len(req.vector) == h.adapter.d_model:
            v = torch.tensor(req.vector)
        else:
            raise ValueError("give neuron (+ sign) or a vector of length d_model")
        D = lens.directions(req.layer)
        r = js_mod.j_share(D, v, req.k)
        out = {"layer": req.layer, "neuron": req.neuron, "sign": req.sign, "k": req.k, "j_share": r["j_share"],
               "k_used": r["k_used"], "all_coefficients_nonnegative": r["all_coefficients_nonnegative"],
               "residual_norm_sq": r["residual_norm_sq"],
               "directions": [dict(metrics.token_info(h.tokenizer, t), coefficient=c, cosine_with_w=float(D[t] @ (v / v.norm())))
                              for t, c in zip(r["token_ids"], r["coefficients"])],
               "vectors": {"w_unit": (v / v.norm()).tolist(), "verbalizable": r["verbalizable"].tolist(),
                           "unexplained": r["unexplained"].tolist()},
               "method": "nonnegative OMP with NNLS refitting over unit J-lens directions; J-share = 1 - ||w - v||^2",
               "lens": lens.provenance}
        if req.n_random:
            base = js_mod.random_baseline(lens, req.layer, n=req.n_random, k=req.k, seed=req.seed)
            out["random_baseline"] = {"j_shares": base.tolist(), **js_mod.classify(r["j_share"], base),
                                      "label": "paper (2,048)" if req.n_random == 2048 else f"quick ({req.n_random}) — not the paper's 2,048"}
        out["provenance"] = provenance(h, seed=req.seed, params=req.model_dump(exclude={"vector"}))
        return out

    return _wrap(go)


@router.post("/lens/validate")
def jvalidate_route(req: schemas.JValidateRequest):
    h = _handle()

    def go():
        lens = _lens(h)
        lens.check_layer(req.layer)
        y = _single_token(h, req.target)
        ids = h.tokenizer(req.prompt, return_tensors="pt").input_ids[0]
        if req.direction == "neuron":
            if req.neuron is None:
                raise ValueError("direction=neuron needs neuron")
            v = h.adapter.output_vector(req.layer, req.neuron).detach().float() * (1 if req.sign == "+" else -1)
        else:
            v = lens.directions(req.layer)[y].clone()
        out = jvalidate(h, lens, ids, req.layer, v, y, req.position, req.eps_rel)
        out["hooks_remaining"] = _hooks_clean(h)
        out["provenance"] = provenance(h, prompt=req.prompt, input_ids=ids, params=req.model_dump())
        return out

    return _wrap(go)


# ---------------------------------------------------------------- Phase 4: background experiments (Types 2, 3, scanner)

import asyncio  # noqa: E402

from fastapi import WebSocket, WebSocketDisconnect  # noqa: E402

from api.jobs import jobs  # noqa: E402
from experiments import type2 as type2_mod  # noqa: E402
from experiments import type3 as type3_mod  # noqa: E402
from experiments import type3_scan  # noqa: E402


def _submit(kind: str, req, fn):
    h = _handle()
    params = req.model_dump()

    def work(job):
        out = fn(h, job)
        out["hooks_remaining"] = count_hooks(h.model)
        out["provenance"] = provenance(h, seed=params.get("seed"), params=params)
        return out
    j = jobs.submit(kind, params, work)
    return {"job_id": j.id, "status": j.status, "poll": f"/experiments/{j.id}", "stream": f"/experiments/{j.id}/stream"}


@router.post("/experiments/type3")
def exp_type3(req: schemas.Type3Request):
    return _submit("type3", req, lambda h, job: type3_mod.run(
        h, req.layer, req.neuron, req.sign, req.topic_a.model_dump(), req.topic_b.model_dump(),
        max_topic_docs=req.max_topic_docs, n_ablate_docs=req.n_ablate_docs, n_random_words=req.n_random_words,
        n_random_dirs=req.n_random_dirs, n_reference_docs=req.n_reference_docs, seed=req.seed, progress=job.report))


@router.post("/experiments/type2")
def exp_type2(req: schemas.Type2Request):
    return _submit("type2", req, lambda h, job: type2_mod.run(
        h, req.layer, req.neuron, req.words_plus, req.words_minus, req.n_contexts, req.n_random_dirs,
        req.n_random_words, req.seed, progress=job.report))


@router.post("/experiments/type3/scan")
def exp_type3_scan(req: schemas.ScanRequest):
    return _submit("type3_scan", req, lambda h, job: type3_scan.scan(h, get_lens(h), req.layers, req.top, progress=job.report))


@router.get("/experiments")
def exp_list():
    return {"jobs": [j.public(include_result=False) for j in sorted(jobs.jobs.values(), key=lambda j: -j.created)]}


@router.get("/experiments/{job_id}")
def exp_get(job_id: str):
    try:
        return jobs.get(job_id).public()
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.post("/experiments/{job_id}/cancel")
def exp_cancel(job_id: str):
    try:
        return jobs.cancel(job_id).public(include_result=False)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.websocket("/experiments/{job_id}/stream")
async def exp_stream(ws: WebSocket, job_id: str):
    await ws.accept()
    try:
        while True:
            try:
                j = jobs.get(job_id)
            except KeyError:
                await ws.send_json({"error": f"no job {job_id}"})
                break
            await ws.send_json(j.public(include_result=j.status in ("done", "failed", "cancelled")))
            if j.status in ("done", "failed", "cancelled"):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        await ws.close()
