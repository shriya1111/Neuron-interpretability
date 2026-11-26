"""Atlas corpus scan: what every MLP neuron fires on, how often, and what it writes (logit lens).

Corpus (paper, Appendix B.1): NeelNanda/pile-10k minus the Books3, BookCorpus2, StackExchange and OpenWebText2
subsets (the paper reports 7,070 documents remain); first 256 tokens of each document.

Per layer L and neuron i (a = silu(g·x)(u·x), "open" = g·x > 0):
  pass 1 — 200 random documents (seed 0): thresholds thr+ = q99.9(a), thr- = q99.9(-a) over all positions
           (the paper's definition), plus the same excluding position 0 (Qwen has no BOS token, and position 0
           behaves as an attention sink with outsized activations).
  pass 2 — every document: top-K positive and top-K negative activations with the gate open (position 0 excluded),
           as (doc, position, activation); counts of open-gate firings per sign; counts above thr± per Pile domain.
  readout — logit lens of +w and -w: top tokens of W_U (gamma_final ⊙ ±w).

    NEURONLAB_GPU=A10G modal deploy cloud/modal_atlas.py
    python cloud/modal_atlas.py spawn --tag qwen3-0.6b-pile10k
    modal volume get neuronlab-atlas qwen3-0.6b-pile10k data/atlas/
"""

from __future__ import annotations

import json
import os
import time

import modal

GPU = os.environ.get("NEURONLAB_GPU", "A10G")
app = modal.App("neuronlab-atlas")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=5.5", "datasets", "huggingface_hub", "numpy")
    .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_XET": "1"})
)
out_vol = modal.Volume.from_name("neuronlab-atlas", create_if_missing=True)
hf_vol = modal.Volume.from_name("neuronlab-hf-cache", create_if_missing=True)
EXCLUDED = {"Books3", "BookCorpus2", "StackExchange", "OpenWebText2"}


@app.function(image=image, gpu=GPU, volumes={"/out": out_vol, "/hf": hf_vol}, timeout=4 * 3600, memory=49152, cpu=8.0)
def scan(model_id: str, tag: str, k: int = 24, n_tokens: int = 256, batch: int = 16, n_ref_docs: int = 200,
         readout_k: int = 12, dtype: str = "float32", max_docs: int | None = None) -> dict:
    import numpy as np
    import torch
    import transformers
    from datasets import load_dataset

    t0 = time.time()
    torch.manual_seed(0)
    out = f"/out/{tag}"
    os.makedirs(out, exist_ok=True)
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dt).to("cuda").eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    layers = model.model.layers
    nL, dM = len(layers), layers[0].mlp.gate_proj.weight.shape[0]

    # ---- corpus
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    docs, domains = [], []
    for row in ds:
        name = (row.get("meta") or {}).get("pile_set_name", "unknown")
        if name in EXCLUDED:
            continue
        ids = tok(row["text"], add_special_tokens=False).input_ids[:n_tokens]
        if len(ids) < 2:
            continue
        docs.append(ids)
        domains.append(name)
        if max_docs and len(docs) >= max_docs:
            break
    D = len(docs)
    dom_names = sorted(set(domains))
    dom_idx = np.array([dom_names.index(d) for d in domains], dtype=np.int16)
    tokmat = np.full((D, n_tokens), -1, dtype=np.int32)
    for i, ids in enumerate(docs):
        tokmat[i, :len(ids)] = ids
    np.save(f"{out}/tokens.npy", tokmat)
    with open(f"{out}/docs.json", "w") as f:
        json.dump({"n_docs": D, "domains": dom_names, "doc_domain": dom_idx.tolist(),
                   "lengths": [len(d) for d in docs]}, f)
    hf_vol.commit()
    print(f"[atlas] {D} documents kept after excluding {sorted(EXCLUDED)}", flush=True)

    # ---- hooks: capture gate pre-activation and activation (input of down_proj) for every layer
    cap = {}

    def mk(L):
        def gate_hook(m, i, o):
            cap[("g", L)] = o

        def down_pre(m, a):
            cap[("a", L)] = a[0]
        return gate_hook, down_pre
    handles = []
    for L, lay in enumerate(layers):
        gh, dp = mk(L)
        handles.append(lay.mlp.gate_proj.register_forward_hook(gh))
        handles.append(lay.mlp.down_proj.register_forward_pre_hook(dp))

    def batches(indices):
        for s in range(0, len(indices), batch):
            idx = indices[s:s + batch]
            Tm = max(len(docs[i]) for i in idx)
            ids = torch.full((len(idx), Tm), tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((len(idx), Tm), dtype=torch.bool)
            for r, i in enumerate(idx):
                ids[r, :len(docs[i])] = torch.tensor(docs[i])
                mask[r, :len(docs[i])] = True
            yield idx, ids.cuda(), mask.cuda()

    # ---- pass 1: thresholds on 200 random documents
    g = torch.Generator().manual_seed(0)
    ref = sorted(torch.randperm(D, generator=g)[:n_ref_docs].tolist())
    ref_acts = [[] for _ in range(nL)]
    is_pos0 = []
    with torch.no_grad():
        for idx, ids, mask in batches(ref):
            # right-padded causal LM: padding never affects earlier real tokens, so no attention mask is needed
            model(input_ids=ids, use_cache=False)
            p0 = torch.zeros_like(mask)
            p0[:, 0] = True
            is_pos0.append(p0[mask].cpu())
            for L in range(nL):
                ref_acts[L].append(cap[("a", L)].float()[mask].cpu())
            print(f"[atlas] reference batch done ({len(idx)} docs), {time.time() - t0:.0f}s", flush=True)
    is_pos0 = torch.cat(is_pos0)
    thr = {}
    for L in range(nL):
        A = torch.cat(ref_acts[L])
        A0 = A[~is_pos0]
        kk = max(1, int(round(0.999 * A.shape[0])))
        kk0 = max(1, int(round(0.999 * A0.shape[0])))
        torch.set_num_threads(8)
        thr[L] = {"pos": A.kthvalue(kk, 0).values, "neg": (-A).kthvalue(kk, 0).values,
                  "pos_nz": A0.kthvalue(kk0, 0).values, "neg_nz": (-A0).kthvalue(kk0, 0).values}
        ref_acts[L] = None
        del A, A0
        print(f"[atlas] thresholds layer {L}, {time.time() - t0:.0f}s", flush=True)
    print(f"[atlas] thresholds from {len(ref)} reference docs, {time.time() - t0:.0f}s", flush=True)

    # ---- pass 2: top firings, counts, per-domain counts above threshold
    NEG = -1e30
    topv = {s: torch.full((nL, k, dM), NEG, device="cuda") for s in ("pos", "neg")}
    topi = {s: torch.full((nL, k, dM), -1, dtype=torch.long, device="cuda") for s in ("pos", "neg")}
    n_open = {s: torch.zeros((nL, dM), dtype=torch.long, device="cuda") for s in ("pos", "neg")}
    n_strong = {s: torch.zeros((nL, len(dom_names), dM), dtype=torch.long, device="cuda") for s in ("pos", "neg")}
    max_abs = torch.zeros((nL, dM), device="cuda")
    n_positions = torch.zeros(len(dom_names), dtype=torch.long)
    thr_gpu = {L: {kk: v.cuda() for kk, v in thr[L].items()} for L in range(nL)}
    with torch.no_grad():
        for bi, (idx, ids, mask) in enumerate(batches(list(range(D)))):
            model(input_ids=ids, use_cache=False)
            B, T = ids.shape
            m0 = mask.clone()
            m0[:, 0] = False
            flat_id = (torch.tensor(idx, device="cuda").view(-1, 1) * n_tokens + torch.arange(T, device="cuda").view(1, -1))
            fm = m0.flatten()
            fid = flat_id.flatten()[fm]
            bdom = torch.tensor(dom_idx[idx], device="cuda").view(-1, 1).expand(B, T).flatten()[fm]
            for d in bdom.unique().tolist():
                n_positions[d] += int((bdom == d).sum())
            for L in range(nL):
                a = cap[("a", L)].float().reshape(B * T, dM)[fm]
                gt = cap[("g", L)].float().reshape(B * T, dM)[fm] > 0
                max_abs[L] = torch.maximum(max_abs[L], a.abs().max(0).values)
                for s, sg in (("pos", 1.0), ("neg", -1.0)):
                    v = sg * a
                    ok = gt & (v > 0)
                    n_open[s][L] += ok.sum(0)
                    strong = v > thr_gpu[L][s]
                    for d in bdom.unique().tolist():
                        n_strong[s][L, d] += (strong & (bdom == d).view(-1, 1)).sum(0)
                    cand = torch.where(ok, v, torch.full_like(v, NEG))
                    kk = min(k, cand.shape[0])
                    cv, ci = cand.topk(kk, dim=0)
                    allv = torch.cat([topv[s][L], cv], 0)
                    alli = torch.cat([topi[s][L], fid[ci]], 0)
                    nv, ni = allv.topk(k, dim=0)
                    topv[s][L] = nv
                    topi[s][L] = alli.gather(0, ni)
            if bi % 50 == 0:
                print(f"[atlas] batch {bi}, {time.time() - t0:.0f}s", flush=True)

    # ---- logit-lens readouts of +w and -w
    W_U = model.lm_head.weight.float()
    gam = model.model.norm.weight.float()
    read = {}
    with torch.no_grad():
        for L in range(nL):
            Wd = layers[L].mlp.down_proj.weight.float()  # [d, dM]
            ids_p, sc_p, ids_n, sc_n = [], [], [], []
            for c in range(0, dM, 512):
                S = W_U @ (gam.view(-1, 1) * Wd[:, c:c + 512])  # [V, chunk]
                vp, ip = S.topk(readout_k, dim=0)
                vn, inn = (-S).topk(readout_k, dim=0)
                ids_p.append(ip.T); sc_p.append(vp.T); ids_n.append(inn.T); sc_n.append(vn.T)
            read[L] = [torch.cat(x).cpu().numpy() for x in (ids_p, sc_p, ids_n, sc_n)]

    for h in handles:
        h.remove()
    for L in range(nL):
        np.savez_compressed(
            f"{out}/L{L:02d}.npz",
            top_pos_val=topv["pos"][L].T.cpu().numpy().astype(np.float32), top_pos_idx=topi["pos"][L].T.cpu().numpy(),
            top_neg_val=(-topv["neg"][L]).T.cpu().numpy().astype(np.float32), top_neg_idx=topi["neg"][L].T.cpu().numpy(),
            n_open_pos=n_open["pos"][L].cpu().numpy(), n_open_neg=n_open["neg"][L].cpu().numpy(),
            n_strong_pos=n_strong["pos"][L].T.cpu().numpy(), n_strong_neg=n_strong["neg"][L].T.cpu().numpy(),
            thr_pos=thr[L]["pos"].numpy(), thr_neg=thr[L]["neg"].numpy(),
            thr_pos_nz=thr[L]["pos_nz"].numpy(), thr_neg_nz=thr[L]["neg_nz"].numpy(),
            max_abs=max_abs[L].cpu().numpy(),
            read_pos_ids=read[L][0], read_pos_score=read[L][1], read_neg_ids=read[L][2], read_neg_score=read[L][3],
        )
    meta = {
        "model_id": model_id, "model_revision": getattr(model.config, "_commit_hash", None), "dtype": dtype, "gpu": GPU,
        "corpus": "NeelNanda/pile-10k minus " + ", ".join(sorted(EXCLUDED)), "n_docs": D, "tokens_per_doc": n_tokens,
        "positions_scanned_excl_pos0": int(n_positions.sum()), "positions_per_domain": dict(zip(dom_names, n_positions.tolist())),
        "top_k": k, "reference_docs_for_thresholds": len(ref), "threshold_quantile": 0.999, "readout_k": readout_k,
        "notes": ["top firings require an open gate and exclude position 0",
                  "thr_pos/thr_neg follow the paper (all positions); *_nz exclude position 0",
                  "index encoding: flat = doc * tokens_per_doc + position"],
        "torch": torch.__version__, "transformers": transformers.__version__, "seconds": round(time.time() - t0, 1),
    }
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    out_vol.commit()
    return meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["spawn"])
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tag", default="qwen3-0.6b-pile10k")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--n-ref-docs", type=int, default=200)
    a = ap.parse_args()
    call = modal.Function.from_name("neuronlab-atlas", "scan").spawn(a.model_id, a.tag, n_ref_docs=a.n_ref_docs,
                                                                      max_docs=a.max_docs)
    print("spawned function call:", call.object_id)
