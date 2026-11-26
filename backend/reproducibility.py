"""Provenance attached to every result."""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def code_version() -> dict:
    root = Path(__file__).resolve().parent.parent
    sha = os.environ.get("NEURONLAB_GIT_SHA")
    dirty = None
    if not sha:
        try:
            sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                 text=True, timeout=5).stdout.strip() or None
            dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                                        text=True, timeout=5).stdout.strip())
        except Exception:
            sha = None
    return {"git_sha": sha, "dirty_worktree": dirty}


def provenance(handle, *, prompt: str | None = None, input_ids=None, seed: int | None = None,
               params: dict | None = None, mode: str = "exploratory") -> dict:
    return {
        **handle.provenance(),
        "code_version": code_version(),
        "mode": mode,
        "input_text": prompt,
        "input_token_ids": input_ids.tolist() if input_ids is not None else None,
        "seed": seed,
        "params": params or {},
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
