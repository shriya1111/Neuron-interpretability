"""Minimal background jobs for long experiments: progress reporting and cooperative cancellation."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"  # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = ""
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def report(self, progress: float, message: str = "") -> None:
        if self._cancel.is_set():
            raise Cancelled()
        self.progress = max(0.0, min(1.0, progress))
        self.message = message

    def public(self, include_result: bool = True) -> dict:
        out = {"id": self.id, "kind": self.kind, "status": self.status, "progress": self.progress,
               "message": self.message, "error": self.error, "params": self.params,
               "created": self.created, "finished": self.finished}
        if include_result:
            out["result"] = self.result
        return out


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    def submit(self, kind: str, params: dict, fn) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, params)
        self.jobs[job.id] = job

        def run():
            job.status = "running"
            try:
                job.result = fn(job)
                job.status = "done"
                job.progress = 1.0
            except Cancelled:
                job.status = "cancelled"
            except Exception as e:  # noqa: BLE001 - surface any failure to the client
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.message = traceback.format_exc(limit=3)
            job.finished = time.time()

        threading.Thread(target=run, daemon=True).start()
        return job

    def get(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise KeyError(f"no job {job_id}")
        return self.jobs[job_id]

    def cancel(self, job_id: str) -> Job:
        j = self.get(job_id)
        j._cancel.set()
        return j


jobs = JobManager()
