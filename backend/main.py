"""NeuronLab backend entry point.

    cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

Environment:
    NEURONLAB_CORS_ORIGINS  comma-separated origins (default http://localhost:3000)
    NEURONLAB_PRELOAD       model id to load at startup (optional)
    HF_TOKEN                Hugging Face token for gated models (Llama)
"""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from models.loader import manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    pre = os.environ.get("NEURONLAB_PRELOAD")
    if pre:
        manager.load(pre)
    yield


app = FastAPI(title="NeuronLab backend", version="0.4.0", lifespan=lifespan)

ACCESS_KEY = os.environ.get("NEURONLAB_ACCESS_KEY", "")


@app.middleware("http")
async def access_key(request: Request, call_next):
    """Optional shared-secret guard for hosted deployments (unset locally = no guard)."""
    if ACCESS_KEY and request.method != "OPTIONS" and request.url.path != "/health":
        got = request.headers.get("x-neuronlab-key") or request.query_params.get("key") or ""
        if not hmac.compare_digest(got, ACCESS_KEY):
            return JSONResponse({"detail": "missing or wrong access key (open the link you were given)"}, status_code=401)
    return await call_next(request)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("NEURONLAB_CORS_ORIGINS", "http://localhost:3000").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
