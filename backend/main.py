"""NeuronLab backend entry point.

    cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

Environment:
    NEURONLAB_CORS_ORIGINS  comma-separated origins (default http://localhost:3000)
    NEURONLAB_PRELOAD       model id to load at startup (optional)
    HF_TOKEN                Hugging Face token for gated models (Llama)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from models.loader import manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    pre = os.environ.get("NEURONLAB_PRELOAD")
    if pre:
        manager.load(pre)
    yield


app = FastAPI(title="NeuronLab backend", version="0.1.0-phase1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("NEURONLAB_CORS_ORIGINS", "http://localhost:3000").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
