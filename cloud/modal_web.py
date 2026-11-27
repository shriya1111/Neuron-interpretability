"""Host NeuronLab on Modal: the FastAPI backend (CPU) and the Next.js frontend, each with its own URL.

    python -c "import secrets; print(secrets.token_urlsafe(24))"      # make an access key
    modal secret create neuronlab-access NEURONLAB_ACCESS_KEY=<key>
    modal deploy cloud/modal_web.py

Open:  https://<workspace>--neuronlab.us-east.modal.run/?key=<key>

* The backend reads the Atlas, J-lens, readouts and census from the existing Modal volumes.
* One backend container (weight edits and jobs live in its memory); it scales to zero after 15 idle minutes.
* Every API call except /health needs the access key (sent by the frontend after you open the link once).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = os.environ.get("NEURONLAB_MODAL_WORKSPACE", "shriyae3")
API_LABEL, WEB_LABEL = "neuronlab-api", "neuronlab"
DOMAIN = os.environ.get("NEURONLAB_MODAL_DOMAIN", "us-east.modal.run")  # Modal prints the exact URLs on deploy
API_URL = f"https://{WORKSPACE}--{API_LABEL}.{DOMAIN}"
WEB_URL = f"https://{WORKSPACE}--{WEB_LABEL}.{DOMAIN}"

app = modal.App("neuronlab-web")
atlas_vol = modal.Volume.from_name("neuronlab-atlas")
jlens_vol = modal.Volume.from_name("neuronlab-jlens")
hf_vol = modal.Volume.from_name("neuronlab-hf-cache", create_if_missing=True)

api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements(str(ROOT / "backend" / "requirements.txt"))
    .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_XET": "1", "NEURONLAB_PRELOAD": "Qwen/Qwen3-0.6B",
          "NEURONLAB_CORS_ORIGINS": WEB_URL, "OMP_NUM_THREADS": "8"})
    .add_local_dir(str(ROOT / "backend"), "/app/backend", ignore=["**/__pycache__", "**/.pytest_cache"])
)


@app.function(image=api_image, cpu=8.0, memory=24576, timeout=3600, scaledown_window=900, max_containers=1,
              volumes={"/app/data/atlas": atlas_vol, "/app/data/jlens": jlens_vol, "/hf": hf_vol},
              secrets=[modal.Secret.from_name("neuronlab-access")])
@modal.concurrent(max_inputs=32)
@modal.asgi_app(label=API_LABEL)
def api():
    import sys
    sys.path.insert(0, "/app/backend")
    os.chdir("/app/backend")
    from main import app as fastapi_app
    return fastapi_app


web_image = (
    modal.Image.from_registry("node:22-slim", add_python="3.12")
    .add_local_dir(str(ROOT / "frontend"), "/app/frontend", copy=True,
                   ignore=["node_modules", ".next", "**/*.tsbuildinfo"])
    .env({"NEXT_PUBLIC_BACKEND_URL": API_URL, "NEXT_TELEMETRY_DISABLED": "1"})
    .run_commands("cd /app/frontend && npm ci && npm run build")
)


@app.function(image=web_image, cpu=1.0, memory=2048, scaledown_window=900, max_containers=1)
@modal.concurrent(max_inputs=100)
@modal.web_server(3000, startup_timeout=120, label=WEB_LABEL)
def web():
    subprocess.Popen("cd /app/frontend && npx next start -p 3000 -H 0.0.0.0", shell=True)
