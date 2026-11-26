# NeuronLab GPU backend. Build: docker build -t neuronlab-backend backend/
# Run (GPU):  docker run --gpus all -p 8000:8000 -e HF_TOKEN=... -v hf-cache:/root/.cache/huggingface neuronlab-backend
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# The default PyPI torch wheel ships CUDA libraries and also runs on CPU.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ARG GIT_SHA=unknown
ENV NEURONLAB_GIT_SHA=${GIT_SHA}
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
