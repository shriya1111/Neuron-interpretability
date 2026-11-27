#!/usr/bin/env bash
# Download the precomputed data NeuronLab uses (not stored in git) from your Modal volumes.
# Needs: `pip install modal` and `modal token set ...` (the account that ran the jobs).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/atlas data/jlens/qwen3-0.6b-wikitext-n1000

echo "Atlas (corpus scan, ~55 MB)…"
modal volume get --force neuronlab-atlas qwen3-0.6b-pile10k data/atlas/

echo "J-lens (~110 MB) and its config…"
for f in jacobian_lens.pt config.json; do
  modal volume get --force neuronlab-jlens "qwen3-0.6b-wikitext-n1000/$f" "data/jlens/qwen3-0.6b-wikitext-n1000/$f"
done

echo "J-share census (~20 MB)…"
modal volume get --force neuronlab-jlens qwen3-0.6b-wikitext-n1000/census data/jlens/qwen3-0.6b-wikitext-n1000/

echo "Done. J-lens readouts for the Atlas are computed locally next (~5 min on a laptop CPU):"
echo "  cd backend && python scripts/precompute_jlens_readouts.py"
