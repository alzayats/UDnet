#!/usr/bin/env bash
# One-time setup for UDnet V2 + AI Max (see AI_MAX.md).
# Clones DiffBIR at the tested commit and creates its own conda environment, so the main UDnet
# environment is left unchanged. DiffBIR downloads its model weights (about 6.7 GB) on first use.
#
#   bash setup_ai_max.sh [DIFFBIR_DIR] [ENV_NAME]
set -euo pipefail

DIR=${1:-third_party/DiffBIR}
ENV=${2:-udnet-aimax}
COMMIT=5c2d6c177c3f2965e6d33a32931248d242ef7b3d   # DiffBIR main, 29 Jul 2025 (v2.1)

command -v conda >/dev/null || { echo "conda is required (https://docs.conda.io)"; exit 1; }

if [ ! -d "$DIR/.git" ]; then
  git clone https://github.com/XPixelGroup/DiffBIR.git "$DIR"
fi
git -C "$DIR" checkout -q "$COMMIT"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  conda create -y -n "$ENV" python=3.10
fi
# DiffBIR's pinned requirements (PyTorch 2.2.2 + CUDA 11.8, xformers); pandas is needed but not listed.
conda run -n "$ENV" pip install -r "$DIR/requirements.txt" pandas

echo
echo "Done. Run:  python ai_max.py --input path/to/images --output results_ai_max/"
