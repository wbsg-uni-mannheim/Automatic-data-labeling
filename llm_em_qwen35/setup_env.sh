#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip==26.0.1 wheel==0.45.1 setuptools==80.9.0

# CUDA 12.8 wheels. Override TORCH_INDEX_URL if the training host uses a
# different CUDA wheel index.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install --index-url "${TORCH_INDEX_URL}" \
  torch==2.11.0 \
  torchvision==0.26.0

python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

python "${SCRIPT_DIR}/check_env.py"

