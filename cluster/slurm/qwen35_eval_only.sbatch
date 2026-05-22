#!/bin/bash
#SBATCH --job-name=qwen35-eval
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/work/aasteine/Automatic-data-labeling}"
cd "${REPO_ROOT}"
mkdir -p logs

# Required envs:
: "${ADAPTER_PATH:?ADAPTER_PATH env var must be set (path to adapter checkpoint dir)}"
: "${DATA_PATH:?DATA_PATH env var must be set (path to test.jsonl)}"
: "${OUTPUT_DIR:?OUTPUT_DIR env var must be set (where to write metrics)}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"

export HF_HOME="${HF_HOME:-/work/shared_data/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.env"
  set +a
fi
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

# shellcheck disable=SC1090
source "${REPO_ROOT}/llm_em_qwen35/.venv/bin/activate"

python llm_em_qwen35/evaluate_lora.py \
  --data-path "${DATA_PATH}" \
  --model-path "${ADAPTER_PATH}" \
  --base-model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}"

echo "=== done ==="
cat "${OUTPUT_DIR}/metrics.json"
