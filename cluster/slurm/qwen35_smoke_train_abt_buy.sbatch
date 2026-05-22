#!/bin/bash
#SBATCH --job-name=qwen35-smoke-train-abt-buy
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/work/aasteine/Automatic-data-labeling}"
cd "${REPO_ROOT}"
mkdir -p logs

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
DATA_DIR="${DATA_DIR:-output/qwen35_em/abt-buy_small/sft_data_tiny}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen35_em/abt-buy_small/smoke_run}"

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

rm -rf "${OUTPUT_DIR}"
python llm_em_qwen35/train_unsloth_lora.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --max-seq-length 1024 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --num-train-epochs 1 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --logging-steps 1 \
  --eval-steps 4 \
  --save-steps 100
