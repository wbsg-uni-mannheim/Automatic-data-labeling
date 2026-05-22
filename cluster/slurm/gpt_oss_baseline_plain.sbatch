#!/bin/bash
#SBATCH --job-name=gpt-oss-baseline-plain
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/work/aasteine/Automatic-data-labeling}"
cd "${REPO_ROOT}"
mkdir -p logs

DATA_PATH="${DATA_PATH:-output/gpt_oss_em/abt-buy_test/sft_data/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/gpt_oss_em/abt-buy_test/baseline_zero_shot_plain}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"

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

python llm_em_qwen35/baseline_zero_shot_eval_plain.py \
  --data-path "${DATA_PATH}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}"
