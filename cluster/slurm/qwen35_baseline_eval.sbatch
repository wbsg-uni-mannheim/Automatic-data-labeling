#!/bin/bash
#SBATCH --job-name=qwen35-baseline
#SBATCH --partition=gpu-vram-48gb
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

# Required envs set by the submit wrapper:
: "${BENCHMARK:?BENCHMARK env var must be set (e.g. abt-buy, wdc, ...)}"
: "${FIELDS:?FIELDS env var must be set (comma-separated list)}"

TEST_JSON_GZ="${TEST_JSON_GZ:-output/ditto_baseline/${BENCHMARK}/splits/test.json.gz}"
SFT_DIR="${SFT_DIR:-output/qwen35_em/${BENCHMARK}_test/sft_data}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen35_em/${BENCHMARK}_test/baseline_zero_shot}"
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

# Only the test split is needed for zero-shot baseline. convert_wdc_to_sft.py
# requires train+valid+test, so pass the test split for all three; only
# test.jsonl is consumed downstream.
mkdir -p "${SFT_DIR}"
python llm_em_qwen35/convert_wdc_to_sft.py \
  --train-json-gz "${TEST_JSON_GZ}" \
  --valid-json-gz "${TEST_JSON_GZ}" \
  --test-json-gz "${TEST_JSON_GZ}" \
  --fields "${FIELDS}" \
  --output-dir "${SFT_DIR}"

python llm_em_qwen35/baseline_zero_shot_eval.py \
  --data-path "${SFT_DIR}/test.jsonl" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}"
