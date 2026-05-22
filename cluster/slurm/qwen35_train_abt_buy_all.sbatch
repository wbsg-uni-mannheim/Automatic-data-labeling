#!/bin/bash
#SBATCH --job-name=qwen35-train-abt-buy-all
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/work/aasteine/Automatic-data-labeling}"
cd "${REPO_ROOT}"
mkdir -p logs

TRAIN_JSON_GZ="${TRAIN_JSON_GZ:-output/three_phase_labeling_ditto_only_v2/benchmark_abt-buy_20260323_202820/profiles/all/active_labels_latest_abt-buy_all_train.json.gz}"
VALID_JSON_GZ="${VALID_JSON_GZ:-output/ditto_baseline/abt-buy/splits/valid.json.gz}"
TEST_JSON_GZ="${TEST_JSON_GZ:-output/ditto_baseline/abt-buy/splits/test.json.gz}"
FIELDS="${FIELDS:-title,description,price}"

RUN_NAME="${RUN_NAME:-abt-buy_all}"
SFT_DIR="${SFT_DIR:-output/qwen35_em/${RUN_NAME}/sft_data}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen35_em/${RUN_NAME}/qwen35_9b_lora}"
EVAL_DIR="${EVAL_DIR:-output/qwen35_em/${RUN_NAME}/eval_on_test}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"

# Training knobs (README-recommended + early-stopping).
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
PER_DEVICE_BSZ="${PER_DEVICE_BSZ:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0}"

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

echo "=== convert splits to SFT ==="
python llm_em_qwen35/convert_wdc_to_sft.py \
  --train-json-gz "${TRAIN_JSON_GZ}" \
  --valid-json-gz "${VALID_JSON_GZ}" \
  --test-json-gz "${TEST_JSON_GZ}" \
  --fields "${FIELDS}" \
  --output-dir "${SFT_DIR}"

echo "=== train ==="
python llm_em_qwen35/train_unsloth_lora.py \
  --data-dir "${SFT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --per-device-train-batch-size "${PER_DEVICE_BSZ}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --num-train-epochs "${NUM_EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --lora-r "${LORA_R}" \
  --logging-steps "${LOGGING_STEPS}" \
  --eval-steps "${EVAL_STEPS}" \
  --save-steps "${EVAL_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-threshold "${EARLY_STOPPING_THRESHOLD}" \
  --load-best-model-at-end

echo "=== evaluate on test ==="
python llm_em_qwen35/evaluate_lora.py \
  --data-path "${SFT_DIR}/test.jsonl" \
  --model-path "${OUTPUT_DIR}/final_adapter" \
  --base-model-name "${MODEL_NAME}" \
  --output-dir "${EVAL_DIR}"

echo "=== done ==="
cat "${EVAL_DIR}/metrics.json"
