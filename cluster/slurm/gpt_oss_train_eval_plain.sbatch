#!/bin/bash
#SBATCH --job-name=gpt-oss-train-plain
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/work/aasteine/Automatic-data-labeling}"
cd "${REPO_ROOT}"
mkdir -p logs

: "${BENCHMARK:?BENCHMARK env var must be set}"
: "${FIELDS:?FIELDS env var must be set}"

TRAIN_JSON_GZ="${TRAIN_JSON_GZ:-output/ditto_baseline/${BENCHMARK}/splits/train.json.gz}"
VALID_JSON_GZ="${VALID_JSON_GZ:-output/ditto_baseline/${BENCHMARK}/splits/valid.json.gz}"
TEST_JSON_GZ="${TEST_JSON_GZ:-output/ditto_baseline/${BENCHMARK}/splits/test.json.gz}"
RUN_NAME="${RUN_NAME:-${BENCHMARK}_plain}"
SFT_DIR="${SFT_DIR:-output/gpt_oss_em/${RUN_NAME}/sft_data}"
OUTPUT_DIR="${OUTPUT_DIR:-output/gpt_oss_em/${RUN_NAME}/gpt_oss_20b_lora}"
EVAL_DIR="${EVAL_DIR:-output/gpt_oss_em/${RUN_NAME}/eval_on_test}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
PER_DEVICE_BSZ="${PER_DEVICE_BSZ:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_R="${LORA_R:-16}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
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
source "${REPO_ROOT}/scripts/archive/qwen_internal/.venv/bin/activate"

echo "=== convert splits to SFT ==="
python scripts/archive/qwen_internal/convert_wdc_to_sft.py \
  --train-json-gz "${TRAIN_JSON_GZ}" \
  --valid-json-gz "${VALID_JSON_GZ}" \
  --test-json-gz "${TEST_JSON_GZ}" \
  --fields "${FIELDS}" \
  --output-dir "${SFT_DIR}"

echo "=== train (plain transformers + peft) ==="
python scripts/archive/qwen_internal/train_plain_lora.py \
  --data-dir "${SFT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --per-device-train-batch-size "${PER_DEVICE_BSZ}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --num-train-epochs "${NUM_EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --lora-r "${LORA_R}" \
  --lora-target-modules "${LORA_TARGET_MODULES}" \
  --logging-steps "${LOGGING_STEPS}" \
  --eval-steps "${EVAL_STEPS}" \
  --save-steps "${EVAL_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-threshold "${EARLY_STOPPING_THRESHOLD}" \
  --load-best-model-at-end

echo "=== evaluate on test (plain transformers) ==="
python scripts/archive/qwen_internal/baseline_zero_shot_eval_plain.py \
  --data-path "${SFT_DIR}/test.jsonl" \
  --model-name "${OUTPUT_DIR}/final_adapter" \
  --output-dir "${EVAL_DIR}"

echo "=== done ==="
cat "${EVAL_DIR}/metrics.json"
