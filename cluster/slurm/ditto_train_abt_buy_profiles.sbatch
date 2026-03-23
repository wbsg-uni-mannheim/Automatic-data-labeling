#!/bin/bash
#SBATCH --job-name=ditto-abt-buy-profiles
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Submit from repo root:
#   sbatch cluster/slurm/ditto_train_abt_buy_profiles.sbatch
#
# Optional overrides:
#   export CONDA_ENV_NAME=ditto-modern
#   export LABEL_CONFIG=configs/labeling/benchmarks_active.yaml
#   export LABEL_RUN_NAME=abt_buy_local_test
#   export LABEL_OUTPUT_ROOT=output/simple_labeling
#   export PROFILES="small,medium,large,all"
#   export DITTO_OUTPUT_ROOT=output/ditto_runs
#   export DITTO_RUN_PREFIX=abt_buy_profiles
#   export DITTO_SEED=42
#   export MODEL_NAME=roberta-base
#   export EPOCHS=50
#   export EARLY_STOPPING_PATIENCE=10
#   export FP16=true

mkdir -p logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-ditto-modern}"
else
  echo "conda not found in PATH; aborting" >&2
  exit 1
fi

LABEL_CONFIG="${LABEL_CONFIG:-configs/labeling/benchmarks_active.yaml}"
LABEL_OUTPUT_ROOT="${LABEL_OUTPUT_ROOT:-output/simple_labeling}"
LABEL_RUN_NAME="${LABEL_RUN_NAME:-abt_buy_local_test}"
PROFILES_RAW="${PROFILES:-small,medium,large,all}"
DITTO_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT:-output/ditto_runs}"
DITTO_RUN_PREFIX="${DITTO_RUN_PREFIX:-abt_buy_profiles}"
DITTO_SEED="${DITTO_SEED:-42}"
MODEL_NAME="${MODEL_NAME:-roberta-base}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
FP16="${FP16:-true}"

PROFILES_RAW="${PROFILES_RAW//,/ }"
read -r -a PROFILE_LIST <<< "${PROFILES_RAW}"
if [ "${#PROFILE_LIST[@]}" -eq 0 ]; then
  echo "No profiles provided." >&2
  exit 1
fi

RUN_DIR="${LABEL_OUTPUT_ROOT}/${LABEL_RUN_NAME}"
CANON_LEFT="${RUN_DIR}/canonical/left.csv"
CANON_RIGHT="${RUN_DIR}/canonical/right.csv"

if [ ! -f "${LABEL_CONFIG}" ]; then
  echo "Label config not found: ${LABEL_CONFIG}" >&2
  exit 1
fi
if [ ! -f "${CANON_LEFT}" ] || [ ! -f "${CANON_RIGHT}" ]; then
  echo "Canonical CSVs not found under ${RUN_DIR}/canonical." >&2
  echo "Expected: ${CANON_LEFT} and ${CANON_RIGHT}" >&2
  exit 1
fi

echo "Preparing abt-buy profile train files (if missing)..."
NEEDS_EXPORT=0
for P in "${PROFILE_LIST[@]}"; do
  TRAIN_JSON="${RUN_DIR}/profiles/${P}/active_labels_latest_abt-buy_${P}_train.json.gz"
  if [ ! -f "${TRAIN_JSON}" ]; then
    NEEDS_EXPORT=1
    break
  fi
done

if [ "${NEEDS_EXPORT}" = "1" ]; then
  PROFILE_CSV="$(IFS=,; echo "${PROFILE_LIST[*]}")"
  python scripts/labeling/run_benchmark_labeling.py \
    --config "${LABEL_CONFIG}" \
    --benchmark "abt-buy" \
    --profiles "${PROFILE_CSV}" \
    --output-root "${LABEL_OUTPUT_ROOT}" \
    --run-name "${LABEL_RUN_NAME}" \
    --resume
fi

echo "Generating normalized abt-buy valid/test splits (dry-run benchmark prep)..."
SPLIT_OUTPUT_ROOT="output/ditto_benchmark_runs"
SPLIT_RUN_NAME="abt_buy_splits_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
python scripts/ditto/run_benchmark_training.py \
  --config configs/ditto/benchmarks_training.yaml \
  --output-root "${SPLIT_OUTPUT_ROOT}" \
  --run-name "${SPLIT_RUN_NAME}" \
  --benchmarks "abt-buy" \
  --dry-run

VAL_JSON_GZ="${SPLIT_OUTPUT_ROOT}/${SPLIT_RUN_NAME}/abt-buy/splits/valid.json.gz"
TEST_JSON_GZ="${SPLIT_OUTPUT_ROOT}/${SPLIT_RUN_NAME}/abt-buy/splits/test.json.gz"
if [ ! -f "${VAL_JSON_GZ}" ] || [ ! -f "${TEST_JSON_GZ}" ]; then
  echo "Missing normalized validation/test splits." >&2
  echo "Expected: ${VAL_JSON_GZ} and ${TEST_JSON_GZ}" >&2
  exit 1
fi

RUN_STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT_DIR="${DITTO_OUTPUT_ROOT}/${DITTO_RUN_PREFIX}_${LABEL_RUN_NAME}_${RUN_STAMP}"
mkdir -p "${BASE_OUT_DIR}"
TRAIN_CFG="${BASE_OUT_DIR}/abt_buy_train_config.yaml"

cat > "${TRAIN_CFG}" <<EOF
model_name: ${MODEL_NAME}
batch_size: 64
max_len: 256
epochs: ${EPOCHS}
lr: 5.0e-5
weight_decay: 0.01
warmup_ratio: 0.1
grad_accum_steps: 1
early_stopping_patience: ${EARLY_STOPPING_PATIENCE}
seed: ${DITTO_SEED}
num_workers: 2
fp16: ${FP16}
da: del
alpha_aug: 0.8
summarize: false
dk: null
spacy_model: en_core_web_sm
fields: title,description,price
max_field_len: 350
EOF

echo "Starting Ditto training for profiles: ${PROFILES_RAW}"
echo "Run dir: ${BASE_OUT_DIR}"
echo "Validation: ${VAL_JSON_GZ}"
echo "Test: ${TEST_JSON_GZ}"

for P in "${PROFILE_LIST[@]}"; do
  TRAIN_JSON="${RUN_DIR}/profiles/${P}/active_labels_latest_abt-buy_${P}_train.json.gz"
  if [ ! -f "${TRAIN_JSON}" ]; then
    echo "Missing profile train file: ${TRAIN_JSON}" >&2
    exit 1
  fi

  PROFILE_OUT="${BASE_OUT_DIR}/${P}"
  mkdir -p "${PROFILE_OUT}"

  echo
  echo "=== Training profile ${P} ==="
  echo "Train: ${TRAIN_JSON}"
  python scripts/ditto/train.py \
    --train-json-gz "${TRAIN_JSON}" \
    --val-json-gz "${VAL_JSON_GZ}" \
    --test-json-gz "${TEST_JSON_GZ}" \
    --output-dir "${PROFILE_OUT}" \
    --config "${TRAIN_CFG}" \
    --seed "${DITTO_SEED}"
done

echo
echo "Done. Outputs: ${BASE_OUT_DIR}"
