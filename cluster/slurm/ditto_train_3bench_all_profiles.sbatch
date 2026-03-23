#!/bin/bash
#SBATCH --job-name=ditto-walmart-profiles
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Submit from repo root:
#   sbatch cluster/slurm/ditto_train_3bench_all_profiles.sbatch
#
# Optional overrides:
#   export CONDA_ENV_NAME=ditto-modern
#   export PROFILES="small,medium,large,all"
#   export LABEL_CONFIG=configs/labeling/benchmarks_active.yaml
#   export TRAIN_CONFIG=configs/ditto/benchmarks_training.yaml
#   export RUN_DIR_WALMART_AMAZON=output/simple_labeling/benchmark_walmart-amazon_20260302_113733
#   export DITTO_OUTPUT_ROOT=output/ditto_runs
#   export DITTO_RUN_PREFIX=walmart_amazon_profiles
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

BENCHMARK="walmart-amazon"
LABEL_CONFIG="${LABEL_CONFIG:-configs/labeling/benchmarks_active.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/ditto/benchmarks_training.yaml}"
PROFILES_RAW="${PROFILES:-small,medium,large,all}"
DITTO_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT:-output/ditto_runs}"
DITTO_RUN_PREFIX="${DITTO_RUN_PREFIX:-walmart_amazon_profiles}"
DITTO_SEED="${DITTO_SEED:-42}"
MODEL_NAME="${MODEL_NAME:-roberta-base}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
FP16="${FP16:-true}"

# Keep compatibility with previous env var naming from the request.
RUN_DIR_DBLP_SCHOLAR="${RUN_DIR_WALMART_AMAZON:-output/simple_labeling/benchmark_walmart-amazon_20260302_113733}"
RUN_DIR="${RUN_DIR_DBLP_SCHOLAR}"

if [ ! -d "${RUN_DIR}" ]; then
  echo "Missing run directory for ${BENCHMARK}: ${RUN_DIR}" >&2
  exit 1
fi

PROFILES_RAW="${PROFILES_RAW//,/ }"
read -r -a PROFILE_LIST <<< "${PROFILES_RAW}"
if [ "${#PROFILE_LIST[@]}" -eq 0 ]; then
  echo "No profiles provided." >&2
  exit 1
fi

FP16_NORM="$(echo "${FP16}" | tr '[:upper:]' '[:lower:]')"
NO_FP16_FLAG=()
if [ "${FP16_NORM}" != "true" ] && [ "${FP16_NORM}" != "1" ] && [ "${FP16_NORM}" != "yes" ] && [ "${FP16_NORM}" != "y" ]; then
  NO_FP16_FLAG=(--no-fp16)
fi

if [ ! -f "${LABEL_CONFIG}" ]; then
  echo "Label config not found: ${LABEL_CONFIG}" >&2
  exit 1
fi
if [ ! -f "${TRAIN_CONFIG}" ]; then
  echo "Train config not found: ${TRAIN_CONFIG}" >&2
  exit 1
fi

# Ensure missing "all" export is backfilled from existing active_labels_latest.csv.
ALL_TRAIN_JSON="${RUN_DIR}/profiles/all/active_labels_latest_${BENCHMARK}_all_train.json.gz"
if [ ! -f "${ALL_TRAIN_JSON}" ]; then
  RUN_NAME="$(basename "${RUN_DIR}")"
  OUTPUT_ROOT="$(dirname "${RUN_DIR}")"
  echo "Backfilling missing all-profile export for ${BENCHMARK} in ${RUN_DIR}"
  python scripts/labeling/run_benchmark_labeling.py \
    --config "${LABEL_CONFIG}" \
    --benchmark "${BENCHMARK}" \
    --profiles "all" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}" \
    --resume
fi

# Validate requested profile files exist.
for P in "${PROFILE_LIST[@]}"; do
  TRAIN_JSON="${RUN_DIR}/profiles/${P}/active_labels_latest_${BENCHMARK}_${P}_train.json.gz"
  if [ ! -f "${TRAIN_JSON}" ]; then
    echo "Missing train file: ${TRAIN_JSON}" >&2
    exit 1
  fi
done

# Build normalized valid/test splits and benchmark train config.
SPLIT_OUTPUT_ROOT="output/ditto_benchmark_runs"
SPLIT_RUN_NAME="walmart_amazon_splits_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
python scripts/ditto/run_benchmark_training.py \
  --config "${TRAIN_CONFIG}" \
  --output-root "${SPLIT_OUTPUT_ROOT}" \
  --run-name "${SPLIT_RUN_NAME}" \
  --benchmarks "${BENCHMARK}" \
  --dry-run

SPLIT_ROOT="${SPLIT_OUTPUT_ROOT}/${SPLIT_RUN_NAME}"
VAL_JSON="${SPLIT_ROOT}/${BENCHMARK}/splits/valid.json.gz"
TEST_JSON="${SPLIT_ROOT}/${BENCHMARK}/splits/test.json.gz"
TRAIN_CFG="${SPLIT_ROOT}/${BENCHMARK}/train_config.yaml"

if [ ! -f "${VAL_JSON}" ] || [ ! -f "${TEST_JSON}" ]; then
  echo "Missing normalized valid/test splits for ${BENCHMARK} in ${SPLIT_ROOT}/${BENCHMARK}/splits" >&2
  exit 1
fi
if [ ! -f "${TRAIN_CFG}" ]; then
  echo "Missing train config for ${BENCHMARK}: ${TRAIN_CFG}" >&2
  exit 1
fi

RUN_STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT_DIR="${DITTO_OUTPUT_ROOT}/${DITTO_RUN_PREFIX}_${RUN_STAMP}"
mkdir -p "${BASE_OUT_DIR}"

SUMMARY_TSV="${BASE_OUT_DIR}/metrics.tsv"
echo -e "benchmark\tprofile\trun_dir\tbest_val_f1\ttest_f1\ttest_precision\ttest_recall\ttest_accuracy" > "${SUMMARY_TSV}"

echo "Starting Ditto training"
echo "Benchmark:  ${BENCHMARK}"
echo "Profiles:   ${PROFILES_RAW}"
echo "Run dir:    ${RUN_DIR}"
echo "Output dir: ${BASE_OUT_DIR}"

for P in "${PROFILE_LIST[@]}"; do
  TRAIN_JSON="${RUN_DIR}/profiles/${P}/active_labels_latest_${BENCHMARK}_${P}_train.json.gz"
  PROFILE_OUT="${BASE_OUT_DIR}/${BENCHMARK}/${P}"
  mkdir -p "${PROFILE_OUT}"

  echo
  echo "=== Training ${BENCHMARK} / ${P} ==="
  echo "Train: ${TRAIN_JSON}"
  python scripts/ditto/train.py \
    --train-json-gz "${TRAIN_JSON}" \
    --val-json-gz "${VAL_JSON}" \
    --test-json-gz "${TEST_JSON}" \
    --output-dir "${PROFILE_OUT}" \
    --config "${TRAIN_CFG}" \
    --model-name "${MODEL_NAME}" \
    --epochs "${EPOCHS}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --seed "${DITTO_SEED}" \
    "${NO_FP16_FLAG[@]}"

  RUN_PTR="${PROFILE_OUT}/LATEST_RUN"
  if [ ! -f "${RUN_PTR}" ]; then
    echo "Missing LATEST_RUN for ${BENCHMARK}/${P} at ${PROFILE_OUT}" >&2
    exit 1
  fi
  RUN_PATH="$(cat "${RUN_PTR}")"
  METRICS_JSON="${RUN_PATH}/metrics.json"
  if [ ! -f "${METRICS_JSON}" ]; then
    echo "Missing metrics.json for ${BENCHMARK}/${P}: ${METRICS_JSON}" >&2
    exit 1
  fi

  python - "${BENCHMARK}" "${P}" "${RUN_PATH}" "${METRICS_JSON}" >> "${SUMMARY_TSV}" <<'PY'
import json
import sys

benchmark = sys.argv[1]
profile = sys.argv[2]
run_dir = sys.argv[3]
metrics_path = sys.argv[4]

with open(metrics_path, "r") as f:
    m = json.load(f)
test = m.get("test", {}) if isinstance(m, dict) else {}
best_val_f1 = float(m.get("best_val_f1", float("nan")))
test_f1 = float(test.get("f1", float("nan")))
test_precision = float(test.get("precision", float("nan")))
test_recall = float(test.get("recall", float("nan")))
test_accuracy = float(test.get("accuracy", float("nan")))
print(f"{benchmark}\t{profile}\t{run_dir}\t{best_val_f1}\t{test_f1}\t{test_precision}\t{test_recall}\t{test_accuracy}")
PY
done

echo
echo "Done. Output root: ${BASE_OUT_DIR}"
echo "Per-run metrics: ${SUMMARY_TSV}"
cat "${SUMMARY_TSV}"
