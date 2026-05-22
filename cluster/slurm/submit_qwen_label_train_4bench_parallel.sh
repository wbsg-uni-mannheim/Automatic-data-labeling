#!/bin/bash

set -euo pipefail

# Submit, in parallel, one labeling job + one chained 3-seed Ditto training job
# per benchmark, using OpenRouter Qwen 3.6 Plus for active labeling.
#
# Run from repo root:
#   bash cluster/slurm/submit_qwen_label_train_4bench_parallel.sh
#
# Optional overrides:
#   export BENCHMARKS="wdc dblp-acm dblp-scholar walmart-amazon"
#   export PROFILES="small,medium,large,all"
#   export LABEL_OUTPUT_ROOT=output/ditto_labeling/qwen
#   export DITTO_OUTPUT_ROOT=output/training_from_generated_labels/qwen
#   export RUN_NAME_PREFIX=benchmark
#   export DITTO_SEEDS="42 52 62"
#   export PROFILE_FILTER="__all__"   # which profiles to train; default = all in manifest
#   export LABEL_TIME=96:00:00
#   export TRAIN_TIME=72:00:00
#   export DRY_RUN=1                  # print sbatch invocations without submitting

REPO_ROOT="${PWD}"

BENCHMARKS_RAW="${BENCHMARKS:-wdc dblp-acm dblp-scholar walmart-amazon}"
BENCHMARKS_RAW="${BENCHMARKS_RAW//,/ }"
read -r -a BENCHMARK_LIST <<< "${BENCHMARKS_RAW}"

PROFILES="${PROFILES:-small,medium,large,all}"
LABEL_OUTPUT_ROOT="${LABEL_OUTPUT_ROOT:-output/ditto_labeling/qwen}"
DITTO_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT:-output/training_from_generated_labels/qwen}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-benchmark}"
DITTO_SEEDS="${DITTO_SEEDS:-42 52 62}"
PROFILE_FILTER="${PROFILE_FILTER:-__all__}"
LABEL_TIME="${LABEL_TIME:-96:00:00}"
TRAIN_TIME="${TRAIN_TIME:-72:00:00}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ditto-modern}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/ditto/benchmarks_training.yaml}"
RUN_NAME_PREFIX_TRAIN="${RUN_NAME_PREFIX_TRAIN:-generated_ditto_qwen36plus}"
DRY_RUN="${DRY_RUN:-0}"

LABEL_SBATCH="cluster/slurm/run_three_phase_labeling_qwen36plus.sbatch"
TRAIN_SBATCH="cluster/slurm/train_ditto_from_generated_labels.sbatch"

if [ ! -f "${LABEL_SBATCH}" ]; then
  echo "Missing labeling sbatch: ${LABEL_SBATCH}" >&2
  exit 1
fi
if [ ! -f "${TRAIN_SBATCH}" ]; then
  echo "Missing training sbatch: ${TRAIN_SBATCH}" >&2
  exit 1
fi

mkdir -p logs
mkdir -p "${LABEL_OUTPUT_ROOT}"
mkdir -p "${DITTO_OUTPUT_ROOT}"

echo "Benchmarks:        ${BENCHMARKS_RAW}"
echo "Profiles:          ${PROFILES}"
echo "Label output root: ${LABEL_OUTPUT_ROOT}"
echo "Ditto output root: ${DITTO_OUTPUT_ROOT}"
echo "Run name prefix:   ${RUN_NAME_PREFIX}"
echo "Ditto seeds:       ${DITTO_SEEDS}"
echo "Profile filter:    ${PROFILE_FILTER}"
echo "Label walltime:    ${LABEL_TIME}"
echo "Train walltime:    ${TRAIN_TIME}"
echo "Dry run:           ${DRY_RUN}"
echo

TS_BASE="$(date +%Y%m%d_%H%M%S)"

submit_pair() {
  local BENCH="$1"
  local SAFE_BENCH="${BENCH//[^a-zA-Z0-9]/-}"
  local RUN_NAME="${RUN_NAME_PREFIX}_${BENCH}_${TS_BASE}"

  # Use a sub-shell with `export` + `--export=ALL`. This avoids SLURM's comma-as-separator
  # parsing in --export=KEY=VAL,... which corrupts PROFILES=small,medium,large,all.
  local LABEL_JOB_ID
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] LABEL ${BENCH}: env { BENCHMARK=${BENCH} PROFILES=${PROFILES} RUN_NAME=${RUN_NAME} ... } sbatch ${LABEL_SBATCH}"
    LABEL_JOB_ID="DRY-LABEL-${SAFE_BENCH}"
  else
    LABEL_JOB_ID="$(
      BENCHMARK="${BENCH}" \
      PROFILES="${PROFILES}" \
      OUTPUT_ROOT="${LABEL_OUTPUT_ROOT}" \
      RUN_NAME="${RUN_NAME}" \
      RUN_NAME_PREFIX="${RUN_NAME_PREFIX}" \
      RESUME=0 \
      CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
      sbatch \
        --parsable \
        --job-name="active-${SAFE_BENCH}-qwen36" \
        --time="${LABEL_TIME}" \
        --output="logs/active-${SAFE_BENCH}-qwen36-%j.out" \
        --error="logs/active-${SAFE_BENCH}-qwen36-%j.err" \
        --export=ALL \
        "${LABEL_SBATCH}"
    )"
    echo "Submitted labeling ${BENCH}: job ${LABEL_JOB_ID} (run_name=${RUN_NAME})"
  fi

  local TRAIN_JOB_ID
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] TRAIN ${BENCH} (afterok:${LABEL_JOB_ID}): env { LABEL_RUN_ROOT=${LABEL_OUTPUT_ROOT} RUN_GLOB=${RUN_NAME} DITTO_SEEDS='${DITTO_SEEDS}' PROFILE_FILTER=${PROFILE_FILTER} ... } sbatch ${TRAIN_SBATCH}"
    TRAIN_JOB_ID="DRY-TRAIN-${SAFE_BENCH}"
  else
    TRAIN_JOB_ID="$(
      CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
      LABEL_RUN_ROOT="${LABEL_OUTPUT_ROOT}" \
      RUN_GLOB="${RUN_NAME}" \
      PROFILE_FILTER="${PROFILE_FILTER}" \
      TRAIN_CONFIG="${TRAIN_CONFIG}" \
      DITTO_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT}" \
      RUN_NAME_PREFIX="${RUN_NAME_PREFIX_TRAIN}" \
      DITTO_SEEDS="${DITTO_SEEDS}" \
      METHOD_NAME="qwen36plus_three_phase_v2" \
      sbatch \
        --parsable \
        --dependency="afterok:${LABEL_JOB_ID}" \
        --kill-on-invalid-dep=yes \
        --job-name="ditto-${SAFE_BENCH}-qwen36-3runs" \
        --time="${TRAIN_TIME}" \
        --output="logs/ditto-${SAFE_BENCH}-qwen36-3runs-%j.out" \
        --error="logs/ditto-${SAFE_BENCH}-qwen36-3runs-%j.err" \
        --export=ALL \
        "${TRAIN_SBATCH}"
    )"
    echo "  -> Chained training ${BENCH}: job ${TRAIN_JOB_ID} (afterok:${LABEL_JOB_ID})"
  fi
  echo
}

for BENCH in "${BENCHMARK_LIST[@]}"; do
  submit_pair "${BENCH}"
done

echo "Done. Use 'squeue -u \$USER' to monitor."
