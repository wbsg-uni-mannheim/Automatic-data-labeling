#!/bin/bash

set -euo pipefail

# Submit Ditto training jobs in parallel, optionally for multiple labeling methods.
# Submit Ditto training jobs in parallel, optionally for multiple labeling methods.
#
# Run from repo root:
#   bash cluster/slurm/submit_ditto_training_all_parallel.sh
#
# Optional overrides before launch:
#   export CONDA_ENV_NAME=ditto-modern
#   export LABEL_RUN_ROOT=output/three_phase_labeling_ditto_only_v2
#   export LABEL_RUN_ROOT=output
#   export LABEL_RUN_ROOT=output
#   export BENCHMARKS="wdc abt-buy amazon-google dblp-acm dblp-scholar walmart-amazon"
#   export PROFILE_FILTER="small,medium,small_plus20random,medium_plus20random,all,all_plus20random"
#   export TRAIN_CONFIG=configs/ditto/benchmarks_training.yaml
#   export DITTO_OUTPUT_ROOT=output/training_from_generated_labels
#   export RUN_NAME_PREFIX=generated_ditto
#   export DITTO_SEEDS="42 52 62"
#   export PARTITION=gpu-vram-48gb
#   export CPUS_PER_TASK=12
#   export MEM=64G
#   export TIME_LIMIT=72:00:00
#   export GPU_COUNT=1

mkdir -p logs

REPO_ROOT="${PWD}"
LABEL_RUN_ROOT="${LABEL_RUN_ROOT:-output/three_phase_labeling_ditto_only_v2}"
BENCHMARKS_RAW="${BENCHMARKS:-wdc abt-buy amazon-google dblp-acm dblp-scholar walmart-amazon}"
BENCHMARKS_RAW="${BENCHMARKS_RAW//,/ }"
read -r -a BENCHMARK_LIST <<< "${BENCHMARKS_RAW}"

PROFILE_FILTER="${PROFILE_FILTER:-__all__}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/ditto/benchmarks_training.yaml}"
DITTO_OUTPUT_ROOT_DEFAULT="output/training_from_generated_labels"
DITTO_OUTPUT_ROOT_IS_EXPLICIT="${DITTO_OUTPUT_ROOT+set}"
DITTO_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT:-${DITTO_OUTPUT_ROOT_DEFAULT}}"
DITTO_OUTPUT_ROOT_DEFAULT="output/training_from_generated_labels"
DITTO_OUTPUT_ROOT_IS_EXPLICIT="${DITTO_OUTPUT_ROOT+set}"
DITTO_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT:-${DITTO_OUTPUT_ROOT_DEFAULT}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-generated_ditto}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ditto-modern}"
DITTO_SEEDS="${DITTO_SEEDS:-42 52 62}"
DITTO_SEEDS="${DITTO_SEEDS:-42 52 62}"

PARTITION="${PARTITION:-gpu-vram-48gb}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-32G}"
TIME_LIMIT="${TIME_LIMIT:-6:00:00}"
GPU_COUNT="${GPU_COUNT:-1}"

if [ ! -d "${LABEL_RUN_ROOT}" ]; then
  echo "Label run root not found: ${LABEL_RUN_ROOT}" >&2
  exit 1
fi

if [ ! -f "${TRAIN_CONFIG}" ]; then
  echo "Ditto training config not found: ${TRAIN_CONFIG}" >&2
  exit 1
fi

if [ "${#BENCHMARK_LIST[@]}" -eq 0 ]; then
  echo "No benchmarks provided." >&2
  exit 1
fi

METHOD_DIRS=()
while IFS= read -r METHOD_DIR; do
  METHOD_DIRS+=( "${METHOD_DIR}" )
done < <(find "${LABEL_RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | while read -r CANDIDATE; do
  if compgen -G "${CANDIDATE}/benchmark_*" >/dev/null; then
    echo "${CANDIDATE}"
  fi
done)

if [ "${#METHOD_DIRS[@]}" -eq 0 ] && compgen -G "${LABEL_RUN_ROOT}/benchmark_*" >/dev/null; then
  METHOD_DIRS+=( "${LABEL_RUN_ROOT}" )
fi

if [ "${#METHOD_DIRS[@]}" -eq 0 ]; then
  echo "No labeling method directories found under ${LABEL_RUN_ROOT}" >&2
  exit 1
fi

echo "Label run root: ${LABEL_RUN_ROOT}"
echo "Method dirs:    ${#METHOD_DIRS[@]}"
echo "Benchmarks:     ${BENCHMARKS_RAW}"
echo "Profile filter: ${PROFILE_FILTER}"
echo "Train config:   ${TRAIN_CONFIG}"
echo "Ditto output:   ${DITTO_OUTPUT_ROOT}"
echo "Run prefix:     ${RUN_NAME_PREFIX}"
echo "Ditto seeds:    ${DITTO_SEEDS}"
echo "Partition:      ${PARTITION}"
echo "CPUs/task:      ${CPUS_PER_TASK}"
echo "Mem:            ${MEM}"
echo "Time:           ${TIME_LIMIT}"
echo "GPUs:           ${GPU_COUNT}"
echo

for METHOD_DIR in "${METHOD_DIRS[@]}"; do
  METHOD_NAME="$(basename "${METHOD_DIR}")"
  if [ "${#METHOD_DIRS[@]}" -eq 1 ] && [ "${METHOD_DIR}" = "${LABEL_RUN_ROOT}" ] && [ -n "${DITTO_OUTPUT_ROOT_IS_EXPLICIT}" ]; then
    METHOD_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT}"
  else
    METHOD_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT}/${METHOD_NAME}"
  fi

  mkdir -p "${METHOD_OUTPUT_ROOT}"

  for BENCH in "${BENCHMARK_LIST[@]}"; do
    SAFE_BENCH="${BENCH//[^a-zA-Z0-9]/-}"
    SAFE_METHOD="${METHOD_NAME//[^a-zA-Z0-9]/-}"
    RUN_GLOB="benchmark_${BENCH}_*"

    JOB_ID="$(
      sbatch \
        --parsable \
        --job-name="ditto-${SAFE_METHOD}-${SAFE_BENCH}" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEM}" \
        --time="${TIME_LIMIT}" \
        --gres="gpu:${GPU_COUNT}" \
        --export=ALL,CONDA_ENV_NAME="${CONDA_ENV_NAME}",LABEL_RUN_ROOT="${METHOD_DIR}",RUN_GLOB="${RUN_GLOB}",PROFILE_FILTER="${PROFILE_FILTER}",TRAIN_CONFIG="${TRAIN_CONFIG}",DITTO_OUTPUT_ROOT="${METHOD_OUTPUT_ROOT}",RUN_NAME_PREFIX="${RUN_NAME_PREFIX}",DITTO_SEEDS="${DITTO_SEEDS}",METHOD_NAME="${METHOD_NAME}" \
        cluster/slurm/train_ditto_from_generated_labels.sbatch
    )"

    echo "Submitted method=${METHOD_NAME} benchmark=${BENCH}: job ${JOB_ID} (RUN_GLOB=${RUN_GLOB}, output=${METHOD_OUTPUT_ROOT})"
  done
done
