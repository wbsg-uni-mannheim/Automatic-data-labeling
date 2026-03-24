#!/bin/bash

set -euo pipefail

# Submit one Ditto training job per benchmark in parallel.
#
# Run from repo root:
#   bash cluster/slurm/submit_ditto_training_all_parallel.sh
#
# Optional overrides before launch:
#   export CONDA_ENV_NAME=ditto-modern
#   export LABEL_RUN_ROOT=output/three_phase_labeling_ditto_only_v2
#   export BENCHMARKS="wdc abt-buy amazon-google dblp-acm dblp-scholar walmart-amazon"
#   export PROFILE_FILTER="small,medium,small_plus20random,medium_plus20random,all,all_plus20random"
#   export TRAIN_CONFIG=configs/ditto/benchmarks_training.yaml
#   export DITTO_OUTPUT_ROOT=output/ditto_generated_labels
#   export RUN_NAME_PREFIX=generated_ditto
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
DITTO_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT:-output/training_from_generated_labels/three_phase_labeling_ditto_only_v2}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-generated_ditto}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ditto-modern}"

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

echo "Label run root: ${LABEL_RUN_ROOT}"
echo "Benchmarks:     ${BENCHMARKS_RAW}"
echo "Profile filter: ${PROFILE_FILTER}"
echo "Train config:   ${TRAIN_CONFIG}"
echo "Ditto output:   ${DITTO_OUTPUT_ROOT}"
echo "Run prefix:     ${RUN_NAME_PREFIX}"
echo "Partition:      ${PARTITION}"
echo "CPUs/task:      ${CPUS_PER_TASK}"
echo "Mem:            ${MEM}"
echo "Time:           ${TIME_LIMIT}"
echo "GPUs:           ${GPU_COUNT}"
echo

for BENCH in "${BENCHMARK_LIST[@]}"; do
  SAFE_BENCH="${BENCH//[^a-zA-Z0-9]/-}"
  RUN_GLOB="benchmark_${BENCH}_*"

  JOB_ID="$(
    sbatch \
      --parsable \
      --job-name="ditto-${SAFE_BENCH}" \
      --partition="${PARTITION}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --mem="${MEM}" \
      --time="${TIME_LIMIT}" \
      --gres="gpu:${GPU_COUNT}" \
      --export=ALL,CONDA_ENV_NAME="${CONDA_ENV_NAME}",LABEL_RUN_ROOT="${LABEL_RUN_ROOT}",RUN_GLOB="${RUN_GLOB}",PROFILE_FILTER="${PROFILE_FILTER}",TRAIN_CONFIG="${TRAIN_CONFIG}",DITTO_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT}",RUN_NAME_PREFIX="${RUN_NAME_PREFIX}" \
      cluster/slurm/train_ditto_from_generated_labels.sbatch
  )"

  echo "Submitted ${BENCH}: job ${JOB_ID} (RUN_GLOB=${RUN_GLOB})"
done
