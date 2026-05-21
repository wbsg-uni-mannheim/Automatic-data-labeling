#!/bin/bash
set -euo pipefail

LABELLERS_RAW="${LABELLERS:-gpt qwen kimi}"
BENCHMARKS_RAW="${BENCHMARKS:-abt-buy dblp-acm dblp-scholar walmart-amazon wdc}"
DRY_RUN="${DRY_RUN:-0}"

LABELLERS_RAW="${LABELLERS_RAW//,/ }"
BENCHMARKS_RAW="${BENCHMARKS_RAW//,/ }"
read -r -a LABELLER_LIST <<< "${LABELLERS_RAW}"
read -r -a BENCHMARK_LIST <<< "${BENCHMARKS_RAW}"

mkdir -p logs

for LABELLER in "${LABELLER_LIST[@]}"; do
  for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
    JOB_NAME="ditto-llmv-${LABELLER}-${BENCHMARK}"
    if [ "${DRY_RUN}" = "1" ]; then
      echo "[dry] LABELLER=${LABELLER} BENCHMARK=${BENCHMARK} sbatch --job-name=${JOB_NAME} cluster/slurm/train_ditto_cleaned_llmvalid.sbatch"
    else
      LABELLER="${LABELLER}" BENCHMARK="${BENCHMARK}" \
        sbatch --job-name="${JOB_NAME}" cluster/slurm/train_ditto_cleaned_llmvalid.sbatch
    fi
  done
done
