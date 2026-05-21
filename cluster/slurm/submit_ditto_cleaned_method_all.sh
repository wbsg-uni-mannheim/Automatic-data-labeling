#!/bin/bash
set -euo pipefail

# Submit one sbatch per (method, benchmark). Each runs 3 seeds.

METHODS_RAW="${METHODS:-seedround simpleactive}"
BENCHMARKS_RAW="${BENCHMARKS:-abt-buy dblp-acm dblp-scholar walmart-amazon wdc}"
DRY_RUN="${DRY_RUN:-0}"

METHODS_RAW="${METHODS_RAW//,/ }"
BENCHMARKS_RAW="${BENCHMARKS_RAW//,/ }"
read -r -a METHOD_LIST <<< "${METHODS_RAW}"
read -r -a BENCHMARK_LIST <<< "${BENCHMARKS_RAW}"

mkdir -p logs

for METHOD in "${METHOD_LIST[@]}"; do
  for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
    JOB_NAME="ditto-${METHOD}-${BENCHMARK}"
    if [ "${DRY_RUN}" = "1" ]; then
      echo "[dry] METHOD=${METHOD} BENCHMARK=${BENCHMARK} sbatch --job-name=${JOB_NAME} cluster/slurm/train_ditto_cleaned_method.sbatch"
    else
      METHOD="${METHOD}" BENCHMARK="${BENCHMARK}" \
        sbatch --job-name="${JOB_NAME}" cluster/slurm/train_ditto_cleaned_method.sbatch
    fi
  done
done
