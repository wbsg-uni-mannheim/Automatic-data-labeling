#!/bin/bash

set -euo pipefail

# Submit one CPU seed-round-only profile job per benchmark.
#
# Run from repo root:
#   bash cluster/slurm/submit_seed_round_only_profiles_parallel.sh
#
# Optional overrides:
#   export CONDA_ENV_NAME=ditto-modern
#   export BENCHMARKS="abt-buy,amazon-google,dblp-acm,dblp-scholar,walmart-amazon,wdc"
#   export PROFILES="small,medium,large,all"
#   export OUTPUT_ROOT=output/seed_round_only_profiles
#   export CONFIG=configs/labeling/benchmarks_active.yaml
#   export MODEL=gpt-5.2
#   export SEED_MAX_CALLS=0
#   export SKIP_RANDOM_PROFILES=0
#   export NO_EXPORT_DITTO=0
#   export PARTITION=cpu
#   export CPUS_PER_TASK=4
#   export MEM=32G
#   export TIME_LIMIT=72:00:00

mkdir -p logs

BENCHMARKS_RAW="${BENCHMARKS:-abt-buy,amazon-google,dblp-acm,dblp-scholar,walmart-amazon,wdc}"
BENCHMARKS_RAW="${BENCHMARKS_RAW//,/ }"
read -r -a BENCHMARK_LIST <<< "${BENCHMARKS_RAW}"

if [ "${#BENCHMARK_LIST[@]}" -eq 0 ]; then
  echo "No benchmarks provided." >&2
  exit 1
fi

CONFIG="${CONFIG:-configs/labeling/benchmarks_active.yaml}"
PROFILES="${PROFILES:-small,medium,large,all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/seed_round_only_profiles}"
MODEL="${MODEL:-}"
SEED_MAX_CALLS="${SEED_MAX_CALLS:-0}"
SKIP_RANDOM_PROFILES="${SKIP_RANDOM_PROFILES:-0}"
NO_EXPORT_DITTO="${NO_EXPORT_DITTO:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ditto-modern}"

PARTITION="${PARTITION:-cpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-32G}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"

echo "Config:       ${CONFIG}"
echo "Benchmarks:   ${BENCHMARKS_RAW}"
echo "Profiles:     ${PROFILES}"
echo "Output root:  ${OUTPUT_ROOT}"
echo "Model:        ${MODEL:-<config default>}"
echo "Max calls:    ${SEED_MAX_CALLS}"
echo "Random:       $([ "${SKIP_RANDOM_PROFILES}" = "1" ] && echo disabled || echo enabled)"
echo "Export Ditto: $([ "${NO_EXPORT_DITTO}" = "1" ] && echo disabled || echo enabled)"
echo "Partition:    ${PARTITION}"
echo "CPUs/task:    ${CPUS_PER_TASK}"
echo "Mem:          ${MEM}"
echo "Time:         ${TIME_LIMIT}"
echo

for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
  SAFE_BENCHMARK="${BENCHMARK//[^a-zA-Z0-9]/-}"
  JOB_ID="$(
    sbatch \
      --parsable \
      --job-name="seed-only-${SAFE_BENCHMARK}" \
      --partition="${PARTITION}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --mem="${MEM}" \
      --time="${TIME_LIMIT}" \
      --export=ALL,CONDA_ENV_NAME="${CONDA_ENV_NAME}",BENCHMARKS="${BENCHMARK}",PROFILES="${PROFILES}",OUTPUT_ROOT="${OUTPUT_ROOT}",CONFIG="${CONFIG}",MODEL="${MODEL}",SEED_MAX_CALLS="${SEED_MAX_CALLS}",SKIP_RANDOM_PROFILES="${SKIP_RANDOM_PROFILES}",NO_EXPORT_DITTO="${NO_EXPORT_DITTO}",INCLUDE_RANDOM_PROFILES="1" \
      cluster/slurm/run_seed_round_only_profiles.sbatch
  )"
  echo "Submitted ${BENCHMARK}: job ${JOB_ID}"
done
