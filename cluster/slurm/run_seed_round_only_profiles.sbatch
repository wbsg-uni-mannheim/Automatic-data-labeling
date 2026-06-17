#!/bin/bash
#SBATCH --job-name=seed-only-profiles
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

# Run from repo root with, for example:
#   sbatch cluster/slurm/run_seed_round_only_profiles.sbatch
#
# Optional overrides:
#   export CONDA_ENV_NAME=ditto-modern
#   export BENCHMARKS="abt-buy,amazon-google,dblp-acm,dblp-scholar,walmart-amazon,wdc"
#   export PROFILES="small,medium,large,all"
#   export OUTPUT_ROOT=output/seed_round_only_profiles
#   export CONFIG=configs/labeling/benchmarks_active.yaml
#   export MODEL=gpt-5.2
#   export SEED_MAX_CALLS=0
#   export NO_EXPORT_DITTO=0
#   export SKIP_RANDOM_PROFILES=0

mkdir -p logs

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-ditto-modern}"
fi

CONFIG="${CONFIG:-configs/labeling/benchmarks_active.yaml}"
BENCHMARKS="${BENCHMARKS:-}"
PROFILES="${PROFILES:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/seed_round_only_profiles}"
MODEL="${MODEL:-}"
SEED_MAX_CALLS="${SEED_MAX_CALLS:-0}"
NO_EXPORT_DITTO="${NO_EXPORT_DITTO:-0}"
SKIP_RANDOM_PROFILES="${SKIP_RANDOM_PROFILES:-0}"
INCLUDE_RANDOM_PROFILES="${INCLUDE_RANDOM_PROFILES:-1}"

CMD=(
  python3 scripts/labeling/similarity_search.py
  --config "${CONFIG}"
  --output-root "${OUTPUT_ROOT}"
)

if [ -n "${BENCHMARKS}" ]; then
  CMD+=(--benchmarks "${BENCHMARKS}")
fi
if [ -n "${PROFILES}" ]; then
  CMD+=(--profiles "${PROFILES}")
fi
if [ -n "${MODEL}" ]; then
  CMD+=(--model "${MODEL}")
fi
if [ "${SEED_MAX_CALLS}" != "0" ]; then
  CMD+=(--seed-max-calls "${SEED_MAX_CALLS}")
fi
if [ "${NO_EXPORT_DITTO}" = "1" ]; then
  CMD+=(--no-export-ditto)
fi
if [ "${SKIP_RANDOM_PROFILES}" = "1" ]; then
  CMD+=(--skip-random-profiles)
fi
if [ "${INCLUDE_RANDOM_PROFILES}" = "1" ]; then
  CMD+=(--include-random-profiles)
fi

echo "Command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
