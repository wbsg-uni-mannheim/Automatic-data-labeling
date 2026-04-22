#!/bin/bash
#SBATCH --job-name=seed-only-resume
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

# Resume one existing seed-round-only run in-place.
#
# Run from repo root, for example:
#   export RUN_DIR=output/seed_round_only_profiles/benchmark_abt-buy_20260415_190530
#   sbatch cluster/slurm/continue_seed_round_only_run.sbatch
#
# Optional overrides:
#   export CONDA_ENV_NAME=ditto-modern
#   export CONFIG=configs/labeling/benchmarks_active.yaml
#   export PROFILES="small,medium,large,all"
#   export MODEL=gpt-5.2
#   export SEED_MAX_CALLS=0
#   export NO_EXPORT_DITTO=0
#   export SKIP_RANDOM_PROFILES=0

mkdir -p logs

if [ -z "${RUN_DIR:-}" ]; then
  echo "RUN_DIR is required, e.g. output/seed_round_only_profiles/benchmark_abt-buy_20260415_190530" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-ditto-modern}"
fi

CONFIG="${CONFIG:-configs/labeling/benchmarks_active.yaml}"
PROFILES="${PROFILES:-small,medium,large,all}"
MODEL="${MODEL:-}"
SEED_MAX_CALLS="${SEED_MAX_CALLS:-0}"
NO_EXPORT_DITTO="${NO_EXPORT_DITTO:-0}"
SKIP_RANDOM_PROFILES="${SKIP_RANDOM_PROFILES:-0}"

CMD=(
  python3 scripts/labeling/continue_seed_round_only_run.py
  "${RUN_DIR}"
  --config "${CONFIG}"
  --profiles "${PROFILES}"
)

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

echo "Run dir:      ${RUN_DIR}"
echo "Config:       ${CONFIG}"
echo "Profiles:     ${PROFILES}"
echo "Model:        ${MODEL:-<config default>}"
echo "Max calls:    ${SEED_MAX_CALLS}"
echo "Random:       $([ "${SKIP_RANDOM_PROFILES}" = "1" ] && echo disabled || echo enabled)"
echo "Export Ditto: $([ "${NO_EXPORT_DITTO}" = "1" ] && echo disabled || echo enabled)"
echo "Command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
