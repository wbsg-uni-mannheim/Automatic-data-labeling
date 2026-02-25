#!/bin/bash
#SBATCH --job-name=ditto-benchmarks
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
#   sbatch cluster/slurm/ditto_train_benchmarks.sbatch
#
# Optional overrides before sbatch:
#   export CONDA_ENV_NAME=ditto-modern
#   export CONFIG=configs/ditto/benchmarks_training.yaml
#   export OUTPUT_ROOT=output/ditto_benchmark_runs
#   export RUN_NAME=baseline_$(date +%Y%m%d_%H%M%S)
#   export BENCHMARKS="wdc,dblp-acm,dblp-scholar"   # comma-separated
#   export DRY_RUN=0                                 # set to 1 for dry run
#   export EXTRA_ARGS=""                             # additional args forwarded to runner

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

CONFIG="${CONFIG:-configs/ditto/benchmarks_training.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/ditto_benchmark_runs}"
RUN_NAME="${RUN_NAME:-slurm_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
BENCHMARKS="${BENCHMARKS:-}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -f "${CONFIG}" ]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

CMD=(
  python scripts/ditto/run_benchmark_training.py
  --config "${CONFIG}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
)

if [ -n "${BENCHMARKS}" ]; then
  # Normalize spaces to commas for the runner argument.
  BENCHMARKS="${BENCHMARKS// /,}"
  CMD+=(--benchmarks "${BENCHMARKS}")
fi

if [ "${DRY_RUN}" = "1" ]; then
  CMD+=(--dry-run)
fi

echo "Running Ditto benchmark training:"
echo "  CONFIG=${CONFIG}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  BENCHMARKS=${BENCHMARKS:-<all>}"
echo "  DRY_RUN=${DRY_RUN}"
if [ -n "${EXTRA_ARGS}" ]; then
  echo "  EXTRA_ARGS=${EXTRA_ARGS}"
fi

# shellcheck disable=SC2206
EXTRA_ARR=(${EXTRA_ARGS})
"${CMD[@]}" "${EXTRA_ARR[@]}"
