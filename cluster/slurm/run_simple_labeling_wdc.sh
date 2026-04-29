#!/bin/bash
#SBATCH --job-name=simple-label-wdc
#SBATCH --partition=gpu-vram-12gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Submit from repo root:
#   sbatch cluster/slurm/run_simple_labeling_wdc.sbatch
#
# Optional overrides before sbatch:
#   export CONDA_ENV_NAME=ditto-modern
#   export CONFIG=configs/labeling/benchmarks.yaml
#   export BENCHMARK=wdc
#   export PROFILES=small,medium,large,all
#   export OUTPUT_ROOT=output/simple_active_learning_labeling
#   export RUN_NAME=benchmark_wdc_manual
#   export RESUME=0
#   export NO_EXPORT_DITTO=1

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

CONFIG="${CONFIG:-configs/labeling/benchmarks.yaml}"
BENCHMARK="${BENCHMARK:-wdc}"
PROFILES="${PROFILES:-small,medium,large,all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/simple_active_learning_labeling}"
RUN_NAME="${RUN_NAME:-benchmark_wdc_$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-0}"
NO_EXPORT_DITTO="${NO_EXPORT_DITTO:-1}"

if [ ! -f "${CONFIG}" ]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

CMD=(
  python scripts/labeling/run_benchmark_labeling.py
  --config "${CONFIG}"
  --benchmark "${BENCHMARK}"
  --profiles "${PROFILES}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
)

if [ "${RESUME}" = "1" ]; then
  CMD+=(--resume)
fi
if [ "${NO_EXPORT_DITTO}" = "1" ]; then
  CMD+=(--no-export-ditto)
fi

echo "Running simple labeling benchmark job:"
echo "  CONFIG=${CONFIG}"
echo "  BENCHMARK=${BENCHMARK}"
echo "  PROFILES=${PROFILES}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  RESUME=${RESUME}"
echo "  NO_EXPORT_DITTO=${NO_EXPORT_DITTO}"
echo

"${CMD[@]}"
