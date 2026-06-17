#!/bin/bash
#SBATCH --job-name=active-label
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Submit from repo root:
#   sbatch cluster/slurm/run_three_phase_labeling_walmart_amazon_medium.sbatch
#
# Optional overrides before sbatch:
#   export CONDA_ENV_NAME=ditto-modern
#   export CONFIG=configs/labeling/benchmarks_active.yaml
#   export BENCHMARK=walmart-amazon
#   export PROFILES=medium
#   export OUTPUT_ROOT=output/simple_labeling
#   export RUN_NAME_PREFIX=benchmark
#   export RESUME=0

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

CONFIG="${CONFIG:-configs/labeling/benchmarks_active.yaml}"
BENCHMARK="${BENCHMARK:-walmart-amazon}"
PROFILES="${PROFILES:-medium}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/three_phase_labeling_ditto_only}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-benchmark}"
RESUME="${RESUME:-0}"

if [ ! -f "${CONFIG}" ]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME_PREFIX}_${BENCHMARK}_${TS}"

CMD=(
  python scripts/archive/labeling_helpers/run_benchmark_labeling.py
  --config "${CONFIG}"
  --benchmark "${BENCHMARK}"
  --profiles "${PROFILES}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
)

if [ "${RESUME}" = "1" ]; then
  CMD+=(--resume)
fi

echo "Config:      ${CONFIG}"
echo "Benchmark:   ${BENCHMARK}"
echo "Profiles:    ${PROFILES}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run name:    ${RUN_NAME}"
echo "Resume:      ${RESUME}"
echo

"${CMD[@]}"
