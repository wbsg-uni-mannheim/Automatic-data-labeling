#!/bin/bash
#SBATCH --job-name=active-label-qwen36
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Generic Qwen 3.6 Plus (OpenRouter) active-labeling sbatch.
# Three-phase labeling (seed -> LLM-only AL -> Ditto-driven AL) for any benchmark
# defined in configs/labeling/benchmarks_active.yaml.
#
# Submit from repo root (override env vars per submission):
#   BENCHMARK=wdc \
#   PROFILES=small,medium,large,all \
#   OUTPUT_ROOT=output/ditto_labeling/qwen \
#   RUN_NAME=benchmark_wdc_$(date +%Y%m%d_%H%M%S) \
#   sbatch cluster/slurm/run_three_phase_labeling_qwen36plus.sbatch
#
# Optional overrides:
#   export CONDA_ENV_NAME=ditto-modern
#   export CONFIG=configs/labeling/benchmarks_active.yaml
#   export RESUME=0
#   export RUN_NAME_PREFIX=benchmark   # used only when RUN_NAME unset
#
# RUN_NAME is preserved verbatim — set it to an existing run name to resume.
# RESUME=1 forwards --resume to the runner so checkpoints are picked up.

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
BENCHMARK="${BENCHMARK:?BENCHMARK env var required (e.g. wdc, dblp-acm, dblp-scholar, walmart-amazon)}"
PROFILES="${PROFILES:-small,medium,large,all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/ditto_labeling/qwen}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-benchmark}"
RESUME="${RESUME:-0}"

if [ ! -f "${CONFIG}" ]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -z "${OPEN_ROUTER_API_KEY:-}" ]; then
  echo "OPEN_ROUTER_API_KEY not set (expected in .env or environment); aborting" >&2
  exit 1
fi

if [ -z "${RUN_NAME:-}" ]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_NAME="${RUN_NAME_PREFIX}_${BENCHMARK}_${TS}"
fi

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
echo "Provider:    OpenRouter (qwen/qwen3.6-plus)"
echo

"${CMD[@]}"
