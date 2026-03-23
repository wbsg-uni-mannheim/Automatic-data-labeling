#!/bin/bash
#SBATCH --job-name=active-label
#SBATCH --partition=gpu-vram-12gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Submit from repo root:
#   sbatch cluster/slurm/run_active_labeling_all.sbatch
#
# Optional overrides before sbatch:
#   export CONDA_ENV_NAME=ditto-modern
#   export CONFIG=configs/labeling/benchmarks_active.yaml
#   export BENCHMARKS="wdc abt-buy amazon-google dblp-acm dblp-scholar walmart-amazon"
#   export OUTPUT_ROOT=output/simple_labeling
#   export PROFILES=""                 # e.g. "small,medium,large"
#   export RUN_NAME_PREFIX=benchmark   # run name becomes <prefix>_<benchmark>_<timestamp>
#   export RESUME=0                    # set to 1 to pass --resume

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
BENCHMARKS_RAW="${BENCHMARKS:-amazon-google dblp-acm dblp-scholar walmart-amazon}"
BENCHMARKS_RAW="${BENCHMARKS_RAW//,/ }"
read -r -a BENCHMARK_LIST <<< "${BENCHMARKS_RAW}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/simple_labeling}"
PROFILES="${PROFILES:-small,medium,large}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-benchmark}"
RESUME="${RESUME:-0}"

if [ ! -f "${CONFIG}" ]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

if [ "${#BENCHMARK_LIST[@]}" -eq 0 ]; then
  echo "No benchmarks provided." >&2
  exit 1
fi

echo "Config:      ${CONFIG}"
echo "Benchmarks:  ${BENCHMARKS_RAW}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Profiles:    ${PROFILES:-<all>}"
echo "Resume:      ${RESUME}"

TS="$(date +%Y%m%d_%H%M%S)"

for BENCH in "${BENCHMARK_LIST[@]}"; do
  RUN_NAME="${RUN_NAME_PREFIX}_${BENCH}_${TS}"
  CMD=(
    python scripts/labeling/run_benchmark_labeling.py
    --config "${CONFIG}"
    --benchmark "${BENCH}"
    --output-root "${OUTPUT_ROOT}"
    --run-name "${RUN_NAME}"
  )

  if [ -n "${PROFILES}" ]; then
    CMD+=(--profiles "${PROFILES}")
  fi
  if [ "${RESUME}" = "1" ]; then
    CMD+=(--resume)
  fi

  echo
  echo "=== Running benchmark: ${BENCH} (run_name=${RUN_NAME}) ==="
  "${CMD[@]}"
done

echo
echo "All benchmark labeling runs completed."
