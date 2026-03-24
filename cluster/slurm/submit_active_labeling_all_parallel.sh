#!/bin/bash

set -euo pipefail

# Submit one Slurm job per benchmark in parallel.
#
# Run from repo root:
#   bash cluster/slurm/submit_active_labeling_all_parallel.sh
#
# Optional overrides before launch:
#   export CONDA_ENV_NAME=ditto-modern
#   export CONFIG=configs/labeling/benchmarks_active.yaml
#   export BENCHMARKS="wdc abt-buy amazon-google dblp-acm dblp-scholar walmart-amazon"
#   export OUTPUT_ROOT=output/three_phase_labeling_ditto_only_v2
#   export PROFILES=small,medium
#   export RUN_NAME_PREFIX=benchmark
#   export RESUME=1
#   export PARTITION=gpu-vram-48gb
#   export CPUS_PER_TASK=4
#   export MEM=32G
#   export TIME_LIMIT=48:00:00
#   export GPU_COUNT=1

mkdir -p logs

REPO_ROOT="${PWD}"
CONFIG="${CONFIG:-configs/labeling/benchmarks_active.yaml}"
BENCHMARKS_RAW="${BENCHMARKS:-wdc abt-buy amazon-google dblp-acm dblp-scholar walmart-amazon}"
BENCHMARKS_RAW="${BENCHMARKS_RAW//,/ }"
read -r -a BENCHMARK_LIST <<< "${BENCHMARKS_RAW}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/three_phase_labeling_ditto_only_v2}"
PROFILES="${PROFILES:-small,medium}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-benchmark}"
RESUME="${RESUME:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ditto-modern}"

PARTITION="${PARTITION:-gpu-vram-48gb}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-32G}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
GPU_COUNT="${GPU_COUNT:-1}"

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
echo "Partition:   ${PARTITION}"
echo "CPUs/task:   ${CPUS_PER_TASK}"
echo "Mem:         ${MEM}"
echo "Time:        ${TIME_LIMIT}"
echo "GPUs:        ${GPU_COUNT}"
echo

for BENCH in "${BENCHMARK_LIST[@]}"; do
  EXISTING_RUN=""
  if [ "${RESUME}" = "1" ] && [ -d "${OUTPUT_ROOT}" ]; then
    EXISTING_RUN="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "${RUN_NAME_PREFIX}_${BENCH}_*" | sort | tail -n 1 || true)"
  fi
  if [ -n "${EXISTING_RUN}" ]; then
    RUN_NAME="$(basename "${EXISTING_RUN}")"
  else
    RUN_NAME="${RUN_NAME_PREFIX}_${BENCH}_$(date +%Y%m%d_%H%M%S)"
  fi
  SAFE_BENCH="${BENCH//[^a-zA-Z0-9]/-}"

  JOB_ID="$(
    sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=active-${SAFE_BENCH}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --gres=gpu:${GPU_COUNT}
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs
cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "conda not found in PATH; aborting" >&2
  exit 1
fi

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

echo "=== Running benchmark: ${BENCH} (run_name=${RUN_NAME}) ==="
"\${CMD[@]}"
EOF
  )"

  if [ -n "${EXISTING_RUN}" ]; then
    echo "Submitted ${BENCH}: job ${JOB_ID} (resuming run_name=${RUN_NAME})"
  else
    echo "Submitted ${BENCH}: job ${JOB_ID} (new run_name=${RUN_NAME})"
  fi
done
