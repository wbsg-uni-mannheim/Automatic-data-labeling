#!/bin/bash
#SBATCH --job-name=ditto-cleaned-nv
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Train Ditto on cleaned-no-validation training sets for one (labeller, benchmark),
# repeated for three seeds.
#
# Required env:
#   export LABELLER=gpt|qwen|kimi
#   export BENCHMARK=abt-buy|dblp-acm|dblp-scholar|walmart-amazon|wdc
# Optional:
#   export SEEDS="42 52 62"
#   export OUTPUT_ROOT=output/ditto_cleaned_no_val_runs

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

LABELLER="${LABELLER:?LABELLER env var required (gpt|qwen|kimi)}"
BENCHMARK="${BENCHMARK:?BENCHMARK env var required}"
SEEDS_RAW="${SEEDS:-42 52 62}"
SEEDS_RAW="${SEEDS_RAW//,/ }"
read -r -a SEED_LIST <<< "${SEEDS_RAW}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/ditto_cleaned_no_val_runs}"
CONFIG="configs/ditto/benchmarks_training_cleaned_no_val_${LABELLER}.yaml"

if [ ! -f "${CONFIG}" ]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Labeller:    ${LABELLER}"
echo "Benchmark:   ${BENCHMARK}"
echo "Seeds:       ${SEEDS_RAW}"
echo "Config:      ${CONFIG}"
echo "Output root: ${OUTPUT_ROOT}"
echo

REPEAT=0
for SEED in "${SEED_LIST[@]}"; do
  REPEAT=$((REPEAT + 1))
  SEEDED_CFG="${TMP_DIR}/${LABELLER}_${BENCHMARK}_seed${SEED}.yaml"
  python - "${CONFIG}" "${SEED}" "${SEEDED_CFG}" <<'PY'
import sys, yaml
src, seed, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(open(src)) or {}
cfg.setdefault("defaults", {})["seed"] = seed
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY

  RUN_NAME="cleaned_nv_${LABELLER}_${BENCHMARK}_r${REPEAT}_seed${SEED}_${TS}"
  echo "=== run ${REPEAT}/${#SEED_LIST[@]} | labeller=${LABELLER} benchmark=${BENCHMARK} seed=${SEED} ==="
  python scripts/ditto/run_benchmark_training.py \
    --config "${SEEDED_CFG}" \
    --benchmarks "${BENCHMARK}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}"
  echo
done

echo "Finished cleaned-no-val Ditto training: labeller=${LABELLER} benchmark=${BENCHMARK}"
