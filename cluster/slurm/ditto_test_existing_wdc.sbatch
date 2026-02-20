#!/bin/bash
#SBATCH --job-name=ditto-wdc-test
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

# DWS cluster script for testing Ditto on existing WDC train/val/test splits.
# Submit from login node:
#   sbatch cluster/slurm/ditto_test_existing_wdc.sbatch
#
# Optional overrides before sbatch:
#   export SPLIT_SIZE=small|medium|large
#   export TRAIN_JSON_GZ=data/wdc/...
#   export VAL_JSON_GZ=data/wdc/...
#   export TEST_JSON_GZ=data/wdc/wdcproducts80cc20rnd100un_gs.json.gz
#   export OUTPUT_DIR=output/ditto_runs
#   export CONFIG=configs/ditto/default_train.yaml
#   export MODEL_NAME=roberta-base
#   export CONDA_ENV_NAME=ditto-modern

mkdir -p logs

# Robust conda activation without sourcing interactive bashrc under `set -u`
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-ditto-modern}"
else
  echo "conda not found in PATH; aborting" >&2
  exit 1
fi

# Ensure we're at repo root when launched from arbitrary cwd
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

bash scripts/ditto/run_existing_wdc.sh
