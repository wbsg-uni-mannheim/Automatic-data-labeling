#!/bin/bash
#SBATCH --job-name=ditto-train-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Required envs before sbatch:
# export TRAIN_JSON_GZ=...
# export VAL_JSON_GZ=...
# Optional:
# export TEST_JSON_GZ=...
# export OUTPUT_DIR=output/ditto_runs
# export CONFIG=configs/ditto/default_train.yaml
# export CONDA_ENV_NAME=ditto-modern
# export NPROC_PER_NODE=2

module purge || true
module load cuda/12.1 || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-ditto-modern}"
fi

mkdir -p logs

export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
bash scripts/archive/ditto_internal/run_ddp.sh
