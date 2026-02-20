#!/bin/bash
#SBATCH --job-name=ditto-train-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
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

module purge || true
module load cuda/12.1 || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-ditto-modern}"
fi

mkdir -p logs

bash scripts/ditto/run_single_gpu.sh
