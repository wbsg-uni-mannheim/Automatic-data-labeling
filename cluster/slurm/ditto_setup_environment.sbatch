#!/bin/bash
#SBATCH --job-name=ditto-env-setup
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Submit from repo root:
#   sbatch cluster/slurm/ditto_setup_environment.sbatch
#
# Optional overrides:
#   export CONDA_ENV_NAME=ditto-modern
#   export ENV_FILE=configs/ditto/environment.yml

mkdir -p logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

ENV_FILE="${ENV_FILE:-configs/ditto/environment.yml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ditto-modern}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "conda not found in PATH; aborting" >&2
  exit 1
fi

SOLVER="conda"
if command -v mamba >/dev/null 2>&1; then
  SOLVER="mamba"
fi

if conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  echo "Updating existing env '${CONDA_ENV_NAME}' with ${SOLVER}..."
  "${SOLVER}" env update -n "${CONDA_ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "Creating env '${CONDA_ENV_NAME}' with ${SOLVER}..."
  "${SOLVER}" env create -n "${CONDA_ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${CONDA_ENV_NAME}"

python -V
python - <<'PY'
import importlib

modules = [
    "torch",
    "transformers",
    "openai",
    "dotenv",
    "numpy",
    "pandas",
    "sklearn",
    "yaml",
    "tqdm",
    "faiss",
    "xgboost",
    "sentence_transformers",
]

failed = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append((name, str(exc)))

if failed:
    print("Import check failed:")
    for name, err in failed:
        print(f"  - {name}: {err}")
    raise SystemExit(1)

print("Environment import check passed.")
PY

echo "Environment ready: ${CONDA_ENV_NAME}"
