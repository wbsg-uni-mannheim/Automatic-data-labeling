#!/bin/bash
#SBATCH --job-name=ditto-abtbuy-v2
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Submit from repo root:
#   sbatch cluster/slurm/train_ditto_abt_buy_three_phase_v2_20260323_202820.sbatch
#
# Optional overrides before sbatch:
#   export CONDA_ENV_NAME=ditto-modern
#   export LABEL_RUN_ROOT=output/three_phase_labeling_ditto_only_v2
#   export RUN_GLOB=benchmark_abt-buy_20260323_202820
#   export PROFILE_FILTER=__all__
#   export TRAIN_CONFIG=configs/ditto/benchmarks_training.yaml
#   export DITTO_OUTPUT_ROOT=output/training_from_generated_labels/three_phase_labeling_ditto_only_v2
#   export RUN_NAME_PREFIX=generated_ditto

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

LABEL_RUN_ROOT="${LABEL_RUN_ROOT:-output/three_phase_labeling_ditto_only_v2}"
RUN_GLOB="${RUN_GLOB:-benchmark_abt-buy_20260323_202820}"
PROFILE_FILTER="${PROFILE_FILTER:-__all__}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/ditto/benchmarks_training.yaml}"
DITTO_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT:-output/training_from_generated_labels/three_phase_labeling_ditto_only_v2}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-generated_ditto}"

if [ ! -d "${LABEL_RUN_ROOT}" ]; then
  echo "Label run root not found: ${LABEL_RUN_ROOT}" >&2
  exit 1
fi

RUN_DIR="${LABEL_RUN_ROOT}/${RUN_GLOB}"
if [ ! -d "${RUN_DIR}" ]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

echo "Training Ditto from generated labels:"
echo "  LABEL_RUN_ROOT=${LABEL_RUN_ROOT}"
echo "  RUN_GLOB=${RUN_GLOB}"
echo "  PROFILE_FILTER=${PROFILE_FILTER}"
echo "  TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "  DITTO_OUTPUT_ROOT=${DITTO_OUTPUT_ROOT}"
echo "  RUN_NAME_PREFIX=${RUN_NAME_PREFIX}"
echo

export CONDA_ENV_NAME
export LABEL_RUN_ROOT
export RUN_GLOB
export PROFILE_FILTER
export TRAIN_CONFIG
export DITTO_OUTPUT_ROOT
export RUN_NAME_PREFIX

bash cluster/slurm/train_ditto_from_generated_labels.sbatch

SUMMARY_OUT="${DITTO_OUTPUT_ROOT}/${RUN_NAME_PREFIX}_${RUN_GLOB}_profile_f1_summary.tsv"

python - "${DITTO_OUTPUT_ROOT}" "${RUN_NAME_PREFIX}" "${RUN_GLOB}" "${SUMMARY_OUT}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

output_root = Path(sys.argv[1])
run_name_prefix = sys.argv[2]
run_glob = sys.argv[3]
summary_out = Path(sys.argv[4])

rows = []
pattern = f"{run_name_prefix}_{run_glob}_*_*"
for run_dir in sorted(output_root.glob(pattern)):
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.exists():
        continue
    df = pd.read_csv(summary_csv)
    if df.empty:
        continue
    row = df.iloc[0].to_dict()
    stem = run_dir.name
    suffix = stem.removeprefix(f"{run_name_prefix}_{run_glob}_")
    profile = suffix.rsplit("_", 1)[0] if "_" in suffix else suffix
    rows.append(
        {
            "dataset": row.get("benchmark"),
            "profile": profile,
            "status": row.get("status"),
            "test_f1": row.get("test_f1"),
            "test_precision": row.get("test_precision"),
            "test_recall": row.get("test_recall"),
            "test_accuracy": row.get("test_accuracy"),
            "summary_csv": str(summary_csv),
            "run_dir": str(run_dir),
        }
    )

summary_out.parent.mkdir(parents=True, exist_ok=True)
if rows:
    out_df = pd.DataFrame(rows).sort_values(["dataset", "profile"]).reset_index(drop=True)
    out_df.to_csv(summary_out, sep="\t", index=False)
    print(f"Wrote profile summary: {summary_out}")
    print(out_df.to_string(index=False))
else:
    empty = pd.DataFrame(
        columns=[
            "dataset",
            "profile",
            "status",
            "test_f1",
            "test_precision",
            "test_recall",
            "test_accuracy",
            "summary_csv",
            "run_dir",
        ]
    )
    empty.to_csv(summary_out, sep="\t", index=False)
    print(f"No summary.csv files found for pattern {pattern}")
    print(f"Wrote empty summary: {summary_out}")
PY
