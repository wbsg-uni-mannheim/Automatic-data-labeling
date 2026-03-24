#!/bin/bash
#SBATCH --job-name=ditto-generated
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

# Train Ditto models from generated label runs and evaluate on the configured test sets.
#
# Submit from repo root:
#   sbatch cluster/slurm/train_ditto_from_generated_labels.sbatch
#
# Optional overrides before sbatch:
#   export CONDA_ENV_NAME=ditto-modern
#   export LABEL_RUN_ROOT=output/three_phase_labeling_ditto_only_v2
#   export RUN_GLOB="benchmark_*"
#   export PROFILES=small,medium
#   export TRAIN_CONFIG=configs/ditto/benchmarks_training.yaml
#   export DITTO_OUTPUT_ROOT=output/ditto_generated_labels
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
RUN_GLOB="${RUN_GLOB:-benchmark_*}"
PROFILES="${PROFILES:-small,medium}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/ditto/benchmarks_training.yaml}"
DITTO_OUTPUT_ROOT="${DITTO_OUTPUT_ROOT:-output/ditto_generated_labels}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-generated_ditto}"

if [ ! -d "${LABEL_RUN_ROOT}" ]; then
  echo "Label run root not found: ${LABEL_RUN_ROOT}" >&2
  exit 1
fi

if [ ! -f "${TRAIN_CONFIG}" ]; then
  echo "Ditto training config not found: ${TRAIN_CONFIG}" >&2
  exit 1
fi

echo "Label run root:   ${LABEL_RUN_ROOT}"
echo "Run glob:         ${RUN_GLOB}"
echo "Profiles:         ${PROFILES}"
echo "Train config:     ${TRAIN_CONFIG}"
echo "Ditto output:     ${DITTO_OUTPUT_ROOT}"
echo "Run name prefix:  ${RUN_NAME_PREFIX}"
echo

shopt -s nullglob
RUN_DIRS=( "${LABEL_RUN_ROOT}"/${RUN_GLOB} )
shopt -u nullglob

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
  echo "No run directories matched ${LABEL_RUN_ROOT}/${RUN_GLOB}" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
TS="$(date +%Y%m%d_%H%M%S)"

for RUN_DIR in "${RUN_DIRS[@]}"; do
  MANIFEST="${RUN_DIR}/profile_manifest.json"
  if [ ! -f "${MANIFEST}" ]; then
    echo "Skipping ${RUN_DIR}: missing profile_manifest.json" >&2
    continue
  fi

  while IFS=$'\t' read -r BENCHMARK PROFILE TMP_CFG; do
    RUN_STEM="$(basename "${RUN_DIR}")"
    TRAIN_RUN_NAME="${RUN_NAME_PREFIX}_${RUN_STEM}_${PROFILE}_${TS}"
    echo
    echo "=== Ditto training: benchmark=${BENCHMARK} profile=${PROFILE} run=${RUN_STEM} ==="
    python scripts/ditto/run_benchmark_training.py \
      --config "${TMP_CFG}" \
      --benchmarks "${BENCHMARK}" \
      --output-root "${DITTO_OUTPUT_ROOT}" \
      --run-name "${TRAIN_RUN_NAME}"
  done < <(
    python - "${MANIFEST}" "${TRAIN_CONFIG}" "${PROFILES}" "${TMP_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

manifest_path = Path(sys.argv[1])
train_cfg_path = Path(sys.argv[2])
profiles_raw = sys.argv[3]
tmp_dir = Path(sys.argv[4])

manifest = json.loads(manifest_path.read_text())
base = yaml.safe_load(train_cfg_path.read_text()) or {}
defaults = dict(base.get("defaults") or {})
benchmarks = dict(base.get("benchmarks") or {})

benchmark = str(manifest.get("benchmark", "")).strip()
if benchmark not in benchmarks:
    raise SystemExit(f"Benchmark {benchmark!r} not found in {train_cfg_path}")

profiles = [p.strip() for p in profiles_raw.replace(",", " ").split() if p.strip()]
if not profiles:
    raise SystemExit("No profiles requested")

manifest_profiles = manifest.get("profiles") or {}
for profile in profiles:
    if profile not in manifest_profiles:
        raise SystemExit(f"Profile {profile!r} not found in {manifest_path}")
    meta = manifest_profiles[profile] or {}
    train_json = str(meta.get("ditto_train_json_gz", "")).strip()
    if not train_json:
        raise SystemExit(
            f"Profile {profile!r} in {manifest_path} is missing ditto_train_json_gz. "
            "Re-export labels with Ditto export enabled."
        )

    bench_cfg = dict(benchmarks[benchmark])
    bench_cfg["train"] = train_json
    if meta.get("ditto_fields"):
        bench_cfg["fields"] = list(meta["ditto_fields"])

    cfg = {
        "defaults": defaults,
        "benchmarks": {
            benchmark: bench_cfg,
        },
    }
    cfg["defaults"]["early_stopping_patience"] = 10

    out_path = tmp_dir / f"{manifest_path.parent.name}__{profile}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{benchmark}\t{profile}\t{out_path}")
PY
  )
done

echo
echo "Finished Ditto training from generated label runs."
