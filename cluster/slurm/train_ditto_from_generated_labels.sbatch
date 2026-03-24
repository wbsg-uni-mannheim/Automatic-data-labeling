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
#   export PROFILES=small,medium,small_plus20random,medium_plus20random,all,all_plus20random
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
PROFILES="${PROFILES:-small,medium,small_plus20random,medium_plus20random,all,all_plus20random}"
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

mkdir -p "${DITTO_OUTPUT_ROOT}"

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
    EXISTING_DONE="$(find "${DITTO_OUTPUT_ROOT}" -maxdepth 1 -type d -name "${RUN_NAME_PREFIX}_${RUN_STEM}_${PROFILE}_*" | sort | while read -r CANDIDATE; do
      if [ -f "${CANDIDATE}/${BENCHMARK}/metrics.json" ]; then
        echo "${CANDIDATE}"
      fi
    done | tail -n 1 || true)"
    if [ -n "${EXISTING_DONE}" ]; then
      echo
      echo "=== Ditto training: benchmark=${BENCHMARK} profile=${PROFILE} run=${RUN_STEM} ==="
      echo "Skipping completed training: ${EXISTING_DONE}"
      continue
    fi

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
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

manifest_path = Path(sys.argv[1])
train_cfg_path = Path(sys.argv[2])
profiles_raw = sys.argv[3]
tmp_dir = Path(sys.argv[4])

manifest = json.loads(manifest_path.read_text())
base = yaml.safe_load(train_cfg_path.read_text()) or {}
defaults = dict(base.get("defaults") or {})
benchmarks = dict(base.get("benchmarks") or {})
run_dir = manifest_path.parent

benchmark = str(manifest.get("benchmark", "")).strip()
if benchmark not in benchmarks:
    raise SystemExit(f"Benchmark {benchmark!r} not found in {train_cfg_path}")

profiles = [p.strip() for p in profiles_raw.replace(",", " ").split() if p.strip()]
if not profiles:
    raise SystemExit("No profiles requested")

manifest_profiles = manifest.get("profiles") or {}
for profile in profiles:
    meta = manifest_profiles.get(profile)
    if meta is None and profile.endswith("random") and "_plus" in profile:
        base_name, pct_suffix = profile.rsplit("_plus", 1)
        pct_text = pct_suffix.removesuffix("random")
        try:
            pct = max(1, int(pct_text))
        except Exception:
            pct = 20

        random_labels = run_dir / "random_profile_labels.csv"
        left_csv = Path(str(manifest.get("canonical_left_csv", "")).strip())
        right_csv = Path(str(manifest.get("canonical_right_csv", "")).strip())
        base_meta = manifest_profiles.get(base_name) or {}

        if base_name == "all":
            base_labels = run_dir / "active_labels_latest.csv"
        else:
            base_labels_raw = str(base_meta.get("labels_csv", "")).strip()
            base_labels = Path(base_labels_raw) if base_labels_raw else run_dir / "profiles" / base_name / "active_labels_latest.csv"

        if base_labels.exists() and random_labels.exists() and left_csv.exists() and right_csv.exists():
            profile_dir = run_dir / "profiles" / profile
            profile_dir.mkdir(parents=True, exist_ok=True)
            labels_csv = profile_dir / "active_labels_latest.csv"
            labels_final_csv = profile_dir / "labels_final.csv"
            train_json_out = profile_dir / f"active_labels_latest_{benchmark}_{profile}_train.json.gz"
            if not labels_csv.exists():
                base_df = pd.read_csv(base_labels).reset_index(drop=True)
                rand_df = pd.read_csv(random_labels).reset_index(drop=True)
                extra_n = max(1, round(len(base_df) * (pct / 100.0)))
                rand_df = rand_df.head(int(extra_n)).copy()
                merged = pd.concat([base_df, rand_df], ignore_index=True)
                pair_cols = [c for c in ["rid1", "rid2"] if c in merged.columns]
                if len(pair_cols) != 2:
                    pair_cols = ["id1", "id2"]
                merged = merged.drop_duplicates(subset=pair_cols, keep="first").reset_index(drop=True)
                merged.to_csv(labels_csv, index=False)
                merged.to_csv(labels_final_csv, index=False)
            if not train_json_out.exists():
                cmd = [
                    sys.executable,
                    "scripts/ditto/convert_active_labels_to_wdc.py",
                    "--labels-csv",
                    str(labels_csv),
                    "--left-csv",
                    str(left_csv),
                    "--right-csv",
                    str(right_csv),
                    "--output-json-gz",
                    str(train_json_out),
                ]
                subprocess.run(cmd, check=True)
            meta = {
                "ditto_train_json_gz": str(train_json_out),
                "ditto_fields": base_meta.get("ditto_fields") or manifest.get("ditto_fields") or [],
            }
    if meta is None:
        raise SystemExit(f"Profile {profile!r} not found in {manifest_path}")
    meta = meta or {}
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
