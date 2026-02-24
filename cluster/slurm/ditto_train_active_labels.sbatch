#!/bin/bash
#SBATCH --job-name=ditto-active-train
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

# Submit from repo root (or any dir):
#   sbatch cluster/slurm/ditto_train_active_labels.sbatch
#
# Optional overrides before sbatch:
#   export TRAIN_JSON_GZ=output/simple_labeling/run_20260222_204534/active_labels_latest_wdc_train.json.gz
#   export VAL_JSON_GZ=data/wdc/wdcproducts80cc20rnd000un_valid_medium.json.gz
#   export TEST_JSON_GZ=data/wdc/wdcproducts80cc20rnd100un_gs.json.gz
#   export OUTPUT_DIR=output/ditto_runs
#   export DITTO_SEEDS="42 52 62"
#   export SWEEP_NAME=my_seed_sweep_001
#   export CONDA_ENV_NAME=ditto-modern

mkdir -p logs

# Robust conda activation without interactive shell assumptions.
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-ditto-modern}"
else
  echo "conda not found in PATH; aborting" >&2
  exit 1
fi

# Ensure we run from submit directory.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

TRAIN_JSON_GZ="${TRAIN_JSON_GZ:-data/wdc/wdcproducts80cc20rnd000un_train_small.json.gz}"
VAL_JSON_GZ="${VAL_JSON_GZ:-data/wdc/wdcproducts80cc20rnd000un_valid_medium.json.gz}"
TEST_JSON_GZ="${TEST_JSON_GZ:-data/wdc/wdcproducts80cc20rnd100un_gs.json.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-output/ditto_runs}"

# Use dedicated var to avoid accidental collisions with unrelated env vars like SEEDS.
SEEDS_RAW="${DITTO_SEEDS:-69 420 1337}"
SEEDS_RAW="${SEEDS_RAW//,/ }"
read -r -a SEED_LIST <<< "${SEEDS_RAW}"
if [ "${#SEED_LIST[@]}" -eq 0 ]; then
  echo "No seeds provided in DITTO_SEEDS" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
SWEEP_NAME="${SWEEP_NAME:-seed_sweep_$(date +%Y%m%d_%H%M%S)}"
SWEEP_RUN_DIR="${OUTPUT_DIR}/${SWEEP_NAME}"
mkdir -p "${SWEEP_RUN_DIR}"

TMP_METRICS_TSV="$(mktemp)"
echo -e "seed\trun_dir\tbest_val_f1\ttest_f1\ttest_precision\ttest_recall\ttest_accuracy" > "${TMP_METRICS_TSV}"
FAILED_SEEDS=""

echo "Sweep directory: ${SWEEP_RUN_DIR}"
echo "Train file: ${TRAIN_JSON_GZ}"
echo "Val file:   ${VAL_JSON_GZ}"
echo "Test file:  ${TEST_JSON_GZ}"
echo "Seeds:      ${SEEDS_RAW}"

for SEED in "${SEED_LIST[@]}"; do
  echo "=== Ditto training seed ${SEED} ==="
  SEED_OUTPUT_DIR="${SWEEP_RUN_DIR}/seed_${SEED}"
  mkdir -p "${SEED_OUTPUT_DIR}"
  if ! python scripts/ditto/train.py \
    --train-json-gz "${TRAIN_JSON_GZ}" \
    --val-json-gz "${VAL_JSON_GZ}" \
    --test-json-gz "${TEST_JSON_GZ}" \
    --output-dir "${SEED_OUTPUT_DIR}" \
    --seed "${SEED}"; then
    echo "Seed ${SEED} failed during training; continuing to next seed." >&2
    FAILED_SEEDS="${FAILED_SEEDS} ${SEED}"
    continue
  fi

  RUN_DIR="$(cat "${SEED_OUTPUT_DIR}/LATEST_RUN")"
  METRICS_JSON="${RUN_DIR}/metrics.json"

  # Small retry window for shared filesystem latency.
  for _ in $(seq 1 20); do
    [ -f "${METRICS_JSON}" ] && break
    sleep 1
  done

  if [ ! -f "${METRICS_JSON}" ]; then
    echo "Missing metrics file for seed ${SEED}: ${METRICS_JSON}; continuing." >&2
    FAILED_SEEDS="${FAILED_SEEDS} ${SEED}"
    continue
  fi

  python - "${SEED}" "${RUN_DIR}" "${METRICS_JSON}" >> "${TMP_METRICS_TSV}" <<'PY'
import json
import sys

seed = sys.argv[1]
run_dir = sys.argv[2]
metrics_path = sys.argv[3]
with open(metrics_path, "r") as f:
    m = json.load(f)
test = m.get("test", {}) if isinstance(m, dict) else {}
best_val_f1 = float(m.get("best_val_f1", float("nan")))
test_f1 = float(test.get("f1", float("nan")))
test_precision = float(test.get("precision", float("nan")))
test_recall = float(test.get("recall", float("nan")))
test_accuracy = float(test.get("accuracy", float("nan")))
print(f"{seed}\t{run_dir}\t{best_val_f1}\t{test_f1}\t{test_precision}\t{test_recall}\t{test_accuracy}")
PY
done

echo
echo "=== Per-seed metrics ==="
cat "${TMP_METRICS_TSV}"
if [ -n "${FAILED_SEEDS// /}" ]; then
  echo "Failed seeds:${FAILED_SEEDS}"
fi

# Require at least one successful run row beyond header.
if [ "$(wc -l < "${TMP_METRICS_TSV}")" -le 1 ]; then
  echo "No successful seeded runs with metrics were found." >&2
  exit 1
fi

PER_SEED_TSV="${SWEEP_RUN_DIR}/per_seed_metrics.tsv"
cp "${TMP_METRICS_TSV}" "${PER_SEED_TSV}"

SUMMARY_JSON="${SWEEP_RUN_DIR}/seed_sweep_summary.json"
python - "${TMP_METRICS_TSV}" "${SUMMARY_JSON}" "${SWEEP_RUN_DIR}" "${TRAIN_JSON_GZ}" "${VAL_JSON_GZ}" "${TEST_JSON_GZ}" "${SEEDS_RAW}" <<'PY'
import csv
import json
import math
import statistics
import sys

tsv_path = sys.argv[1]
summary_path = sys.argv[2]
sweep_run_dir = sys.argv[3]
train_json_gz = sys.argv[4]
val_json_gz = sys.argv[5]
test_json_gz = sys.argv[6]
seeds_raw = sys.argv[7]

rows = []
with open(tsv_path, "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

def parse_float(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def agg(name):
    vals = [parse_float(r.get(name, "")) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals), "n": len(vals)}

summary = {
    "sweep_run_dir": sweep_run_dir,
    "data": {
        "train_json_gz": train_json_gz,
        "val_json_gz": val_json_gz,
        "test_json_gz": test_json_gz,
    },
    "seeds_raw": seeds_raw,
    "num_runs": len(rows),
    "seeds": [r["seed"] for r in rows],
    "runs": rows,
    "aggregate": {
        "best_val_f1": agg("best_val_f1"),
        "test_f1": agg("test_f1"),
        "test_precision": agg("test_precision"),
        "test_recall": agg("test_recall"),
        "test_accuracy": agg("test_accuracy"),
    },
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Aggregate (mean ± std) ===")
for k, v in summary["aggregate"].items():
    mean = v["mean"]
    std = v["std"]
    n = v["n"]
    if mean is None:
        print(f"{k}: n=0")
    else:
        print(f"{k}: {mean:.6f} ± {std:.6f} (n={n})")
print(f"\nSaved summary: {summary_path}")
print(f"Per-seed table: {sweep_run_dir}/per_seed_metrics.tsv")
PY
