#!/bin/bash
#SBATCH --job-name=ditto-scan-m
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Train Ditto on a cleaned abt-buy training set for a given METHOD + PROFILE.
# Required env:
#   export METHOD=seed_round|simple_active
#   export PROFILE=small|medium|large|all|*_plus20random
# Optional:
#   export SEEDS="42 52 62"

mkdir -p logs
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then cd "${SLURM_SUBMIT_DIR}"; fi
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-ditto-modern}"
else
  echo "conda not found" >&2; exit 1
fi

METHOD="${METHOD:?METHOD required}"
PROFILE="${PROFILE:?PROFILE required}"
SEEDS_RAW="${SEEDS:-42 52 62}"
SEEDS_RAW="${SEEDS_RAW//,/ }"
read -r -a SEED_LIST <<< "${SEEDS_RAW}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/ditto_size_scan_abtbuy_methods_runs}"
mkdir -p "${OUTPUT_ROOT}"
TS="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

TRAIN_GZ="output/labeling_cleaned_size_scan/${METHOD}/abt-buy/profiles/${PROFILE}/active_labels_latest_abt-buy_${PROFILE}_train.json.gz"
[ -f "${TRAIN_GZ}" ] || { echo "missing: ${TRAIN_GZ}" >&2; exit 1; }

cat > "${TMP_DIR}/cfg.yaml" <<EOF
defaults:
  output_root: ${OUTPUT_ROOT}
  active_profiles: [all]
  model_name: roberta-base
  batch_size: 64
  max_len: 256
  epochs: 50
  lr: 5.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  grad_accum_steps: 1
  early_stopping_patience: 10
  seed: 42
  num_workers: 2
  fp16: true
  da: del
  alpha_aug: 0.8
  summarize: false
  dk: null
  spacy_model: en_core_web_sm
  max_field_len: 350
  exclude_valid_from_train: true

benchmarks:
  abt-buy:
    train: ${TRAIN_GZ}
    valid: benchmarks/abt-buy/abt-buy-valid.csv
    valid_pair_id_only: true
    valid_lookup_train: benchmarks/abt-buy/abt-buy-train.json
    test: benchmarks/abt-buy/abt-buy-gs.json.gz
    fields: [title, description, price]
    field_aliases:
      title: [name]
EOF

REPEAT=0
for SEED in "${SEED_LIST[@]}"; do
  REPEAT=$((REPEAT + 1))
  SEEDED_CFG="${TMP_DIR}/cfg_seed${SEED}.yaml"
  python - "${TMP_DIR}/cfg.yaml" "${SEED}" "${SEEDED_CFG}" <<'PY'
import sys, yaml
src, seed, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(open(src)) or {}
cfg.setdefault("defaults", {})["seed"] = seed
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY
  RUN_NAME="scanm_${METHOD}_${PROFILE}_r${REPEAT}_seed${SEED}_${TS}"
  echo "=== ${REPEAT}/${#SEED_LIST[@]} | method=${METHOD} profile=${PROFILE} seed=${SEED} ==="
  python scripts/training/train_ditto.py \
    --config "${SEEDED_CFG}" \
    --benchmarks "abt-buy" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}"
done
echo "Done size-scan-methods: method=${METHOD} profile=${PROFILE}"
