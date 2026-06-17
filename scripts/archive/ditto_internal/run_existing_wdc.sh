#!/usr/bin/env bash
set -euo pipefail

# Train Ditto on existing WDC train/valid splits and evaluate on GS test set.
# Defaults are chosen for a first stable run; override via env vars.

WDC_DIR="${WDC_DIR:-data/wdc}"
SPLIT_SIZE="${SPLIT_SIZE:-medium}"   # small | medium | large

case "${SPLIT_SIZE}" in
  small|medium|large) ;;
  *)
    echo "Invalid SPLIT_SIZE='${SPLIT_SIZE}'. Use: small|medium|large" >&2
    exit 1
    ;;
esac

TRAIN_JSON_GZ="${TRAIN_JSON_GZ:-${WDC_DIR}/wdcproducts80cc20rnd000un_train_${SPLIT_SIZE}.json.gz}"
VAL_JSON_GZ="${VAL_JSON_GZ:-${WDC_DIR}/wdcproducts80cc20rnd000un_valid_${SPLIT_SIZE}.json.gz}"
TEST_JSON_GZ="${TEST_JSON_GZ:-${WDC_DIR}/wdcproducts80cc20rnd100un_gs.json.gz}"

OUTPUT_DIR="${OUTPUT_DIR:-output/ditto_runs}"
CONFIG="${CONFIG:-configs/ditto/default_train.yaml}"
MODEL_NAME="${MODEL_NAME:-roberta-base}"

echo "[Ditto] Training with:"
echo "  TRAIN_JSON_GZ=${TRAIN_JSON_GZ}"
echo "  VAL_JSON_GZ=${VAL_JSON_GZ}"
echo "  TEST_JSON_GZ=${TEST_JSON_GZ}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  CONFIG=${CONFIG}"
echo "  MODEL_NAME=${MODEL_NAME}"

python -u scripts/archive/ditto_internal/train.py \
  --train-json-gz "${TRAIN_JSON_GZ}" \
  --val-json-gz "${VAL_JSON_GZ}" \
  --test-json-gz "${TEST_JSON_GZ}" \
  --output-dir "${OUTPUT_DIR}" \
  --config "${CONFIG}" \
  --model-name "${MODEL_NAME}" \
  "$@"
