#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_JSON_GZ:?Set TRAIN_JSON_GZ}"
: "${VAL_JSON_GZ:?Set VAL_JSON_GZ}"

TEST_JSON_GZ="${TEST_JSON_GZ:-data/wdc/wdcproducts80cc20rnd100un_gs.json.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-output/ditto_runs}"
CONFIG="${CONFIG:-configs/ditto/default_train.yaml}"

python scripts/ditto/train.py \
  --train-json-gz "${TRAIN_JSON_GZ}" \
  --val-json-gz "${VAL_JSON_GZ}" \
  --test-json-gz "${TEST_JSON_GZ}" \
  --output-dir "${OUTPUT_DIR}" \
  --config "${CONFIG}" \
  "$@"
