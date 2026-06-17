#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT}"
TEST_JSON_GZ="${TEST_JSON_GZ:-benchmarks/wdc/wdcproducts80cc20rnd100un_gs.json.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-output/ditto_runs}"

python scripts/archive/ditto_internal/evaluate.py \
  --checkpoint "${CHECKPOINT}" \
  --test-json-gz "${TEST_JSON_GZ}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
