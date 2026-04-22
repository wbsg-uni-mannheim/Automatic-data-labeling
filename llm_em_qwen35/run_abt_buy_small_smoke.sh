#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${OUT_ROOT:-output/qwen35_em/abt-buy_small}"
DATA_DIR="${OUT_ROOT}/sft_data"
MODEL_DIR="${OUT_ROOT}/qwen35_9b_lora"
EVAL_DIR="${OUT_ROOT}/eval"

python llm_em_qwen35/convert_wdc_to_sft.py \
  --train-json-gz analysis/three_phase_relabel_benchmark_abt-buy_20260323_202820/small/active_labels_latest_abt-buy_small_train.json.gz \
  --valid-json-gz output/baseline/abt-buy/splits/valid.json.gz \
  --test-json-gz output/baseline/abt-buy/splits/test.json.gz \
  --fields title,description,price \
  --output-dir "${DATA_DIR}"

python llm_em_qwen35/train_unsloth_lora.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${MODEL_DIR}" \
  --model-name Qwen/Qwen3.5-9B \
  --max-seq-length 2048 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --lora-r 16

python llm_em_qwen35/evaluate_lora.py \
  --data-path "${DATA_DIR}/test.jsonl" \
  --model-path "${MODEL_DIR}/final_adapter" \
  --output-dir "${EVAL_DIR}"

