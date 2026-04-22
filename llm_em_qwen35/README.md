# Qwen3.5 Entity Matching Fine-Tuning

This directory is isolated from the Ditto environment. It fine-tunes `Qwen/Qwen3.5-9B`
on existing entity-matching splits exported in the repo's WDC/Ditto-style JSONL gzip
format.

## Model And Tooling Choices

- Model: `Qwen/Qwen3.5-9B`
- License: Apache-2.0
- Training method: bf16 LoRA, no quantization
- Default task format: direct `Yes` / `No` chat SFT
- Primary training stack: Unsloth + TRL + pinned Transformers main commit

The Qwen model card says Qwen3.5 needs current serving/runtime support and recommends
latest framework versions. The Unsloth Qwen3.5 guide says to use Transformers v5,
estimates bf16 LoRA VRAM for Qwen3.5-9B at about 22GB, and does not recommend 4-bit
QLoRA for Qwen3.5. With 48GB VRAM, this setup intentionally uses bf16 LoRA.

Pinned volatile upstream refs captured on 2026-04-22:

- `transformers` main: `77c0e6e7ce5537558f07bb9147d0df4a0132e6f3`
- `unsloth` PyPI: `2026.4.6`
- `unsloth_zoo` PyPI: `2026.4.8`

Official references:

- Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-9B
- Unsloth Qwen3.5 guide: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Unsloth install guide: https://unsloth.ai/docs/get-started/install-and-update

## Create The Separate Environment

Run this on the CUDA machine, not on a CPU-only laptop:

```bash
cd /Users/aaronsteiner/Documents/GitHub/Automatic-data-labeling.
bash llm_em_qwen35/setup_env.sh
source llm_em_qwen35/.venv/bin/activate
python llm_em_qwen35/check_env.py
```

The setup script installs CUDA PyTorch first, then the pinned LLM stack from
`requirements.txt`.

## Prepare A Dataset

Example using the relabeled ABT-Buy small split:

```bash
source llm_em_qwen35/.venv/bin/activate
python llm_em_qwen35/convert_wdc_to_sft.py \
  --train-json-gz analysis/three_phase_relabel_benchmark_abt-buy_20260323_202820/small/active_labels_latest_abt-buy_small_train.json.gz \
  --valid-json-gz output/baseline/abt-buy/splits/valid.json.gz \
  --test-json-gz output/baseline/abt-buy/splits/test.json.gz \
  --fields title,description,price \
  --output-dir output/qwen35_em/abt-buy_small/sft_data
```

For other benchmarks, use the same fields as `configs/ditto/benchmarks_training.yaml`.

## Train

```bash
python llm_em_qwen35/train_unsloth_lora.py \
  --data-dir output/qwen35_em/abt-buy_small/sft_data \
  --output-dir output/qwen35_em/abt-buy_small/qwen35_9b_lora \
  --model-name Qwen/Qwen3.5-9B \
  --max-seq-length 2048 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --lora-r 16
```

Start with rank 16 and sequence length 2048. With 48GB VRAM, batch size 2 should be
reasonable; increase only after the first smoke run.

## Evaluate

```bash
python llm_em_qwen35/evaluate_lora.py \
  --data-path output/qwen35_em/abt-buy_small/sft_data/test.jsonl \
  --model-path output/qwen35_em/abt-buy_small/qwen35_9b_lora/final_adapter \
  --output-dir output/qwen35_em/abt-buy_small/eval
```

The evaluator strips `<think>...</think>` blocks if the model emits them and parses
the first clear `Yes` or `No`.

