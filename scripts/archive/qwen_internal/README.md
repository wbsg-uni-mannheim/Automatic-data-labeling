# Qwen3 Students

Helper scripts for the Qwen3 students of the paper (Qwen3-0.6B, 1.7B, and 8B): LoRA fine-tuning on a training set in chat format, evaluation of the adapter, and zero-shot evaluation. Training itself runs through `scripts/training/train_qwen.py`.

## Environment

The Qwen3 runs use a separate Python 3.12 environment with Unsloth, TRL, and Transformers 5.5. On a CUDA machine:

```bash
bash scripts/archive/qwen_internal/setup_env.sh
source scripts/archive/qwen_internal/.venv/bin/activate
python scripts/archive/qwen_internal/check_env.py
```

The setup script installs CUDA PyTorch 2.10.0 first and then the package versions of `requirements.lock.txt`, the environment used for the paper.

## Scripts

| Script | What it does |
|---|---|
| `convert_wdc_to_sft.py` | Converts training, validation, and test pairs into chat examples that ask for a `Yes` or `No` answer. |
| `evaluate_lora.py` | Evaluates a trained adapter on the converted test file and writes `metrics.json` and `predictions.csv`. |
| `baseline_zero_shot_eval.py` | Evaluates a Qwen3 model without fine-tuning. |
| `check_env.py`, `setup_env.sh` | Create and check the environment. |

Both evaluators strip `<think>...</think>` blocks and parse the first clear `Yes` or `No`.

## Settings of the paper

LoRA rank 16 on all attention and MLP projections, learning rate 2e-4, warmup ratio 0.05, batch size 2 with 8 gradient-accumulation steps, maximum sequence length 2,048, attribute values cut at 350 characters, evaluation every 50 steps, and early stopping on the validation loss with patience 10. Qwen3-8B trains for up to 3 epochs, Qwen3-0.6B and 1.7B for up to 10. `artifacts/USAGE.md` shows the full command sequence.
