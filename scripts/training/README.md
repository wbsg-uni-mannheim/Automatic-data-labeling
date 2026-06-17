# Training Entry Points

This directory contains the public student-model training entry points.

| Script | Purpose |
|---|---|
| `train_xgboost.py` | Trains and evaluates traditional feature-based student models, including XGBoost. |
| `train_ditto.py` | Runs Ditto training from generated entity-matching labels. |
| `train_qwen.py` | Fine-tunes a Qwen student model with LoRA. |

Implementation-specific helper code is retained in `scripts/archive/ditto_internal/` and `scripts/archive/qwen_internal/`.
