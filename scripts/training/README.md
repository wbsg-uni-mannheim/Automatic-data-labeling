# Training Entry Points

Train a student model on the machine-labeled sets.

| Script | What it does |
|---|---|
| `train_xgboost.py` | Trains and evaluates the feature-based student (XGBoost). |
| `train_ditto.py` | Trains and evaluates Ditto on generated labels. |
| `train_qwen.py` | Fine-tunes a Qwen student with LoRA. |

Helper code for the Ditto and Qwen students sits in `../archive/ditto_internal/` and `../archive/qwen_internal/`.
