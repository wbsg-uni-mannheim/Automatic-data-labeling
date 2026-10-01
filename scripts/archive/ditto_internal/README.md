# Ditto Internals

Ditto training and evaluation for the Ditto students. `scripts/training/train_ditto.py` calls `train.py` and `evaluate.py` with the settings of its YAML config. The model code is in `third_party/ditto_modern/`.

| File | What it does |
|---|---|
| `train.py` | Fine-tunes RoBERTa-base on a training file and selects the checkpoint with the best validation F1. |
| `evaluate.py` | Evaluates a checkpoint on a test file. |
| `convert_active_labels_to_wdc.py` | Converts the label CSVs of the labeling workflows into the JSON pair format of the training sets. |
| `convert_wdc_to_ditto.py` | Converts JSON pairs into the original Ditto text format. |
| `tests/` | Tests for the conversion. |

Input files use the WDC Products pair format, for example `benchmarks/wdc/wdcproducts80cc20rnd000un_train_large.json.gz`. `artifacts/USAGE.md` describes the environment and a training run.
