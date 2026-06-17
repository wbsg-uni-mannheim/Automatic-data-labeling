# Artifacts

This folder is the entry point for everything released with the paper. It holds the machine-labeled training sets, the prompts, and pointers to the scripts that build and evaluate them. The scripts themselves live in their normal places in the repository and are linked from here.

## Training data

All 50 final training sets are under [`training_data/`](training_data/), organized as:

```text
training_data/{benchmark}/{teacher}/{method}/{benchmark}_train.json.gz
```

Each file is a materialized training set used by the downstream students (Ditto, Qwen3, XGBoost). [`training_data/MANIFEST.csv`](training_data/MANIFEST.csv) records the benchmark, teacher, method, row count, and path for every file, and [`training_data/README.md`](training_data/README.md) explains the coverage matrix and the post-processing variants.

## Scripts

The training sets above are produced and consumed by these scripts. [`USAGE.md`](USAGE.md) gives runnable Abt-Buy commands for the three construction workflows, and [`SCRIPTS_AND_CONFIGS.md`](SCRIPTS_AND_CONFIGS.md) maps every workflow stage to its file.

**Training-set construction (LLM labeling)**

- Similarity search — [`scripts/labeling/similarity_search.py`](../scripts/labeling/similarity_search.py)
- Feature-based active learning (ML student) — [`scripts/labeling/active_learning_ml.py`](../scripts/labeling/active_learning_ml.py)
- Ditto-based active learning — [`scripts/labeling/active_learning_ditto.py`](../scripts/labeling/active_learning_ditto.py)

**Post-processing variants**

- Relabel disagreements — [`scripts/post_processing/relabel_three_phase_generated_labels_batch.py`](../scripts/post_processing/relabel_three_phase_generated_labels_batch.py)
- Relabel-drop profiles — [`scripts/post_processing/build_drop_changed_profiles.py`](../scripts/post_processing/build_drop_changed_profiles.py)
- Closure-drop and combined closure/relabel profiles — [`scripts/post_processing/build_closure_bridge_profiles.py`](../scripts/post_processing/build_closure_bridge_profiles.py)

**Student training**

- XGBoost — [`scripts/training/train_xgboost.py`](../scripts/training/train_xgboost.py)
- Ditto — [`scripts/training/train_ditto.py`](../scripts/training/train_ditto.py)
- Qwen — [`scripts/training/train_qwen.py`](../scripts/training/train_qwen.py), with supporting utilities retained under [`scripts/archive/qwen_internal/`](../scripts/archive/qwen_internal/).

## Prompts

[`prompts/`](prompts/) holds the teacher-labeling prompt and points to the post-processing review prompts. See [`prompts/README.md`](prompts/README.md).

## Human audit

The human review of teacher labels is in [`results/error_anlysis/`](../results/error_anlysis/) — one CSV per benchmark batch, each row holding the `human_decision` for a reviewed pair.
