# Artifacts

The release index for the paper: the machine-labeled training sets, the prompts, the human audit, and links to the scripts that build and evaluate them.

## Training data

All 50 training sets sit under [`training_data/`](training_data/):

```text
training_data/{benchmark}/{teacher}/{method}/{benchmark}_train.json.gz
```

Each file trains the downstream students (Ditto, Qwen3, XGBoost). [`training_data/README.md`](training_data/README.md) covers the teacher/method matrix and the post-processing variants.

## Scripts

[`USAGE.md`](USAGE.md) gives runnable Abt-Buy commands for the three construction workflows. [`SCRIPTS_AND_CONFIGS.md`](SCRIPTS_AND_CONFIGS.md) maps every workflow stage to its file.

**Training-set construction**

- Similarity search — [`scripts/labeling/similarity_search.py`](../scripts/labeling/similarity_search.py)
- Feature-based active learning — [`scripts/labeling/active_learning_ml.py`](../scripts/labeling/active_learning_ml.py)
- Ditto-based active learning — [`scripts/labeling/active_learning_ditto.py`](../scripts/labeling/active_learning_ditto.py)

**Post-processing variants**

- Relabel disagreements — [`scripts/post_processing/relabel_three_phase_generated_labels_batch.py`](../scripts/post_processing/relabel_three_phase_generated_labels_batch.py)
- Relabel-drop profiles — [`scripts/post_processing/build_drop_changed_profiles.py`](../scripts/post_processing/build_drop_changed_profiles.py)
- Closure and closure/relabel profiles — [`scripts/post_processing/build_closure_bridge_profiles.py`](../scripts/post_processing/build_closure_bridge_profiles.py)

**Student training**

- XGBoost — [`scripts/training/train_xgboost.py`](../scripts/training/train_xgboost.py)
- Ditto — [`scripts/training/train_ditto.py`](../scripts/training/train_ditto.py)
- Qwen — [`scripts/training/train_qwen.py`](../scripts/training/train_qwen.py)

## Prompts

[`prompts/`](prompts/) holds the teacher-labeling prompt and the post-processing relabeling prompt. See [`prompts/README.md`](prompts/README.md).

## Human audit

[`error_anlysis/`](../error_anlysis/) holds the human review of teacher labels: one CSV per benchmark batch with the `human_decision` for each reviewed pair, plus the per-teacher error rates.
