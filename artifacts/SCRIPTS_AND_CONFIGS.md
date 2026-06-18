# Scripts and Configuration Index

This maps each stage of the workflow to the file that implements it.

## Candidate Generation and Initial Labeling

- `configs/labeling/benchmarks_active.yaml`: benchmark definitions, schema mappings, profile targets, and output settings.
- `scripts/labeling/similarity_search.py`: similarity-search profile construction.
- `scripts/labeling/active_learning_ml.py`: feature-based active-learning runner.
- `scripts/labeling/active_learning_ditto.py`: seed, feature-based active-learning, and Ditto-based active-learning phases.

## Relabeling and Post-Processing

- `scripts/post_processing/relabel_three_phase_generated_labels_batch.py`: batch relabeling of generated labels.
- `scripts/post_processing/build_drop_changed_profiles.py`: construction of relabel-drop profiles.
- `scripts/post_processing/build_closure_bridge_profiles.py`: construction of closure-drop and combined closure/relabel profiles.

## Downstream Training

- `scripts/training/train_xgboost.py`: XGBoost student training and evaluation.
- `scripts/training/train_ditto.py`: Ditto student training and evaluation.
- `scripts/training/train_qwen.py`: Qwen student fine-tuning.

