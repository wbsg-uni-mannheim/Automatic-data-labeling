# Scripts and Configuration Index

This file maps the study workflow to the repository files that implement it. The list is intentionally explicit so that the artifact bundle can be inspected without reverse-engineering the directory tree.

## Candidate Generation and Initial Labeling

- `configs/labeling/benchmarks.yaml`: benchmark definitions, schema mappings, profile targets, and output settings.
- `scripts/labeling/run_benchmark_labeling.py`: benchmark-aware labeling runner that materializes profiles and Ditto training files.
- `scripts/labeling/run_three_phase_labeling.py`: seed, classic active-learning, and Ditto-ensemble labeling phases.
- `scripts/labeling/run_simple_labeling.py`: simpler active-labeling runner.
- `scripts/labeling/run_seed_round_only_profiles.py`: seed/similarity-search profile construction.

## Relabeling and Post-Processing

- `scripts/labeling/relabel_three_phase_generated_labels_batch.py`: batch relabeling of generated labels.
- `scripts/labeling/build_drop_changed_profiles.py`: construction of relabel-drop profiles.
- `scripts/labeling/build_closure_bridge_profiles.py`: construction of closure-drop and combined closure/relabel profiles.
- `scripts/labeling/build_rebuilt_profile_manifest.py`: profile-manifest rebuilding utility.

## Downstream Training

- `scripts/ditto/run_benchmark_training.py`: Ditto training wrapper used for benchmark evaluation.
- `scripts/ditto/train.py`: Ditto training entry point.
- `llm_em_qwen35/`: Qwen fine-tuning and conversion utilities.
- `scripts/cross_encoder/`: cross-encoder training utilities retained for related experiments.

## Analysis and Audit

- `scripts/profile_autolabel_training_sets.py`: training-set profile analysis.
- `scripts/analysis/`: audit batch generation, review-interface generation, and audit summary utilities.
- `paper_artifacts/training_data/MANIFEST.csv`: public manifest for materialized training data.

## Artifact Verification

- `scripts/artifacts/verify_release_artifacts.py`: verifies presence, gzip readability, and row counts for files listed in `paper_artifacts/training_data/MANIFEST.csv`.
