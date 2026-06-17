# Automatic Data Labeling for Entity Matching

This repository contains the experimental code, prompts, result tables, and artifact documentation for a study on constructing entity-matching training sets with large language model teachers. The workflow builds candidate pools, selects informative record pairs, labels them with LLMs, applies post-processing checks, and evaluates the resulting materialized training sets with downstream matchers.

The artifact layer in `paper_artifacts/` contains the release materials required to inspect and reuse the study outputs. Manuscript sources are maintained outside this repository; any local manuscript directory is a convenience link and is not part of the public release.

## Scope

The experiments cover five entity-matching benchmarks:

- Abt-Buy
- Walmart-Amazon
- WDC Products
- DBLP-ACM
- DBLP-Scholar

The main experimental dimensions are:

- pair-selection strategy: similarity search, feature-based active learning, and Ditto-based active learning;
- teacher model: GPT-5.2, Qwen 3.6 Plus, and Kimi K2.6;
- post-processing: relabeling, relabel-drop, closure-drop, and combined closure/relabel variants;
- downstream student model: Ditto, XGBoost, and Qwen3 fine-tuning variants;
- audit material for teacher and benchmark label quality.

## Repository Map

- `configs/labeling/`: benchmark-level labeling configuration.
- `scripts/labeling/`: candidate-pool construction, benchmark labeling, relabeling, and post-processing utilities.
- `scripts/ditto/`: Ditto training wrappers used for downstream evaluation.
- `llm_em_qwen35/`: Qwen fine-tuning and conversion utilities.
- `scripts/analysis/`: audit, profiling, and result-analysis utilities.
- `results/`: study result tables and evaluation summaries.
- `analysis/`: audit sampling frames, review interface exports, and supporting analysis outputs.
- `results/error_anlysis/labelers/`: human audit annotations. The directory name is preserved from the original experiment output.
- `paper_artifacts/`: publication-facing prompts, audit references, and materialized training data.

## Artifact Status

The artifact release provides code, prompts, post-processing material, human audit decisions, and materialized machine-labeled training sets. The public artifact inventory is:

- `paper_artifacts/README.md`
- `paper_artifacts/USAGE.md`
- `paper_artifacts/training_data/README.md`
- `paper_artifacts/training_data/MANIFEST.csv`
- `scripts/artifacts/verify_release_artifacts.py`

The materialized training files are stored under `paper_artifacts/training_data/`. The manifest records the benchmark, teacher, construction method, row count, and public artifact path for each file.

To check the materialized training data, run:

```bash
python scripts/artifacts/verify_release_artifacts.py
```

For concrete Abt-Buy commands covering similarity search, feature-based active learning, and Ditto-based active learning, see `paper_artifacts/USAGE.md`.

## Reproduction Entry Points

For a worked Abt-Buy example covering similarity search, feature-based active learning, and Ditto-based active learning, see `paper_artifacts/USAGE.md`.

For command-line help on the main labeling runners:

```bash
python scripts/labeling/run_seed_round_only_profiles.py --help
python scripts/labeling/run_simple_labeling.py --help
python scripts/labeling/run_three_phase_labeling.py --help
```

Relabeling and post-processing utilities:

```bash
python scripts/labeling/relabel_three_phase_generated_labels_batch.py --help
python scripts/labeling/build_drop_changed_profiles.py --help
python scripts/labeling/build_closure_bridge_profiles.py --help
```

Ditto downstream training:

```bash
python scripts/ditto/run_benchmark_training.py --help
```

Qwen training utilities:

```bash
python llm_em_qwen35/convert_wdc_to_sft.py --help
```

## Notes on Data Release

The repository distinguishes between executable code, documentation artifacts, and materialized training data. The authoritative public list of training data is `paper_artifacts/training_data/MANIFEST.csv`. Run the artifact verifier before tagging or archiving a release.

## Citation

Please cite the accompanying manuscript when using the code, prompts, or materialized training sets.
