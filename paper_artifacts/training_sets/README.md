# Materialized Training Sets

This directory documents the materialized training sets expected for the paper release. The training files themselves are not regenerated here. They must be copied from the canonical server output and placed at the `release_path` values listed in `paper_artifacts/EXPECTED_TRAINING_SETS.csv`.

## Core Scenarios

The core release covers the following paper scenarios:

- Selection-strategy comparison under GPT-5.2:
  - similarity search;
  - feature-based active learning;
  - Ditto-based active learning.
- Teacher-model comparison under Ditto-based active learning:
  - GPT-5.2;
  - Qwen 3.6 Plus;
  - Kimi K2.6.
- Post-processing variants applied to the GPT-5.2 Ditto active-learning set:
  - relabeling;
  - relabel-drop;
  - closure-drop;
  - closure and relabel-drop;
  - closure or relabel-drop.

All scenarios are expected for Abt-Buy, Walmart-Amazon, WDC Products, DBLP-ACM, and DBLP-Scholar. The canonical profile is `all_plus20random` except for WDC active-learning and post-processing runs, where the canonical profile is `all`.

## Optional Scenarios

The server fetch list also records optional validation and ablation output roots:

- `labeling_cleaned_llmvalid`: LLM-validation variants;
- `labeling_cleaned_no_val`: runs without validation split handling;
- `labeling_cleaned_size_scan`: Abt-Buy size-scan outputs.

These files are useful for extended reproducibility but are not required for the main artifact completeness check. They can be added to `EXPECTED_TRAINING_SETS.csv` as optional rows if the final public release should include the full ablation material as independently verified files.
