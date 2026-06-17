# Paper Artifact Inventory

This directory is the publication-facing artifact index for the paper on LLM-labeled training data for entity matching. It documents the materials promised in the paper's artifact section and defines the expected layout for a public release bundle.

The directory does not generate or infer new labels. Materialized training sets must be copied from the canonical experiment output on the server and then verified against the inventory.

## Core Materials

The core artifact set consists of:

- prompts for initial teacher labeling and post-processing;
- scripts for candidate generation, pair selection, LLM labeling, relabeling, post-processing, and downstream training;
- result tables and run registries used by the paper;
- human audit sampling frames and annotation files;
- materialized final training sets for every scenario reported in the paper.

The expected training-set files are listed in `EXPECTED_TRAINING_SETS.csv`.

## Expected Release Layout

Materialized release files should be placed under:

```text
paper_artifacts/training_sets/materialized/
```

The `release_path` column in `EXPECTED_TRAINING_SETS.csv` gives the exact path for every expected file. The `source_path` column records the original experiment-output path used by the run registry.

Core files are those needed for the paper's main artifact promise. Optional files are ablation or validation variants that are useful for deeper reproducibility but are not required for the main paper tables.

## Verification

After copying files from the server, run:

```bash
python scripts/artifacts/verify_release_artifacts.py --tier core --layout release
```

To verify the original server-style output directory instead:

```bash
python scripts/artifacts/verify_release_artifacts.py --tier core --layout source
```

The checker reports missing and corrupt gzip files. It exits with a non-zero status if any expected file is absent or unreadable.

## Documentation Subsections

- `training_sets/README.md`: describes the expected training-set scenarios and profiles.
- `prompts/README.md`: identifies the prompt files used in the workflow.
- `audits/README.md`: identifies the audit sampling frame and human annotation files.
- `SCRIPTS_AND_CONFIGS.md`: maps paper workflow stages to scripts and configuration files.
- `SERVER_FETCH_LIST.md`: summarizes the server directories that must be copied to complete the release bundle.

## Non-Goals

This directory is not a substitute for the public benchmark sources. The benchmark datasets remain externally sourced. The artifact bundle records the machine-labeled training sets, prompts, audit decisions, scripts, and result metadata needed to reproduce the study's reported training-set construction results without rerunning LLM labeling.
