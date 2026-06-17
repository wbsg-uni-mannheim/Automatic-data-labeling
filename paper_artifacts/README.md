# Artifact Inventory

This directory is the publication-facing artifact index for the LLM-labeled entity-matching training data study. It documents the release materials and provides a stable layout for reuse.

The directory does not generate or infer new labels. Materialized training sets are stored under `training_data/` and are verified against `training_data/MANIFEST.csv`.

## Core Materials

The core artifact set consists of:

- prompts for initial teacher labeling and post-processing;
- scripts for candidate generation, pair selection, LLM labeling, relabeling, post-processing, and downstream training;
- result tables used in the study;
- human audit sampling frames and annotation files;
- materialized final training sets for every reported scenario.

The materialized training-set files are listed in `training_data/MANIFEST.csv`.

## Release Layout

Materialized release files are placed under:

```text
paper_artifacts/training_data/
```

The directory structure is organized by benchmark, teacher, and construction method. This keeps the public artifact layout independent of machine-local experiment directories.

## Verification

Run:

```bash
python scripts/artifacts/verify_release_artifacts.py
```

The checker reports missing files, corrupt gzip streams, and row-count mismatches. It exits with a non-zero status if any listed artifact is absent or unreadable.

## Documentation Subsections

- `training_data/README.md`: describes the materialized training-set scenarios.
- `USAGE.md`: provides concrete commands for constructing Abt-Buy training sets with the three selection workflows.
- `prompts/README.md`: identifies the prompt files used in the workflow.
- `audits/README.md`: identifies the audit sampling frame and human annotation files.
- `SCRIPTS_AND_CONFIGS.md`: maps workflow stages to scripts and configuration files.

## Non-Goals

This directory is not a substitute for the public benchmark sources. The benchmark datasets remain externally sourced. The artifact bundle records the machine-labeled training sets, prompts, audit decisions, scripts, and result metadata needed to reproduce the study's reported training-set construction results without rerunning LLM labeling.
