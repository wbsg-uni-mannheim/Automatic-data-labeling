# Artifact Utilities

This directory contains utilities for preparing and validating the paper artifact release.

## Verification

Use `verify_release_artifacts.py` to check the materialized training files listed in `paper_artifacts/training_data/MANIFEST.csv`.

Run from the repository root:

```bash
python scripts/artifacts/verify_release_artifacts.py
```

The script checks file presence, gzip readability, and row counts. It does not create, transform, or relabel training data.
