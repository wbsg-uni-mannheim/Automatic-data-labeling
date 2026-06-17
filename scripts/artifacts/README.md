# Artifact Utilities

This directory contains utilities for preparing and validating the paper artifact release.

## Verification

Use `verify_release_artifacts.py` to check the materialized training files listed in `paper_artifacts/EXPECTED_TRAINING_SETS.csv`.

Core release layout:

```bash
python scripts/artifacts/verify_release_artifacts.py --tier core --layout release
```

Original experiment-output layout:

```bash
python scripts/artifacts/verify_release_artifacts.py --tier core --layout source
```

The script checks for file presence and validates gzip readability for `.gz` files. It does not create, transform, or relabel training data.
