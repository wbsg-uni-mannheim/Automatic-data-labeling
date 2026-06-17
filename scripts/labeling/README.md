# Labeling Scripts

This directory contains the entry points for the initial automatic labeling workflow.

## Main scripts

- `run_simple_labeling.py`: single-run active labeling pipeline
- `run_three_phase_labeling.py`: seed phase, classic active-learning phase, then Ditto-ensemble phase
- `run_benchmark_labeling.py`: benchmark-aware wrapper that materializes profiles and exports training artifacts

## Examples

```bash
python scripts/labeling/run_benchmark_labeling.py \
  --config configs/labeling/benchmarks_active.yaml \
  --benchmark abt-buy \
  --profiles large \
  --dry-run
```

For full Abt-Buy commands for similarity search, feature-based active learning, and Ditto-based active learning, see `paper_artifacts/USAGE.md`.

These scripts are intentionally separate from later-stage analysis, relabeling, or label-cleanup utilities.
