# Labeling Scripts

This directory contains the entry points for the initial automatic labeling workflow.

## Main scripts

- `run_simple_labeling.py`: single-run active labeling pipeline
- `run_three_phase_labeling.py`: seed phase, classic active-learning phase, then Ditto-ensemble phase
- `run_benchmark_labeling.py`: benchmark-aware wrapper that materializes profiles and exports training artifacts

## Example

```bash
python scripts/labeling/run_benchmark_labeling.py \
  --config configs/labeling/benchmarks.yaml \
  --benchmark wdc
```

These scripts are intentionally separate from later-stage analysis, relabeling, or label-cleanup utilities.
