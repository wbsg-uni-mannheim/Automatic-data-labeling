# Benchmark Labeling Config

This folder contains config-driven entry points for running automatic labeling across multiple benchmarks.

## Main Runner

Use:

```bash
python scripts/run_benchmark_labeling.py \
  --config configs/labeling/benchmarks.yaml \
  --benchmark wdc
```

Dry-run (prints resolved command and profile plan without API labeling):

```bash
python scripts/run_benchmark_labeling.py \
  --config configs/labeling/benchmarks.yaml \
  --benchmark wdc \
  --dry-run
```

## Schema Mapping

For each benchmark, `left_schema_map` and `right_schema_map` map canonical fields to source columns:

- `id` (required)
- `title` (optional)
- `brand` (optional)
- `description` (optional)
- `price` (optional)
- `priceCurrency` (optional)

Missing optional fields are allowed and will be filled with empty strings.

## Nested Profiles

Profiles are defined globally in `benchmarks.yaml` (or overridden per benchmark) and can include:

- numeric targets (`small`, `medium`, `large`) as strict nested subsets
- `all` with `{all_examples: true}` to export the full labeled pool

You can override profile targets per benchmark via `benchmarks.<name>.profiles`.

The runner labels once at the largest numeric profile, then writes per-profile outputs.
For `all`, it exports from `active_labels_latest.csv` (untrimmed) when available.

- `.../profiles/<profile>/active_labels_latest.csv`
- `.../profiles/<profile>/labels_final.csv`
- `.../profiles/<profile>/active_labels_latest_<benchmark>_<profile>_train.json.gz`

It also writes:

- `.../profile_manifest.json` with benchmark, canonical source files, and per-profile train file paths.
