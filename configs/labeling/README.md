# Benchmark Labeling Config

`benchmarks_active.yaml` defines each benchmark for the labeling workflows: source files, schema mappings, profile targets, and output settings. `benchmarks_active_kimi.yaml` is the same setup with Kimi K2.6 as the teacher.

## Use

The labeling entry points read this config with `--config`:

```bash
python scripts/labeling/similarity_search.py \
  --config configs/labeling/benchmarks_active.yaml \
  --benchmarks wdc \
  --profiles large \
  --model gpt-5.2
```

To label several benchmarks in one call, `scripts/archive/labeling_helpers/run_benchmark_labeling.py` reads the same config and supports `--dry-run` to print the resolved profile plan without calling the API.

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

Profiles are defined globally in `benchmarks_active.yaml` (or overridden per benchmark) and can include:

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
