# Cross-Encoder Training Pipeline

This is a second EM training stack beside Ditto. It reuses the same benchmark split normalization and `COL ... VAL ...` pair serialization, but trains a plain Hugging Face `AutoModelForSequenceClassification` cross-encoder.

## Default model

The default config uses:

```bash
BAAI/bge-reranker-v2-m3
```

For a faster smoke test or smaller GPU, override it with a lighter model:

```bash
--model-name roberta-base
```

or set `defaults.model_name` in `configs/cross_encoder/benchmarks_training.yaml`.

## Run One Benchmark

```bash
python scripts/cross_encoder/run_benchmark_training.py \
  --benchmarks abt-buy
```

## Run All Benchmarks

```bash
python scripts/cross_encoder/run_benchmark_training.py
```

## Dry Run

This prepares normalized train/valid/test splits and benchmark reports without training:

```bash
python scripts/cross_encoder/run_benchmark_training.py \
  --benchmarks abt-buy \
  --dry-run
```

## Single Prepared Split

```bash
python scripts/cross_encoder/train.py \
  --train-json-gz output/cross_encoder_benchmark_runs/run_YYYYMMDD_HHMMSS/abt-buy/splits/train.json.gz \
  --val-json-gz output/cross_encoder_benchmark_runs/run_YYYYMMDD_HHMMSS/abt-buy/splits/valid.json.gz \
  --test-json-gz output/cross_encoder_benchmark_runs/run_YYYYMMDD_HHMMSS/abt-buy/splits/test.json.gz \
  --config configs/cross_encoder/default_train.yaml
```

## Outputs

Benchmark runs write:

- `summary.json`
- `summary.csv`
- per-benchmark `benchmark_report.json`
- normalized `splits/*.json.gz`
- `training_output/LATEST_RUN`

Training runs write:

- `config.json`
- `history.json`
- `metrics.json`
- `predictions.csv`
- `checkpoints/best/`
