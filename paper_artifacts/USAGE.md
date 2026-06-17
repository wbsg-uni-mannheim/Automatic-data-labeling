# Using the Repository

This guide describes how to inspect the released training data and how to run the three training-set construction workflows on Abt-Buy. Commands assume that they are executed from the repository root.

## Environment

Install the Python dependencies and provide an API key for the teacher model:

```bash
python -m pip install -r requirements.txt
export OPENAI_API_KEY=...
```

The Abt-Buy examples require the following local inputs:

- `data/abt-buy/abt-buy-train-left.csv`
- `data/abt-buy/abt-buy-train-right.csv`
- `data/abt-buy/embeddings/abt-buy_left_embeddings.npy`
- `data/abt-buy/embeddings/abt-buy_right_embeddings.npy`

Generated runs are written below `output/` unless an explicit `--output-root` is provided.

## Inspect Released Training Data

The released materialized training sets are stored under:

```text
paper_artifacts/training_data/
```

For example, the Abt-Buy GPT-5.2 similarity-search training set is:

```text
paper_artifacts/training_data/abt-buy/gpt-5.2/similarity_search/abt-buy_train.json.gz
```

Verify all released training files:

```bash
python scripts/artifacts/verify_release_artifacts.py
```

The verifier checks presence, gzip readability, and the row counts recorded in `paper_artifacts/training_data/MANIFEST.csv`.

## Construct Abt-Buy With Similarity Search

The similarity-search construction is implemented by the seed-round profile runner. It builds a FAISS nearest-neighbor candidate pool from the stored embeddings, labels selected high-similarity candidates with the teacher model, and exports profile-specific `active_labels_latest.csv`, `labels_final.csv`, and Ditto-compatible `*_train.json.gz` files.

Dry run:

```bash
python scripts/labeling/run_seed_round_only_profiles.py \
  --config configs/labeling/benchmarks_active.yaml \
  --benchmarks abt-buy \
  --profiles large \
  --output-root output/seed_round_only_profiles \
  --model gpt-5.2 \
  --dry-run
```

Full run:

```bash
python scripts/labeling/run_seed_round_only_profiles.py \
  --config configs/labeling/benchmarks_active.yaml \
  --benchmarks abt-buy \
  --profiles large \
  --output-root output/seed_round_only_profiles \
  --model gpt-5.2
```

Expected output pattern:

```text
output/seed_round_only_profiles/benchmark_abt-buy_<timestamp>/
```

The `profiles/large/` subdirectory contains the exported training file for the selected profile.

## Construct Abt-Buy With Feature-Based Active Learning

The feature-based active-learning workflow is implemented by `run_simple_labeling.py`. It starts from a teacher-labeled seed set, trains an in-script feature-based matcher, selects uncertain or informative candidates from the FAISS pool, asks the teacher model for additional labels, and repeats until the requested budget is reached.

Preview the first candidate comparisons without labeling:

```bash
python scripts/labeling/run_simple_labeling.py \
  --left-csv data/abt-buy/abt-buy-train-left.csv \
  --right-csv data/abt-buy/abt-buy-train-right.csv \
  --embeddings-dir data/abt-buy/embeddings \
  --left-emb abt-buy_left_embeddings.npy \
  --right-emb abt-buy_right_embeddings.npy \
  --left-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --right-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --model gpt-5.2 \
  --target-size 5000 \
  --target-positives 800 \
  --seed-size 100 \
  --seed-positives 30 \
  --seed-max-calls 400 \
  --labels-per-iteration 500 \
  --active-candidates 20000 \
  --max-iterations 20 \
  --llm-concurrency 10 \
  --output-root output/simple_active_learning \
  --run-name abt-buy_active_ml_example \
  --preview-k 5
```

Full run:

```bash
python scripts/labeling/run_simple_labeling.py \
  --left-csv data/abt-buy/abt-buy-train-left.csv \
  --right-csv data/abt-buy/abt-buy-train-right.csv \
  --embeddings-dir data/abt-buy/embeddings \
  --left-emb abt-buy_left_embeddings.npy \
  --right-emb abt-buy_right_embeddings.npy \
  --left-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --right-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --model gpt-5.2 \
  --target-size 5000 \
  --target-positives 800 \
  --seed-size 100 \
  --seed-positives 30 \
  --seed-max-calls 400 \
  --labels-per-iteration 500 \
  --active-candidates 20000 \
  --max-iterations 20 \
  --llm-concurrency 10 \
  --output-root output/simple_active_learning \
  --run-name abt-buy_active_ml_example
```

Expected output pattern:

```text
output/simple_active_learning/abt-buy_active_ml_example/
```

The main output files include `active_labels_latest.csv`, `labels_final.csv`, and a materialized `*_train.json.gz` export.

## Construct Abt-Buy With Ditto-Based Active Learning

The Ditto-based workflow is implemented by `run_three_phase_labeling.py`. It first constructs a seed set, then runs a feature-based active-learning phase, and finally uses a Ditto ensemble to select further candidates for teacher labeling. This workflow is more compute-intensive than feature-based active learning because Ditto models are trained during candidate selection.

Preview the first candidate comparisons without labeling:

```bash
python scripts/labeling/run_three_phase_labeling.py \
  --left-csv data/abt-buy/abt-buy-train-left.csv \
  --right-csv data/abt-buy/abt-buy-train-right.csv \
  --embeddings-dir data/abt-buy/embeddings \
  --left-emb abt-buy_left_embeddings.npy \
  --right-emb abt-buy_right_embeddings.npy \
  --left-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --right-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --model gpt-5.2 \
  --target-size 5000 \
  --target-positives 800 \
  --seed-size 100 \
  --seed-positives 30 \
  --seed-max-calls 400 \
  --phase2-target-size 1000 \
  --phase3-batch-size 500 \
  --phase3-candidates 20000 \
  --phase3-ensemble-mode ditto_only \
  --llm-concurrency 10 \
  --output-root output/three_phase_labeling_ditto_only \
  --run-name abt-buy_active_ditto_example \
  --preview-k 5
```

Full run:

```bash
python scripts/labeling/run_three_phase_labeling.py \
  --left-csv data/abt-buy/abt-buy-train-left.csv \
  --right-csv data/abt-buy/abt-buy-train-right.csv \
  --embeddings-dir data/abt-buy/embeddings \
  --left-emb abt-buy_left_embeddings.npy \
  --right-emb abt-buy_right_embeddings.npy \
  --left-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --right-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --model gpt-5.2 \
  --target-size 5000 \
  --target-positives 800 \
  --seed-size 100 \
  --seed-positives 30 \
  --seed-max-calls 400 \
  --phase2-target-size 1000 \
  --phase3-batch-size 500 \
  --phase3-candidates 20000 \
  --phase3-ensemble-mode ditto_only \
  --llm-concurrency 10 \
  --output-root output/three_phase_labeling_ditto_only \
  --run-name abt-buy_active_ditto_example
```

Expected output pattern:

```text
output/three_phase_labeling_ditto_only/abt-buy_active_ditto_example/
```

The full run writes label CSVs, run metadata, and a Ditto-compatible training export. Use `--resume` with the same `--run-name` to continue a partially completed run.
