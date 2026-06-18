# Usage

Inspect the released training data and run the three construction workflows on Abt-Buy. Run every command from the repository root.

## Environment

Install the Python dependencies and provide an API key for the teacher model:

```bash
python -m pip install -r requirements.txt
export OPENAI_API_KEY=...
```

The Abt-Buy examples require the following local inputs:

- `benchmarks/abt-buy/abt-buy-train-left.csv`
- `benchmarks/abt-buy/abt-buy-train-right.csv`
- `benchmarks/abt-buy/embeddings/abt-buy_left_embeddings.npy`
- `benchmarks/abt-buy/embeddings/abt-buy_right_embeddings.npy`

Generated runs are written below `output/` unless an explicit `--output-root` is provided.

## Inspect the released training data

The training sets live under:

```text
artifacts/training_data/
```

For example, the Abt-Buy GPT-5.2 similarity-search set is:

```text
artifacts/training_data/abt-buy/gpt-5.2/similarity_search/abt-buy_train.json.gz
```

## Construct Abt-Buy with similarity search

`similarity_search.py` builds a FAISS nearest-neighbor candidate pool from the stored embeddings, labels the top candidates with the teacher, and exports `active_labels_latest.csv`, `labels_final.csv`, and a Ditto-compatible `*_train.json.gz` per profile.

Dry run:

```bash
python scripts/labeling/similarity_search.py \
  --config configs/labeling/benchmarks_active.yaml \
  --benchmarks abt-buy \
  --profiles large \
  --output-root output/seed_round_only_profiles \
  --model gpt-5.2 \
  --dry-run
```

Full run:

```bash
python scripts/labeling/similarity_search.py \
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

## Construct Abt-Buy with feature-based active learning

`active_learning_ml.py` starts from a teacher-labeled seed set, trains a feature-based matcher, picks informative candidates from the FAISS pool, asks the teacher to label them, and repeats until it reaches the target budget.

Preview the first candidate comparisons without labeling:

```bash
python scripts/labeling/active_learning_ml.py \
  --left-csv benchmarks/abt-buy/abt-buy-train-left.csv \
  --right-csv benchmarks/abt-buy/abt-buy-train-right.csv \
  --embeddings-dir benchmarks/abt-buy/embeddings \
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
python scripts/labeling/active_learning_ml.py \
  --left-csv benchmarks/abt-buy/abt-buy-train-left.csv \
  --right-csv benchmarks/abt-buy/abt-buy-train-right.csv \
  --embeddings-dir benchmarks/abt-buy/embeddings \
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

## Construct Abt-Buy with Ditto-based active learning

`active_learning_ditto.py` runs three phases: build a seed set, run a feature-based active-learning phase, then use a Ditto ensemble to select the remaining candidates for teacher labeling. It costs more compute than feature-based active learning because it trains Ditto models during candidate selection.

Preview the first candidate comparisons without labeling:

```bash
python scripts/labeling/active_learning_ditto.py \
  --left-csv benchmarks/abt-buy/abt-buy-train-left.csv \
  --right-csv benchmarks/abt-buy/abt-buy-train-right.csv \
  --embeddings-dir benchmarks/abt-buy/embeddings \
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
python scripts/labeling/active_learning_ditto.py \
  --left-csv benchmarks/abt-buy/abt-buy-train-left.csv \
  --right-csv benchmarks/abt-buy/abt-buy-train-right.csv \
  --embeddings-dir benchmarks/abt-buy/embeddings \
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
