# Benchmarks

The eight entity-matching benchmarks used in the paper, one directory each:

- `abt-buy/`
- `walmart-amazon/`
- `dblp-acm/`
- `dblp-scholar/`
- `dn7-walmart-amazon/` (Dn7 Walmart-Amazon from Papadakis et al., ICDE 2024)
- `wdc/` (WDC Products, test set with 100% unseen entities)
- `billiger-de/` (billiger.de Products, German offers, test set with 50% unseen entities)
- `semi-heter/` (Semi-HETER from Machamp)

The datasets are publicly available. Their splits and precomputed embeddings, in the form the runners read, are included here. [`scripts/benchmarks/`](../scripts/benchmarks/) converts Dn7, billiger.de, and Semi-HETER from their original releases into this layout.

Notes on the three benchmarks added in the PVLDB version:

- **billiger.de:** the source tables come from the large training split (`billiger-de-train.json.gz`). The benchmark reference is trained on the official medium training split with 5,897 pairs (`billiger-de-train-medium.json.gz`). The test set is the default test set with 50% unseen products (4,437 pairs).
- **Semi-HETER:** the attribute names of the five book sources are mapped to shared attributes (title, authors, publisher, year, isbn, pages, price). Unmapped keys are kept in an `extra` field.
- **Dn7 Walmart-Amazon:** uses the same Walmart and Amazon records as Walmart-Amazon with the imbalanced splits of Papadakis et al.

## Layout

Each benchmark directory holds:

- `*-train-left.csv`, `*-train-right.csv`: the two record tables for the training candidate pool.
- `*-valid.csv` and `*-gs.json.gz`: validation split and the labeled gold-standard test set.
- `embeddings/{benchmark}_left_embeddings.npy`, `{benchmark}_right_embeddings.npy`: dense record embeddings used for similarity blocking and candidate selection.
- `*-batch_embed_left.jsonl`, `*-batch_embed_right.jsonl`, the matching `*_map.csv` files, and `*-batch_embed_manifest.json`: the request files and row maps used to produce the embeddings.

WDC Products uses its own file names (for example `wdc_train_large_left.csv` and `wdcproducts80cc20rnd000un_*.json.gz`) but follows the same structure.

The paths here match what the labeling and training configs expect, for example `configs/labeling/benchmarks_active.yaml` and `configs/ditto/benchmarks_training.yaml`.

## Embeddings

The embeddings are OpenAI `text-embedding-3-small` vectors at 256 dimensions, computed over the concatenated record fields (for example `title`, `description`, `price`). The exact fields and the submitted batch jobs are recorded in each `*-batch_embed_manifest.json`.

The two WDC embedding matrices are about 117 MB each, which is over GitHub's 100 MB file limit, so they are excluded from the repository:

- `benchmarks/wdc/embeddings/wdc_left_embeddings.npy`
- `benchmarks/wdc/embeddings/wdc_right_embeddings.npy`

To rebuild them, embed the records in `benchmarks/wdc/batch_embed_left.jsonl` and `benchmarks/wdc/batch_embed_right.jsonl` with the same model. Each line is an OpenAI `/v1/embeddings` request keyed by `custom_id` (`left-0`, `left-1`, and so on). Stack the returned vectors in `custom_id` order into a `(rows × 256)` float array and save each side as the `.npy` file named above. The embeddings for the other seven benchmarks are small and are included directly.
