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

Notes on Dn7 Walmart-Amazon, billiger.de, and Semi-HETER:

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

The embeddings are OpenAI `text-embedding-3-small` vectors computed over the concatenated record fields (for example `title`, `description`, `price`), at 256 dimensions for seven benchmarks and at the model's default 1,536 dimensions for WDC Products. The exact fields and the submitted batch requests are recorded in each `*-batch_embed_manifest.json` and in `benchmarks/wdc/batch_embed_*.jsonl`.

The two WDC Products matrices are stored in two parts each (`wdc_left_embeddings.part1.npy`, `wdc_left_embeddings.part2.npy`, and the same for the right table) to stay below GitHub's file size limit. The scripts read embeddings through `scripts/labeling/embeddings_io.py`, which concatenates the parts into the full matrix, so the configured path `benchmarks/wdc/embeddings/wdc_left_embeddings.npy` works unchanged.
