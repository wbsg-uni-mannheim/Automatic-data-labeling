# Benchmark preparation

Converters that turn external datasets into the layout the labeling and training configs read
(`<name>-train-{left,right}.csv`, `<name>-{train,valid,gs}.json.gz`, `<name>-valid.csv`). Each prints
the YAML blocks for `configs/labeling/benchmarks_active.yaml` and `configs/ditto/benchmarks_training.yaml`.

| Script | Source |
|---|---|
| `prepare_dn.py` | Papadakis et al. re-blocked ER-Magellan sets (`Dn1`…`Dn8` on Zenodo 10.5281/zenodo.8164151) |
| `prepare_machamp.py` | Machamp semi-structured tasks (github.com/megagonlabs/machamp) |
| `prepare_billiger.py` | billiger.de Products (WDC-Products layout) |

Record tables are derived from the records that appear in the benchmark training pairs, which keeps the
benchmark's record selection and undoes its pair selection — the same protocol as the five paper benchmarks.
Embeddings are built afterwards with `scripts/labeling/build_embeddings.py`. See `docs/NEW_BENCHMARKS_RUNBOOK.md`.
