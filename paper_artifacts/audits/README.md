# Human Audit Materials

The human audit materials are retained in their original output locations. They document the label-quality audit described in the paper.

## Sampling Frame and Review Interface

The audit sampling frame and review-interface exports are retained under `analysis/`. They define the reviewed pairs and annotation batches. The HTML files document the annotation environment rather than the authoritative human decision table.

## Human Decisions

- `results/error_anlysis/labelers/*_annotations.csv`
- `results/error_anlysis/labeler_error_rates.csv`

The `human_decision` column in the annotation files records the final human audit decision. The spelling of `error_anlysis` is preserved because it is the historical experiment-output directory referenced by existing scripts and result files.
