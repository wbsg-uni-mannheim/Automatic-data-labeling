# Human Audit of Teacher Labels

To measure teacher-label quality, we reviewed 1,028 test pairs across five benchmarks: all 778 pairs where at least one teacher disagrees with the gold label, plus 25 gold-positive and 25 gold-negative controls per benchmark on which every teacher agrees with gold. Five ambiguous judgments are excluded from error-rate calculation.

## Annotations

`labelers/` holds one CSV per audited batch, named by benchmark (for example `test_teacher_audit__batch_01_test_abt_buy__v1_annotations.csv`). Each row is one reviewed pair:

| Column | Meaning |
|---|---|
| `benchmark` | Benchmark the pair comes from. |
| `pair_id` | Identifier of the reviewed record pair. |
| `human_decision` | The reviewer's label for the pair: `match`, `non_match`, or `ambiguous`. |

## Error rates

`labeler_error_rates.csv` reports inverse-probability-weighted population error estimates, not the unweighted share of audited pairs. [`sampling_manifest.csv`](sampling_manifest.csv) supplies the pair identities, strata, weights, and evaluated labels. A disagreement pair has weight 1; each control receives its positive/negative stratum population divided by the 25 sampled controls. Each rate is the weighted number of errors divided by the total weight of unambiguous judgments.

| Column | Meaning |
|---|---|
| `benchmark` | Benchmark name. |
| `n_annotated`, `n_total` | Number of pairs reviewed out of the sampled total. |
| `partial` | Whether the batch was only partially reviewed. |
| `Benchmark` | Error rate of the benchmark's own gold labels. |
| `gpt-5.2`, `qwen`, `kimi` | Error rate of each teacher model. |
