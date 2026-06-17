# Human Audit of Teacher Labels

To measure teacher-label quality, we manually reviewed samples of teacher-labeled test pairs across the five benchmarks.

## Annotations

`labelers/` holds one CSV per audited batch, named by benchmark (for example `test_teacher_audit__batch_01_test_abt_buy__v1_annotations.csv`). Each row is one reviewed pair:

| Column | Meaning |
|---|---|
| `benchmark` | Benchmark the pair comes from. |
| `pair_id` | Identifier of the reviewed record pair. |
| `human_decision` | The reviewer's label for the pair: `match`, `non_match`, or `ambiguous`. |

## Error rates

`labeler_error_rates.csv` summarizes the audit: the share of audited pairs (in percent) where each label source disagrees with the human decision, per benchmark.

| Column | Meaning |
|---|---|
| `benchmark` | Benchmark name. |
| `n_annotated`, `n_total` | Number of pairs reviewed out of the sampled total. |
| `partial` | Whether the batch was only partially reviewed. |
| `Benchmark` | Error rate of the benchmark's own gold labels. |
| `gpt-5.2`, `qwen`, `kimi` | Error rate of each teacher model. |
