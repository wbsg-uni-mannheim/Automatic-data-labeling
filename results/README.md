# Results

Run-level results behind the tables and figures of the paper. F1 is in percentage points on the benchmark test sets.

| File | Contents |
|---|---|
| `table2_training_sources.csv` | Ditto students trained on the benchmark training sets and on the GPT-5.2 sets of the three selection strategies |
| `table3_benchmark_pairs_gpt52_labels.csv` | Ditto students trained on the benchmark training pairs with GPT-5.2 labels |
| `figure2_label_budget.csv` | Ditto students per selection strategy and label budget |
| `table4_teachers.csv` | Ditto students trained on the AL-Ditto sets of the three teachers |
| `table5_post_processing.csv` | Ditto students trained on the post-processed AL-Ditto sets |
| `table6_student_models.csv` | XGBoost and Qwen3 students, and Qwen3 zero-shot results |
| `table7_direct_teachers.csv` | Zero-shot teacher F1, recomputed from the per-pair predictions in `table7_direct_teachers/` |
| `table8_labeling_costs.csv` | GPT-5.2 tokens and cost per acquisition, and AL-Ditto committee fit hours per benchmark |
| `table9_inference_time.csv` | Inference time per 1,000 pairs, per matcher and benchmark |
| `table10_composition.csv` | Written by `scripts/compute_composition.py` from the training sets |

Columns of the per-seed files:

- `seed`: student training seed.
- `f1`: test F1 of the run.
- `n_train`: number of training pairs the student saw.
- `training_file`: path of the training set in this repository.
- `reported_mean`, `reported_sd`: the values printed in the paper (sample SD).
- `note` (Table 6): deviations of a run from the standard setup.

Table 9 combines local measurements (mean of three post-warmup passes on one A40 GPU with six CPU threads, model loading excluded) with API measurements at ten concurrent workers. For the API teachers, three benchmarks are complete measurements and five are historical measurements or estimates, as marked in the `measurement` column.
