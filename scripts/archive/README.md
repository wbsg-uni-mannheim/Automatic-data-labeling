# Helper Scripts

Scripts that the entry points in `scripts/labeling/`, `scripts/training/`, and `scripts/post_processing/` call, or that built the released training sets.

| Directory | Contents |
|---|---|
| `ditto_internal/` | Ditto training and evaluation called by `scripts/training/train_ditto.py`, and the conversion of labeled pairs into the Ditto input format. |
| `evaluation/` | `run_benchmark_batch_eval.py`, used by the labeling workflows, and `direct_llm_labeling.py`, which labels the test sets zero-shot with the teachers (Table 7). |
| `labeling_helpers/` | Batch runner over `configs/labeling/benchmarks_active.yaml` and the random pools that complement the selected pairs. |
| `qwen_internal/` | Conversion, evaluation, and environment setup for the Qwen3 students. |
| `result_builders/` | Cut the acquired sets to the benchmark training size (`build_benchmark_size_subsets.py`) and to the label budgets of Figure 2 (`build_learning_curve_subsets_all.py`). |
