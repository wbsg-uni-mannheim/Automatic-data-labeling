# Training Data

The machine-labeled training sets behind Tables 2 to 6, Table 10, and Figure 2 of the paper, laid out as:

```text
{benchmark}/{teacher}/{method}/{benchmark}_train.json.gz
{benchmark}/gpt-5.2/learning_curve/{method}/{benchmark}_N{budget}_train.json.gz
```

[`MANIFEST.csv`](MANIFEST.csv) lists every file with its row count, number of positives, SHA256 checksum, and the paper tables it belongs to. Each file is the exact input of the reported student runs.

## Coverage

All eight benchmarks: Abt-Buy, Walmart-Amazon, Dn7 Walmart-Amazon, WDC Products, billiger.de, DBLP-ACM, DBLP-Scholar, and Semi-HETER.

| Teacher | Similarity search | AL-ML | AL-Ditto | Post-processing variants | Benchmark pairs relabeled |
|---|---|---|---|---|---|
| **GPT-5.2** | 8 benchmarks | 8 | 8 | 5 variants × 8 benchmarks | 8 |
| **Qwen 3.6 Plus** | | | 8 | | |
| **Kimi K2.6** | | | 8 | | |

Qwen 3.6 Plus and Kimi K2.6 label only the AL-Ditto setting, for the teacher comparison (Table 4). The post-processing variants start from the GPT-5.2 AL-Ditto sets (Table 5).

For billiger.de, the `*_5897` directories hold the 5,897-pair sets, the size of the benchmark training set.

`learning_curve/` holds the label budgets of Figure 2. Where a curve point has the size of a Table 2 set, it uses that set from the method directory, as listed in [`../../results/figure2_label_budget.csv`](../../results/figure2_label_budget.csv).

## Methods

| Method directory | Description | Table |
|---|---|---|
| `similarity_search` | Similarity-search pair selection, labeled by the teacher | 2, 6, 10 |
| `active_learning_ml` | Active learning with a feature-based committee (AL-ML) | 2, 6, 10 |
| `active_learning_ditto` | Active learning with a bagged Ditto committee (AL-Ditto) | 2, 4, 5, 6, 10 |
| `benchmark_pairs_relabeled` | The benchmark training pairs with GPT-5.2 labels instead of the original labels | 3 |
| `active_learning_ditto_relabel` | Every pair relabeled with GPT-5-mini and the conservative review prompt | 5 |
| `active_learning_ditto_relabel_drop` | Pairs whose review decision differs from the teacher label are dropped | 5 |
| `active_learning_ditto_closure_drop` | Positive bridge edges and negatives that contradict the match closure are dropped | 5 |
| `active_learning_ditto_closure_and_relabel` | Pairs flagged by both closure and relabel-drop are dropped | 5 |
| `active_learning_ditto_closure_or_relabel` | Pairs flagged by either closure or relabel-drop are dropped | 5 |
| `*_5897` (billiger.de) | 5,897-pair sets | 4, 5, 6, 10 |
| `learning_curve/{method}` | Smaller label budgets of the three selection strategies | Fig. 2 |

The prompts are in [`../prompts/`](../prompts/).
