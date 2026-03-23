# Auto-Label Training Set Profile

Stand: 2026-03-10

## Scope

- Profilierung nur fuer Auto-Label-Rohdaten, die aktuell wirklich im Repository liegen.
- Die Analyse nutzt exakte Ueberlappung auf dem kanonischen Paar-Key `id_left#id_right` sowie einen Cluster-basierten Konsistenzcheck (`cluster_id_left == cluster_id_right`).
- Manuelle Fehlerannotation nach Ralphs Schema ist damit noch nicht ersetzt; sie bleibt ein naechster Schritt.

## Verfuegbare generierte Sets

| Benchmark | Profil | Generated File | Analysis Source | Summary File |
| --- | --- | --- | --- | --- |
| abt-buy | large | data/auto_label_v1/abt_buy_local_test/profiles/large/active_labels_latest_abt-buy_large_train.json.gz | data/auto_label_v1/abt_buy_local_test/profiles/large/labels_final.csv | data/auto_label_v1/abt_buy_local_test/summary.json |
| abt-buy | medium | data/auto_label_v1/abt_buy_local_test/profiles/medium/active_labels_latest_abt-buy_medium_train.json.gz | data/auto_label_v1/abt_buy_local_test/profiles/medium/labels_final.csv | data/auto_label_v1/abt_buy_local_test/summary.json |
| abt-buy | small | data/auto_label_v1/abt_buy_local_test/profiles/small/active_labels_latest_abt-buy_small_train.json.gz | data/auto_label_v1/abt_buy_local_test/profiles/small/labels_final.csv | data/auto_label_v1/abt_buy_local_test/summary.json |
| amazon-google | all | data/auto_label_v1/benchmark_amazon-google_20260302_113733/profiles/all/active_labels_latest_amazon-google_all_train.json.gz | data/auto_label_v1/benchmark_amazon-google_20260302_113733/profiles/all/labels_final.csv | data/auto_label_v1/benchmark_amazon-google_20260302_113733/summary.json |
| dblp-acm | all | data/auto_label_v1/benchmark_dblp-acm_20260302_113733/profiles/all/active_labels_latest_dblp-acm_all_train.json.gz | data/auto_label_v1/benchmark_dblp-acm_20260302_113733/profiles/all/labels_final.csv | data/auto_label_v1/benchmark_dblp-acm_20260302_113733/summary.json |
| dblp-scholar | all | data/auto_label_v1/benchmark_dblp-scholar_20260302_113733/profiles/all/active_labels_latest_dblp-scholar_all_train.json.gz | data/auto_label_v1/benchmark_dblp-scholar_20260302_113733/profiles/all/labels_final.csv | data/auto_label_v1/benchmark_dblp-scholar_20260302_113733/summary.json |
| walmart-amazon | all | data/auto_label_v1/benchmark_walmart-amazon_20260302_113733/profiles/all/active_labels_latest_walmart-amazon_all_train.json.gz | data/auto_label_v1/benchmark_walmart-amazon_20260302_113733/profiles/all/labels_final.csv | data/auto_label_v1/benchmark_walmart-amazon_20260302_113733/summary.json |

## Wichtigste Befunde

- `abt-buy/large`: 70.54% Train-Overlap, 17.60% Valid-Leakage, 17.30% Test-Leakage, 12.16% neue Paare, Positive Rate neu vs. Train-Overlap 0.015 vs. 0.186, Cluster-Fehlerrate 3.100%, mean NN cosine novel->train 0.888, coverage 0.379.
- `abt-buy/medium`: 71.43% Train-Overlap, 17.40% Valid-Leakage, 17.37% Test-Leakage, 11.20% neue Paare, Positive Rate neu vs. Train-Overlap 0.024 vs. 0.231, Cluster-Fehlerrate 3.833%, mean NN cosine novel->train 0.891, coverage 0.298.
- `abt-buy/small`: 71.00% Train-Overlap, 16.80% Valid-Leakage, 17.30% Test-Leakage, 11.70% neue Paare, Positive Rate neu vs. Train-Overlap 0.043 vs. 0.224, Cluster-Fehlerrate 3.800%, mean NN cosine novel->train 0.891, coverage 0.185.
- `amazon-google/all`: 56.52% Train-Overlap, 14.23% Valid-Leakage, 12.46% Test-Leakage, 31.02% neue Paare, Positive Rate neu vs. Train-Overlap 0.053 vs. 0.283, Cluster-Fehlerrate 16.823%, mean NN cosine novel->train 0.843, coverage 0.651.
- `dblp-acm/all`: 42.99% Train-Overlap, 10.69% Valid-Leakage, 9.21% Test-Leakage, 47.80% neue Paare, Positive Rate neu vs. Train-Overlap 0.001 vs. 0.453, Cluster-Fehlerrate 2.076%, mean NN cosine novel->train 0.774, coverage 0.822.
- `dblp-scholar/all`: 33.93% Train-Overlap, 8.48% Valid-Leakage, 5.24% Test-Leakage, 60.83% neue Paare, Positive Rate neu vs. Train-Overlap 0.007 vs. 0.455, Cluster-Fehlerrate 6.730%, mean NN cosine novel->train 0.743, coverage 0.767.
- `walmart-amazon/all`: 66.00% Train-Overlap, 16.40% Valid-Leakage, 9.34% Test-Leakage, 24.66% neue Paare, Positive Rate neu vs. Train-Overlap 0.007 vs. 0.139, Cluster-Fehlerrate 46.654%, mean NN cosine novel->train 0.873, coverage 0.683.

## abt-buy / large

- Exakte Ueberlappung mit offiziellen Trainingspaaren: 3527/5000 (70.54%).
- Leakage in offizielle Validierungs-/Test-Splits: 880/5000 bzw. 865/5000.
- Wirklich neue Paare: 608/5000 (12.16%).
- Positive Rate neu vs. train-overlap: 0.015 vs. 0.186.
- Mittlere Text-Jaccard neu vs. train-overlap: 0.147 vs. 0.171.
- Cluster-basierte Inkonsistenzrate im generierten Set: 3.100%.
- Embedding-Raum: mean NN cosine `novel -> official train` 0.888, precision 0.957, coverage 0.379.

| Subset | Rows | Pos | Neg | Pos Rate | Left Cov | Right Cov | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all | 5000 | 800 | 4200 | 0.160 | 0.933 | 0.918 | 0.168 | 0.031 |
| generated_overlap_train | 3527 | 655 | 2872 | 0.186 | 0.893 | 0.881 | 0.171 | 0.033 |
| generated_overlap_valid | 880 | 154 | 726 | 0.175 | 0.499 | 0.487 | 0.170 | 0.033 |
| generated_overlap_test | 865 | 136 | 729 | 0.157 | 0.473 | 0.449 | 0.171 | 0.037 |
| generated_novel | 608 | 9 | 599 | 0.015 | 0.281 | 0.224 | 0.147 | 0.012 |

| Official Split | Group | Overlap Pairs | Rate vs Generated | Rate vs Split | Label Agree | Label Conflict |
| --- | --- | --- | --- | --- | --- | --- |
| abt-buy-gs.json.gz | test | 865 | 0.173 | 0.451 | 834 | 31 |
| abt-buy-train.json | train | 3527 | 0.705 | 0.461 | 3413 | 114 |
| abt-buy-valid.csv | valid | 880 | 0.176 | 0.459 | 0 | 0 |

| Official Split | Rows | Pos | Neg | Pos Rate | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- |
| abt-buy-gs.json.gz | 1916 | 206 | 1710 | 0.108 | 0.140 | 0.001 |
| abt-buy-train.json | 7659 | 822 | 6837 | 0.107 | 0.139 | 0.002 |
| abt-buy-valid.csv | 1916 | 0 | 0 | 0.000 | 1.000 | 0.000 |

Embedding sample sizes: generated_all=1500, generated_novel=608, official_train=1500, official_valid=1500, official_test=1500, k=5.

| Embedding Subset | Mean NN Cosine | Median NN Cosine | P90 NN Cosine | Mean NN Distance |
| --- | --- | --- | --- | --- |
| generated_all | 0.920 | 0.945 | 0.980 | 0.080 |
| generated_novel | 0.922 | 0.962 | 0.982 | 0.078 |
| official_train | 0.898 | 0.918 | 0.968 | 0.102 |

| Embedding Comparison | Centroid Cosine | Mean NN Cosine | Mean NN Distance | Frechet | MMD-RBF | Precision | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all_vs_train | 0.990 | 0.921 | 0.079 | 0.074 | 0.004 | 0.985 | 0.891 |
| generated_novel_vs_train | 0.951 | 0.888 | 0.112 | 0.265 | 0.022 | 0.957 | 0.379 |
| generated_all_vs_valid | 0.988 | 0.920 | 0.080 | - | - | - | - |
| generated_all_vs_test | 0.989 | 0.919 | 0.081 | - | - | - | - |

Run summary:

```json
{
  "faiss": {
    "dedup_dropped_within_query": 0,
    "faiss_bottom_k": 2,
    "faiss_k": 20,
    "faiss_queries": 1025,
    "faiss_random_state": 42,
    "faiss_top_k": 18,
    "neighbor_side": "right",
    "query_side": "left",
    "raw_pairs": 20500,
    "same_source_id_pairs": 0,
    "same_source_id_rate": 0.0,
    "source_pair_dedup_after": 20500,
    "source_pair_dedup_before": 20500,
    "source_pair_dedup_dropped": 0,
    "unique_pairs_after_cap": 20500,
    "unique_pairs_before_cap": 20500
  },
  "final_neg": 4200,
  "final_pos": 800,
  "final_total": 5000,
  "seed_neg": 70,
  "seed_pos": 30,
  "seed_total": 100,
  "target_neg": 4200,
  "target_pos": 800,
  "target_total": 5000,
  "token_usage": {
    "active_completion_tokens": 39624,
    "active_prompt_tokens": 973945,
    "active_total_tokens": 1013569,
    "completion_tokens": 41512,
    "prompt_tokens": 1018406,
    "seed_completion_tokens": 1888,
    "seed_prompt_tokens": 44461,
    "seed_total_tokens": 46349,
    "total_tokens": 1059918
  }
}
```

## abt-buy / medium

- Exakte Ueberlappung mit offiziellen Trainingspaaren: 2143/3000 (71.43%).
- Leakage in offizielle Validierungs-/Test-Splits: 522/3000 bzw. 521/3000.
- Wirklich neue Paare: 336/3000 (11.20%).
- Positive Rate neu vs. train-overlap: 0.024 vs. 0.231.
- Mittlere Text-Jaccard neu vs. train-overlap: 0.152 vs. 0.172.
- Cluster-basierte Inkonsistenzrate im generierten Set: 3.833%.
- Embedding-Raum: mean NN cosine `novel -> official train` 0.891, precision 0.955, coverage 0.298.

| Subset | Rows | Pos | Neg | Pos Rate | Left Cov | Right Cov | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all | 3000 | 600 | 2400 | 0.200 | 0.840 | 0.821 | 0.170 | 0.038 |
| generated_overlap_train | 2143 | 494 | 1649 | 0.231 | 0.774 | 0.766 | 0.172 | 0.040 |
| generated_overlap_valid | 522 | 114 | 408 | 0.218 | 0.361 | 0.370 | 0.170 | 0.038 |
| generated_overlap_test | 521 | 98 | 423 | 0.188 | 0.351 | 0.340 | 0.171 | 0.044 |
| generated_novel | 336 | 8 | 328 | 0.024 | 0.195 | 0.161 | 0.152 | 0.018 |

| Official Split | Group | Overlap Pairs | Rate vs Generated | Rate vs Split | Label Agree | Label Conflict |
| --- | --- | --- | --- | --- | --- | --- |
| abt-buy-gs.json.gz | test | 521 | 0.174 | 0.272 | 499 | 22 |
| abt-buy-train.json | train | 2143 | 0.714 | 0.280 | 2058 | 85 |
| abt-buy-valid.csv | valid | 522 | 0.174 | 0.272 | 0 | 0 |

| Official Split | Rows | Pos | Neg | Pos Rate | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- |
| abt-buy-gs.json.gz | 1916 | 206 | 1710 | 0.108 | 0.140 | 0.001 |
| abt-buy-train.json | 7659 | 822 | 6837 | 0.107 | 0.139 | 0.002 |
| abt-buy-valid.csv | 1916 | 0 | 0 | 0.000 | 1.000 | 0.000 |

Embedding sample sizes: generated_all=1500, generated_novel=336, official_train=1500, official_valid=1500, official_test=1500, k=5.

| Embedding Subset | Mean NN Cosine | Median NN Cosine | P90 NN Cosine | Mean NN Distance |
| --- | --- | --- | --- | --- |
| generated_all | 0.917 | 0.946 | 0.978 | 0.083 |
| generated_novel | 0.905 | 0.950 | 0.979 | 0.095 |
| official_train | 0.898 | 0.918 | 0.968 | 0.102 |

| Embedding Comparison | Centroid Cosine | Mean NN Cosine | Mean NN Distance | Frechet | MMD-RBF | Precision | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all_vs_train | 0.992 | 0.920 | 0.080 | 0.071 | 0.003 | 0.978 | 0.897 |
| generated_novel_vs_train | 0.950 | 0.891 | 0.109 | 0.309 | 0.023 | 0.955 | 0.298 |
| generated_all_vs_valid | 0.990 | 0.920 | 0.080 | - | - | - | - |
| generated_all_vs_test | 0.991 | 0.917 | 0.083 | - | - | - | - |

Run summary:

```json
{
  "faiss": {
    "dedup_dropped_within_query": 0,
    "faiss_bottom_k": 2,
    "faiss_k": 20,
    "faiss_queries": 1025,
    "faiss_random_state": 42,
    "faiss_top_k": 18,
    "neighbor_side": "right",
    "query_side": "left",
    "raw_pairs": 20500,
    "same_source_id_pairs": 0,
    "same_source_id_rate": 0.0,
    "source_pair_dedup_after": 20500,
    "source_pair_dedup_before": 20500,
    "source_pair_dedup_dropped": 0,
    "unique_pairs_after_cap": 20500,
    "unique_pairs_before_cap": 20500
  },
  "final_neg": 4200,
  "final_pos": 800,
  "final_total": 5000,
  "seed_neg": 70,
  "seed_pos": 30,
  "seed_total": 100,
  "target_neg": 4200,
  "target_pos": 800,
  "target_total": 5000,
  "token_usage": {
    "active_completion_tokens": 39624,
    "active_prompt_tokens": 973945,
    "active_total_tokens": 1013569,
    "completion_tokens": 41512,
    "prompt_tokens": 1018406,
    "seed_completion_tokens": 1888,
    "seed_prompt_tokens": 44461,
    "seed_total_tokens": 46349,
    "total_tokens": 1059918
  }
}
```

## abt-buy / small

- Exakte Ueberlappung mit offiziellen Trainingspaaren: 710/1000 (71.00%).
- Leakage in offizielle Validierungs-/Test-Splits: 168/1000 bzw. 173/1000.
- Wirklich neue Paare: 117/1000 (11.70%).
- Positive Rate neu vs. train-overlap: 0.043 vs. 0.224.
- Mittlere Text-Jaccard neu vs. train-overlap: 0.156 vs. 0.175.
- Cluster-basierte Inkonsistenzrate im generierten Set: 3.800%.
- Embedding-Raum: mean NN cosine `novel -> official train` 0.891, precision 0.966, coverage 0.185.

| Subset | Rows | Pos | Neg | Pos Rate | Left Cov | Right Cov | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all | 1000 | 200 | 800 | 0.200 | 0.530 | 0.520 | 0.172 | 0.038 |
| generated_overlap_train | 710 | 159 | 551 | 0.224 | 0.438 | 0.430 | 0.175 | 0.034 |
| generated_overlap_valid | 168 | 36 | 132 | 0.214 | 0.142 | 0.143 | 0.173 | 0.030 |
| generated_overlap_test | 173 | 36 | 137 | 0.208 | 0.147 | 0.147 | 0.169 | 0.052 |
| generated_novel | 117 | 5 | 112 | 0.043 | 0.089 | 0.081 | 0.156 | 0.043 |

| Official Split | Group | Overlap Pairs | Rate vs Generated | Rate vs Split | Label Agree | Label Conflict |
| --- | --- | --- | --- | --- | --- | --- |
| abt-buy-gs.json.gz | test | 173 | 0.173 | 0.090 | 165 | 8 |
| abt-buy-train.json | train | 710 | 0.710 | 0.093 | 687 | 23 |
| abt-buy-valid.csv | valid | 168 | 0.168 | 0.088 | 0 | 0 |

| Official Split | Rows | Pos | Neg | Pos Rate | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- |
| abt-buy-gs.json.gz | 1916 | 206 | 1710 | 0.108 | 0.140 | 0.001 |
| abt-buy-train.json | 7659 | 822 | 6837 | 0.107 | 0.139 | 0.002 |
| abt-buy-valid.csv | 1916 | 0 | 0 | 0.000 | 1.000 | 0.000 |

Embedding sample sizes: generated_all=1000, generated_novel=117, official_train=1500, official_valid=1500, official_test=1500, k=5.

| Embedding Subset | Mean NN Cosine | Median NN Cosine | P90 NN Cosine | Mean NN Distance |
| --- | --- | --- | --- | --- |
| generated_all | 0.905 | 0.933 | 0.975 | 0.095 |
| generated_novel | 0.864 | 0.912 | 0.967 | 0.136 |
| official_train | 0.898 | 0.918 | 0.968 | 0.102 |

| Embedding Comparison | Centroid Cosine | Mean NN Cosine | Mean NN Distance | Frechet | MMD-RBF | Precision | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all_vs_train | 0.993 | 0.916 | 0.084 | 0.079 | 0.002 | 0.981 | 0.842 |
| generated_novel_vs_train | 0.953 | 0.891 | 0.109 | 0.419 | 0.019 | 0.966 | 0.185 |
| generated_all_vs_valid | 0.991 | 0.919 | 0.081 | - | - | - | - |
| generated_all_vs_test | 0.993 | 0.914 | 0.086 | - | - | - | - |

Run summary:

```json
{
  "faiss": {
    "dedup_dropped_within_query": 0,
    "faiss_bottom_k": 2,
    "faiss_k": 20,
    "faiss_queries": 1025,
    "faiss_random_state": 42,
    "faiss_top_k": 18,
    "neighbor_side": "right",
    "query_side": "left",
    "raw_pairs": 20500,
    "same_source_id_pairs": 0,
    "same_source_id_rate": 0.0,
    "source_pair_dedup_after": 20500,
    "source_pair_dedup_before": 20500,
    "source_pair_dedup_dropped": 0,
    "unique_pairs_after_cap": 20500,
    "unique_pairs_before_cap": 20500
  },
  "final_neg": 4200,
  "final_pos": 800,
  "final_total": 5000,
  "seed_neg": 70,
  "seed_pos": 30,
  "seed_total": 100,
  "target_neg": 4200,
  "target_pos": 800,
  "target_total": 5000,
  "token_usage": {
    "active_completion_tokens": 39624,
    "active_prompt_tokens": 973945,
    "active_total_tokens": 1013569,
    "completion_tokens": 41512,
    "prompt_tokens": 1018406,
    "seed_completion_tokens": 1888,
    "seed_prompt_tokens": 44461,
    "seed_total_tokens": 46349,
    "total_tokens": 1059918
  }
}
```

## amazon-google / all

- Exakte Ueberlappung mit offiziellen Trainingspaaren: 4341/7680 (56.52%).
- Leakage in offizielle Validierungs-/Test-Splits: 1093/7680 bzw. 957/7680.
- Wirklich neue Paare: 2382/7680 (31.02%).
- Positive Rate neu vs. train-overlap: 0.053 vs. 0.283.
- Mittlere Text-Jaccard neu vs. train-overlap: 0.198 vs. 0.312.
- Cluster-basierte Inkonsistenzrate im generierten Set: 16.823%.
- Embedding-Raum: mean NN cosine `novel -> official train` 0.843, precision 0.959, coverage 0.651.

| Subset | Rows | Pos | Neg | Pos Rate | Left Cov | Right Cov | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all | 7680 | 1580 | 6100 | 0.206 | 0.959 | 0.881 | 0.276 | 0.168 |
| generated_overlap_train | 4341 | 1230 | 3111 | 0.283 | 0.909 | 0.816 | 0.312 | 0.192 |
| generated_overlap_valid | 1093 | 315 | 778 | 0.288 | 0.500 | 0.371 | 0.312 | 0.213 |
| generated_overlap_test | 957 | 224 | 733 | 0.234 | 0.415 | 0.308 | 0.305 | 0.177 |
| generated_novel | 2382 | 126 | 2256 | 0.053 | 0.595 | 0.469 | 0.198 | 0.121 |

| Official Split | Group | Overlap Pairs | Rate vs Generated | Rate vs Split | Label Agree | Label Conflict |
| --- | --- | --- | --- | --- | --- | --- |
| amazon-google-gs.json | test | 957 | 0.125 | 0.417 | 815 | 142 |
| amazon-google-gs.json.gz | test | 957 | 0.125 | 0.417 | 815 | 142 |
| amazon-google-train-validation.csv | valid | 1093 | 0.142 | 0.477 | 910 | 183 |
| amazon-google-train.json | train | 4341 | 0.565 | 0.474 | 3639 | 702 |
| amazon-google-valid.csv | valid | 1093 | 0.142 | 0.477 | 0 | 0 |

| Official Split | Rows | Pos | Neg | Pos Rate | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- |
| amazon-google-gs.json | 2293 | 234 | 2059 | 0.102 | 0.257 | 0.101 |
| amazon-google-gs.json.gz | 2293 | 234 | 2059 | 0.102 | 0.257 | 0.101 |
| amazon-google-train-validation.csv | 2293 | 234 | 2059 | 0.102 | 0.251 | 0.104 |
| amazon-google-train.json | 9167 | 933 | 8234 | 0.102 | 0.255 | 0.094 |
| amazon-google-valid.csv | 2293 | 0 | 0 | 0.000 | 1.000 | 0.000 |

Embedding sample sizes: generated_all=1500, generated_novel=1500, official_train=1500, official_valid=1500, official_test=1500, k=5.

| Embedding Subset | Mean NN Cosine | Median NN Cosine | P90 NN Cosine | Mean NN Distance |
| --- | --- | --- | --- | --- |
| generated_all | 0.888 | 0.913 | 0.971 | 0.112 |
| generated_novel | 0.902 | 0.925 | 0.981 | 0.098 |
| official_train | 0.865 | 0.890 | 0.967 | 0.135 |

| Embedding Comparison | Centroid Cosine | Mean NN Cosine | Mean NN Distance | Frechet | MMD-RBF | Precision | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all_vs_train | 0.997 | 0.879 | 0.121 | 0.062 | 0.001 | 0.973 | 0.927 |
| generated_novel_vs_train | 0.988 | 0.843 | 0.157 | 0.124 | 0.006 | 0.959 | 0.651 |
| generated_all_vs_valid | 0.997 | 0.879 | 0.121 | - | - | - | - |
| generated_all_vs_test | 0.995 | 0.874 | 0.126 | - | - | - | - |

Run summary:

```json
{
  "faiss": {
    "dedup_dropped_within_query": 0,
    "faiss_bottom_k": 2,
    "faiss_k": 20,
    "faiss_queries": 1225,
    "faiss_random_state": 42,
    "faiss_top_k": 18,
    "neighbor_side": "right",
    "query_side": "left",
    "raw_pairs": 24500,
    "same_source_id_pairs": 0,
    "same_source_id_rate": 0.0,
    "source_pair_dedup_after": 24500,
    "source_pair_dedup_before": 24500,
    "source_pair_dedup_dropped": 0,
    "unique_pairs_after_cap": 24500,
    "unique_pairs_before_cap": 24500
  },
  "final_neg": 6100,
  "final_pos": 900,
  "final_total": 7000,
  "seed_neg": 70,
  "seed_pos": 30,
  "seed_total": 100,
  "target_neg": 6100,
  "target_pos": 900,
  "target_total": 7000,
  "token_usage": {
    "active_completion_tokens": 60640,
    "active_prompt_tokens": 986493,
    "active_total_tokens": 1047133,
    "completion_tokens": 63120,
    "prompt_tokens": 1026510,
    "seed_completion_tokens": 2480,
    "seed_prompt_tokens": 40017,
    "seed_total_tokens": 42497,
    "total_tokens": 1089630
  }
}
```

## dblp-acm / all

- Exakte Ueberlappung mit offiziellen Trainingspaaren: 4059/9442 (42.99%).
- Leakage in offizielle Validierungs-/Test-Splits: 1009/9442 bzw. 870/9442.
- Wirklich neue Paare: 4513/9442 (47.80%).
- Positive Rate neu vs. train-overlap: 0.001 vs. 0.453.
- Mittlere Text-Jaccard neu vs. train-overlap: 0.151 vs. 0.450.
- Cluster-basierte Inkonsistenzrate im generierten Set: 2.076%.
- Embedding-Raum: mean NN cosine `novel -> official train` 0.774, precision 0.924, coverage 0.822.

| Subset | Rows | Pos | Neg | Pos Rate | Left Cov | Right Cov | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all | 9442 | 2142 | 7300 | 0.227 | 0.987 | 0.998 | 0.301 | 0.021 |
| generated_overlap_train | 4059 | 1840 | 2219 | 0.453 | 0.900 | 0.947 | 0.450 | 0.033 |
| generated_overlap_valid | 1009 | 459 | 550 | 0.455 | 0.347 | 0.363 | 0.456 | 0.035 |
| generated_overlap_test | 870 | 296 | 574 | 0.340 | 0.273 | 0.294 | 0.384 | 0.048 |
| generated_novel | 4513 | 6 | 4507 | 0.001 | 0.763 | 0.718 | 0.151 | 0.005 |

| Official Split | Group | Overlap Pairs | Rate vs Generated | Rate vs Split | Label Agree | Label Conflict |
| --- | --- | --- | --- | --- | --- | --- |
| dblp-acm-gs.json.gz | test | 870 | 0.092 | 0.352 | 847 | 23 |
| dblp-acm-train.json.gz | train | 4059 | 0.430 | 0.410 | 3974 | 85 |
| dblp-acm-valid.csv | valid | 1009 | 0.107 | 0.408 | 990 | 19 |

| Official Split | Rows | Pos | Neg | Pos Rate | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- |
| dblp-acm-gs.json.gz | 2473 | 444 | 2029 | 0.180 | 0.248 | 0.011 |
| dblp-acm-train.json.gz | 9890 | 1776 | 8114 | 0.180 | 0.247 | 0.007 |
| dblp-acm-valid.csv | 2473 | 444 | 2029 | 0.180 | 0.249 | 0.009 |

Embedding sample sizes: generated_all=1500, generated_novel=1500, official_train=1500, official_valid=1500, official_test=1500, k=5.

| Embedding Subset | Mean NN Cosine | Median NN Cosine | P90 NN Cosine | Mean NN Distance |
| --- | --- | --- | --- | --- |
| generated_all | 0.810 | 0.831 | 0.900 | 0.190 |
| generated_novel | 0.809 | 0.819 | 0.899 | 0.191 |
| official_train | 0.793 | 0.807 | 0.878 | 0.207 |

| Embedding Comparison | Centroid Cosine | Mean NN Cosine | Mean NN Distance | Frechet | MMD-RBF | Precision | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all_vs_train | 0.997 | 0.801 | 0.199 | 0.064 | 0.002 | 0.937 | 0.903 |
| generated_novel_vs_train | 0.995 | 0.774 | 0.226 | 0.081 | 0.003 | 0.924 | 0.822 |
| generated_all_vs_valid | 0.997 | 0.795 | 0.205 | - | - | - | - |
| generated_all_vs_test | 0.996 | 0.798 | 0.202 | - | - | - | - |

Run summary:

```json
{
  "faiss": {
    "dedup_dropped_within_query": 0,
    "faiss_bottom_k": 2,
    "faiss_k": 20,
    "faiss_queries": 2306,
    "faiss_random_state": 42,
    "faiss_top_k": 18,
    "neighbor_side": "right",
    "query_side": "left",
    "raw_pairs": 46120,
    "same_source_id_pairs": 0,
    "same_source_id_rate": 0.0,
    "source_pair_dedup_after": 46120,
    "source_pair_dedup_before": 46120,
    "source_pair_dedup_dropped": 0,
    "unique_pairs_after_cap": 46120,
    "unique_pairs_before_cap": 46120
  },
  "final_neg": 7300,
  "final_pos": 1700,
  "final_total": 9000,
  "seed_neg": 70,
  "seed_pos": 30,
  "seed_total": 100,
  "target_neg": 7300,
  "target_pos": 1700,
  "target_total": 9000,
  "token_usage": {
    "active_completion_tokens": 74736,
    "active_prompt_tokens": 1621547,
    "active_total_tokens": 1696283,
    "completion_tokens": 76608,
    "prompt_tokens": 1660767,
    "seed_completion_tokens": 1872,
    "seed_prompt_tokens": 39220,
    "seed_total_tokens": 41092,
    "total_tokens": 1737375
  }
}
```

## dblp-scholar / all

- Exakte Ueberlappung mit offiziellen Trainingspaaren: 7740/22809 (33.93%).
- Leakage in offizielle Validierungs-/Test-Splits: 1934/22809 bzw. 1195/22809.
- Wirklich neue Paare: 13874/22809 (60.83%).
- Positive Rate neu vs. train-overlap: 0.007 vs. 0.455.
- Mittlere Text-Jaccard neu vs. train-overlap: 0.133 vs. 0.380.
- Cluster-basierte Inkonsistenzrate im generierten Set: 6.730%.
- Embedding-Raum: mean NN cosine `novel -> official train` 0.743, precision 0.923, coverage 0.767.

| Subset | Rows | Pos | Neg | Pos Rate | Left Cov | Right Cov | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all | 22809 | 4000 | 18809 | 0.175 | 0.994 | 0.766 | 0.227 | 0.067 |
| generated_overlap_train | 7740 | 3520 | 4220 | 0.455 | 0.888 | 0.598 | 0.380 | 0.136 |
| generated_overlap_valid | 1934 | 893 | 1041 | 0.462 | 0.474 | 0.185 | 0.379 | 0.140 |
| generated_overlap_test | 1195 | 386 | 809 | 0.323 | 0.296 | 0.110 | 0.333 | 0.138 |
| generated_novel | 13874 | 94 | 13780 | 0.007 | 0.940 | 0.511 | 0.133 | 0.023 |

| Official Split | Group | Overlap Pairs | Rate vs Generated | Rate vs Split | Label Agree | Label Conflict |
| --- | --- | --- | --- | --- | --- | --- |
| dblp-scholar-gs.json.gz | test | 1195 | 0.052 | 0.208 | 1095 | 100 |
| dblp-scholar-train.json.gz | train | 7740 | 0.339 | 0.337 | 7028 | 712 |
| dblp-scholar-valid.csv | valid | 1934 | 0.085 | 0.337 | 1748 | 186 |

| Official Split | Rows | Pos | Neg | Pos Rate | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- |
| dblp-scholar-gs.json.gz | 5742 | 1070 | 4672 | 0.186 | 0.245 | 0.065 |
| dblp-scholar-train.json.gz | 22965 | 4277 | 18688 | 0.186 | 0.246 | 0.064 |
| dblp-scholar-valid.csv | 5742 | 1070 | 4672 | 0.186 | 0.247 | 0.064 |

Embedding sample sizes: generated_all=1500, generated_novel=1500, official_train=1500, official_valid=1500, official_test=1500, k=5.

| Embedding Subset | Mean NN Cosine | Median NN Cosine | P90 NN Cosine | Mean NN Distance |
| --- | --- | --- | --- | --- |
| generated_all | 0.790 | 0.809 | 0.896 | 0.210 |
| generated_novel | 0.789 | 0.802 | 0.896 | 0.211 |
| official_train | 0.782 | 0.796 | 0.907 | 0.218 |

| Embedding Comparison | Centroid Cosine | Mean NN Cosine | Mean NN Distance | Frechet | MMD-RBF | Precision | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all_vs_train | 0.995 | 0.757 | 0.243 | 0.082 | 0.003 | 0.933 | 0.866 |
| generated_novel_vs_train | 0.993 | 0.743 | 0.257 | 0.098 | 0.003 | 0.923 | 0.767 |
| generated_all_vs_valid | 0.996 | 0.760 | 0.240 | - | - | - | - |
| generated_all_vs_test | 0.993 | 0.745 | 0.255 | - | - | - | - |

Run summary:

```json
{
  "faiss": {
    "dedup_dropped_within_query": 0,
    "faiss_bottom_k": 2,
    "faiss_k": 20,
    "faiss_queries": 2504,
    "faiss_random_state": 42,
    "faiss_top_k": 18,
    "neighbor_side": "right",
    "query_side": "left",
    "raw_pairs": 50080,
    "same_source_id_pairs": 0,
    "same_source_id_rate": 0.0,
    "source_pair_dedup_after": 50080,
    "source_pair_dedup_before": 50080,
    "source_pair_dedup_dropped": 0,
    "unique_pairs_after_cap": 50080,
    "unique_pairs_before_cap": 50080
  },
  "final_neg": 16000,
  "final_pos": 4000,
  "final_total": 20000,
  "seed_neg": 70,
  "seed_pos": 30,
  "seed_total": 100,
  "target_neg": 16000,
  "target_pos": 4000,
  "target_total": 20000,
  "token_usage": {
    "active_completion_tokens": 181672,
    "active_prompt_tokens": 3754143,
    "active_total_tokens": 3935815,
    "completion_tokens": 183944,
    "prompt_tokens": 3799106,
    "seed_completion_tokens": 2272,
    "seed_prompt_tokens": 44963,
    "seed_total_tokens": 47235,
    "total_tokens": 3983050
  }
}
```

## walmart-amazon / all

- Exakte Ueberlappung mit offiziellen Trainingspaaren: 4981/7547 (66.00%).
- Leakage in offizielle Validierungs-/Test-Splits: 1238/7547 bzw. 705/7547.
- Wirklich neue Paare: 1861/7547 (24.66%).
- Positive Rate neu vs. train-overlap: 0.007 vs. 0.139.
- Mittlere Text-Jaccard neu vs. train-overlap: 0.223 vs. 0.336.
- Cluster-basierte Inkonsistenzrate im generierten Set: 46.654%.
- Embedding-Raum: mean NN cosine `novel -> official train` 0.873, precision 0.974, coverage 0.683.

| Subset | Rows | Pos | Neg | Pos Rate | Left Cov | Right Cov | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all | 7547 | 750 | 6797 | 0.099 | 0.925 | 0.795 | 0.308 | 0.467 |
| generated_overlap_train | 4981 | 691 | 4290 | 0.139 | 0.903 | 0.743 | 0.336 | 0.454 |
| generated_overlap_valid | 1238 | 178 | 1060 | 0.144 | 0.476 | 0.236 | 0.332 | 0.452 |
| generated_overlap_test | 705 | 46 | 659 | 0.065 | 0.239 | 0.121 | 0.339 | 0.516 |
| generated_novel | 1861 | 13 | 1848 | 0.007 | 0.371 | 0.236 | 0.223 | 0.481 |

| Official Split | Group | Overlap Pairs | Rate vs Generated | Rate vs Split | Label Agree | Label Conflict |
| --- | --- | --- | --- | --- | --- | --- |
| walmart-amazon-gs.json.gz | test | 705 | 0.093 | 0.344 | 689 | 16 |
| walmart-amazon-train.json.gz | train | 4981 | 0.660 | 0.608 | 4821 | 160 |
| walmart-amazon-valid.csv | valid | 1238 | 0.164 | 0.604 | 1195 | 43 |

| Official Split | Rows | Pos | Neg | Pos Rate | Mean Jaccard | Cluster Err Rate |
| --- | --- | --- | --- | --- | --- | --- |
| walmart-amazon-gs.json.gz | 2049 | 193 | 1856 | 0.094 | 0.310 | 0.484 |
| walmart-amazon-train.json.gz | 8193 | 769 | 7424 | 0.094 | 0.309 | 0.467 |
| walmart-amazon-valid.csv | 2049 | 193 | 1856 | 0.094 | 0.307 | 0.462 |

Embedding sample sizes: generated_all=1500, generated_novel=1500, official_train=1500, official_valid=1500, official_test=1277, k=5.

| Embedding Subset | Mean NN Cosine | Median NN Cosine | P90 NN Cosine | Mean NN Distance |
| --- | --- | --- | --- | --- |
| generated_all | 0.888 | 0.928 | 0.977 | 0.112 |
| generated_novel | 0.922 | 0.946 | 0.985 | 0.078 |
| official_train | 0.869 | 0.912 | 0.974 | 0.131 |

| Embedding Comparison | Centroid Cosine | Mean NN Cosine | Mean NN Distance | Frechet | MMD-RBF | Precision | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generated_all_vs_train | 0.997 | 0.889 | 0.111 | 0.059 | 0.001 | 0.973 | 0.939 |
| generated_novel_vs_train | 0.982 | 0.873 | 0.127 | 0.144 | 0.007 | 0.974 | 0.683 |
| generated_all_vs_valid | 0.997 | 0.887 | 0.113 | - | - | - | - |
| generated_all_vs_test | 0.984 | 0.852 | 0.148 | - | - | - | - |

Run summary:

```json
{
  "faiss": {
    "dedup_dropped_within_query": 0,
    "faiss_bottom_k": 2,
    "faiss_k": 20,
    "faiss_queries": 1579,
    "faiss_random_state": 42,
    "faiss_top_k": 18,
    "neighbor_side": "right",
    "query_side": "left",
    "raw_pairs": 31580,
    "same_source_id_pairs": 0,
    "same_source_id_rate": 0.0,
    "source_pair_dedup_after": 31580,
    "source_pair_dedup_before": 31580,
    "source_pair_dedup_dropped": 0,
    "unique_pairs_after_cap": 31580,
    "unique_pairs_before_cap": 31580
  },
  "final_neg": 5250,
  "final_pos": 750,
  "final_total": 6000,
  "seed_neg": 77,
  "seed_pos": 23,
  "seed_total": 100,
  "target_neg": 5250,
  "target_pos": 750,
  "target_total": 6000,
  "token_usage": {
    "active_completion_tokens": 59576,
    "active_prompt_tokens": 1328967,
    "active_total_tokens": 1388543,
    "completion_tokens": 62776,
    "prompt_tokens": 1400219,
    "seed_completion_tokens": 3200,
    "seed_prompt_tokens": 71252,
    "seed_total_tokens": 74452,
    "total_tokens": 1462995
  }
}
```

## Interpretation

- Wenn die exakte Trainings-Ueberlappung sehr klein ist, erzeugt Auto-Labeling tatsaechlich neue Paarmengen statt nur offizielle Train-Paare wiederzuverwenden.
- Wenn `generated_novel` deutlich andere Positive-Raten oder Aehnlichkeiten als `generated_overlap_train` hat, dann veraendert Auto-Labeling die Trainingsverteilung substanziell.
- Embedding-Metriken ergaenzen die exakte Paar-Ueberlappung: hohe `coverage` bei gleichzeitig niedriger Exakt-Ueberlappung bedeutet semantisch aehnliche, aber nicht identische Trainingspaare.
- Hohe mean NN cosine zu `valid` oder `test` ist ein Soft-Leakage-Signal, auch wenn kein exakter Paar-Key ueberschneidet.
- Cluster-basierte Inkonsistenzen sind ein erster harter Fehlerindikator. Fuer das Paper sollte darauf noch eine manuelle Typisierung folgen.
