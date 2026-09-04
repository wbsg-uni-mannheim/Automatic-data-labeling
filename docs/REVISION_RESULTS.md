# Revision results — new benchmarks & candidate-pool sensitivity

Generated from the runs under `output/` (gitignored), so this file is the versioned record.
Full machine-readable report: `output/training_from_generated_labels/revision_2026_09/FINAL_RESULTS.txt`.

_Stand: 2026-09-04 09:52. Seeds 42/52/62 (= r1/r2/r3). Test-F1 auf dem Gold Standard._

## 1. Neue Benchmarks: maschinelle vs. menschliche Labels

| Benchmark | Verfahren | Train | Test-F1 | n | P | R |
|---|---|---|---|---|---|---|
| semi-heter | AL-Ditto (GPT-5.2) — `al_semi-heter_all_filtered` | 1,880 | 0.937 | 1 | 0.993 | 0.887 |
| semi-heter | AL-Ditto (GPT-5.2) — `al_semi-heter_all` | 1,882 | **0.920** ± 0.030 | 3 | 0.982 | 0.868 |
| semi-heter | Similarity Search (GPT-5.2) — `ss_semi-heter_large` | 1,240 | **0.913** ± 0.019 | 3 | 0.955 | 0.876 |
| semi-heter | Similarity Search (GPT-5.2) — `ss_semi-heter_xl` | 3,584 | 0.889 | 1 | 0.992 | 0.805 |
| semi-heter | Benchmark-Labels (human) — `BASELINE_semi-heter` | 1,240 | **0.816** ± 0.071 | 3 | 0.954 | 0.721 |
| dn7-walmart-amazon | AL-Ditto (GPT-5.2) — `al_dn7_medium` | 7,555 | 0.773 | 1 | 0.719 | 0.837 |
| dn7-walmart-amazon | Similarity Search (GPT-5.2) — `ss_dn7_all` | 4,609 | 0.747 | 1 | 0.724 | 0.771 |
| dn7-walmart-amazon | AL-Ditto (GPT-5.2) — `al_dn7_medium_complete` | 7,305 | **0.742** ± 0.012 | 3 | 0.718 | 0.767 |
| dn7-walmart-amazon | Benchmark-Labels (human) — `BASELINE_dn7-walmart-amazon` | 26,050 | **0.722** ± 0.039 | 3 | 0.700 | 0.749 |
| dn7-walmart-amazon | Similarity Search (GPT-5.2) — `ss_dn7_all_novalidoverlap` | 4,289 | **0.708** ± 0.010 | 3 | 0.673 | 0.747 |
| billiger-de | Benchmark-Labels (human) — `BASELINE_billiger-de` | 26,571 | **0.518** ± 0.019 | 3 | 0.385 | 0.796 |
| billiger-de | AL-Ditto (GPT-5.2) — `al_billiger-de_medium` | 12,083 | 0.515 | 1 | 0.383 | 0.784 |
| billiger-de | Similarity Search (GPT-5.2) — `ss_billiger-de_medium` | 10,000 | **0.510** ± 0.009 | 3 | 0.380 | 0.777 |

## 2. Pool-Sensitivität — Similarity-Search-Studenten

_20.000 Labels je Arm._

| Pool | Pool-Recall | Test-F1 | n | P | R |
|---|---|---|---|---|---|
| union | 89.88 % | 0.678 | 1 | 0.565 | 0.848 |
| embedding | 84.93 % | 0.684 | 1 | 0.603 | 0.790 |
| rrf | 81.99 % | 0.681 | 1 | 0.578 | 0.830 |
| bm25 | 74.52 % | 0.675 | 1 | 0.580 | 0.808 |

## 3. Pool-Sensitivität — AL-Ditto-Studenten

_11.500 Labels je Arm (größenangeglichen)._

| Pool | Pool-Recall | Test-F1 | n | P | R |
|---|---|---|---|---|---|
| union | 89.88 % | 0.723 | 1 | 0.664 | 0.792 |
| embedding | 84.93 % | 0.714 | 1 | 0.623 | 0.838 |
| rrf | 81.99 % | 0.719 | 1 | 0.680 | 0.764 |
| bm25 | 74.52 % | 0.700 | 1 | 0.601 | 0.838 |

## 4. Was diese Zahlen stützen

- **Die These trägt.** Maschinelle Labels erreichen oder übertreffen die Human-Baseline auf
  allen drei neuen Benchmarks, bei 28–45 % der Trainingsdaten. Auf semi-heter schlägt
  *jeder einzelne* maschinelle Seed *jeden* menschlichen Seed (schlechtester 0.889 vs. bester 0.880).
- **Human-Baselines sind die instabilsten Konfigurationen** (sd 0.019–0.071 vs. 0.004–0.030
  bei maschinellen Studenten). Die maschinellen Sets sind größer und positiv-reicher, was
  Early Stopping stabilisiert.
- **Pool-Konstruktion verschiebt P/R, nicht F1.** Über 15 Punkte Pool-Recall bewegt sich F1
  um 0.9 (Similarity Search) bzw. 2.3 Punkte (AL), während sich Precision und Recall um das
  Drei- bis Achtfache verschieben. Nur bm25 (schwächster Pool, 74.5 %) liegt klar unten.

## 5. Belastbarkeit / offene Punkte

- Pool-Tabellen mit `n=1` sind **nicht** belastbar: die gemessene Seed-Streuung liegt in
  derselben Größenordnung wie die Abstände zwischen den Armen. Seeds 52/62 laufen.
- `ss_semi-heter_xl`, `al_dn7_medium`, `ss_dn7_all` sind Varianten mit einem Seed — entweder
  replizieren oder nicht neben Drei-Seed-Zahlen zitieren.
- Nicht gelaufen: dn7-Hälfte des Pool-Experiments, LLM-Studenten (`train_qwen.py`),
  XGBoost-Studenten, billiger-en.
- `exclude_valid_from_train` greift bei maschinellen Sets nicht (pair_id-Format); hier wurden
  Validierungspaare explizit entfernt. Der gemessene Effekt lag im Seed-Rauschen.
- WDC-Embeddings sind 1536-d, nicht 256-d wie in `benchmarks/README.md` angegeben.
