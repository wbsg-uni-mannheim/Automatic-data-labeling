#!/usr/bin/env python3
"""Emit LaTeX tables from the plot CSVs with best=bold, second=underline per row.

Ranking is over F1 mean across the comparison columns in each benchmark row
(the baseline / "Benchmark set" column participates in the ranking).

Outputs .tex files into output/results_summary/latex/.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

OUT = Path("/work/aasteine/Automatic-data-labeling/output/results_summary")
LATEX_DIR = OUT / "latex"
LATEX_DIR.mkdir(parents=True, exist_ok=True)

BM_DISPLAY = {
    "abt-buy": r"\textsc{Abt-Buy}",
    "walmart-amazon": r"\textsc{Walmart-Amazon}",
    "wdc": r"\textsc{WDC}",
    "dblp-acm": r"\textsc{DBLP-ACM}",
    "dblp-scholar": r"\textsc{DBLP-Scholar}",
}
BM_ORDER = ["abt-buy", "walmart-amazon", "wdc", "dblp-acm", "dblp-scholar"]


def fmt(mean, std, rank):
    """rank: 1=best -> bold, 2=second -> underline, else plain."""
    if mean is None or mean == "":
        return "--"
    cell = f"{float(mean) * 100:.2f} $\\pm$ {float(std) * 100:.2f}"
    if rank == 1:
        return r"\textbf{" + cell + "}"
    if rank == 2:
        return r"\underline{" + cell + "}"
    return cell


def rank_row(values):
    """values: list of (key, mean) -> dict key->rank (1=best,2=second). Ties share a rank."""
    valid = [(k, m) for k, m in values if m not in (None, "")]
    ordered = sorted(valid, key=lambda x: -float(x[1]))
    ranks = {}
    last_val = None
    cur_rank = 0
    seen = 0
    for k, m in ordered:
        seen += 1
        if last_val is None or float(m) != float(last_val):
            cur_rank = seen
        ranks[k] = cur_rank
        last_val = m
    return ranks


def build_table(csv_path, col_specs, caption, label, group_field):
    """col_specs: list of (column_header, match_value_for_group_field).
    group_field: the CSV column whose value selects the column (e.g. 'method' or 'variant')."""
    rows = list(csv.DictReader(open(csv_path)))
    # index by benchmark -> {group_value: row}
    by_bm = defaultdict(dict)
    for r in rows:
        by_bm[r["benchmark"]][r[group_field]] = r

    n_cols = len(col_specs)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{" + caption + "}")
    lines.append(r"  \label{" + label + "}")
    lines.append(r"  \setlength{\tabcolsep}{3.5pt}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \begin{tabular}{l" + "r" * n_cols + "}")
    lines.append(r"    \toprule")
    header = "Benchmark & " + " & ".join(h for h, _ in col_specs) + r" \\"
    lines.append("    " + header)
    lines.append(r"    \midrule")
    for bm in BM_ORDER:
        cells = by_bm.get(bm, {})
        # gather (col_header, mean, std)
        vals = []
        for header, match in col_specs:
            r = cells.get(match)
            if r:
                vals.append((header, r.get("f1_mean", ""), r.get("f1_std", "")))
            else:
                vals.append((header, "", ""))
        ranks = rank_row([(h, m) for h, m, s in vals])
        rendered = [fmt(m, s, ranks.get(h, 99)) for h, m, s in vals]
        lines.append("    " + BM_DISPLAY[bm] + " & " + " & ".join(rendered) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def build_main_table_with_delta(csv_path):
    """Main comparison table matching the paper's Table 1 layout, with a
    'Best auto. Delta' column. Bold = overall best per row (Benchmark set
    included), underline = second best. Delta = best auto-labeled - benchmark."""
    rows = list(csv.DictReader(open(csv_path)))
    by_bm = defaultdict(dict)
    for r in rows:
        by_bm[r["benchmark"]][r["method"]] = r

    col_specs = [
        ("Benchmark set", "baseline"),
        ("Similarity search", "similarity_selection"),
        ("Active learning (ML)", "simple_active_learning"),
        ("Active learning (Ditto)", "ditto_active_learning"),
    ]
    auto_methods = ["similarity_selection", "simple_active_learning", "ditto_active_learning"]

    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Downstream F1 of Ditto trained on the benchmark training split and on the three "
        r"auto-labeled training sets, all labeled with \texttt{gpt-5.2}, after removing test-set leakage "
        r"from the auto-labeled training data. Each cell is F1 $\pm$ standard deviation over three runs, "
        r"except Active learning (Ditto), which aggregates six runs. The best value per benchmark is in "
        r"\textbf{bold} and the second best is \underline{underlined}. $\Delta$ is the best auto-labeled "
        r"value minus the benchmark set.}",
        r"  \label{tab:main-comparison-cleaned}",
        r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \scriptsize",
        r"  \begin{tabular}{lrrrrr}",
        r"    \toprule",
        r"    Benchmark & Benchmark set & Similarity search & Active learning (ML) & "
        r"Active learning (Ditto) & Best auto. $\Delta$ \\",
        r"    \midrule",
    ]
    for bm in BM_ORDER:
        cells = by_bm.get(bm, {})
        vals = []
        for header, key in col_specs:
            r = cells.get(key)
            vals.append((key, r.get("f1_mean", "") if r else "", r.get("f1_std", "") if r else ""))
        ranks = rank_row([(k, m) for k, m, s in vals])
        rendered = [fmt(m, s, ranks.get(k, 99)) for k, m, s in vals]
        # Delta = best auto-labeled mean - benchmark mean
        base_mean = cells.get("baseline", {}).get("f1_mean", "")
        auto_means = [float(cells[k]["f1_mean"]) for k in auto_methods
                      if k in cells and cells[k].get("f1_mean") not in (None, "")]
        if auto_means and base_mean not in (None, ""):
            delta = (max(auto_means) - float(base_mean)) * 100
            delta_str = f"{delta:+.2f}"
        else:
            delta_str = "--"
        lines.append("    " + BM_DISPLAY[bm] + " & " + " & ".join(rendered) + " & " + delta_str + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    # 1) Main comparison: baseline + similarity + simple_active + ditto_AL (GPT)
    t1 = build_main_table_with_delta(OUT / "plot_main_comparison.csv")
    (LATEX_DIR / "table_main_comparison.tex").write_text(t1 + "\n")

    # 2) LLM comparison: baseline + ditto_AL across GPT/Qwen/Kimi
    t2 = build_table(
        OUT / "plot_llm_comparison.csv",
        col_specs=[
            ("Benchmark set", "gold (official)"),
            ("AL (Ditto) GPT-5.2", "gpt-5.2"),
            ("AL (Ditto) Qwen 3.6+", "qwen3.6-plus"),
            ("AL (Ditto) Kimi K2.6", "kimi-k2.6"),
        ],
        caption=("Test F1 $\\pm$ standard deviation for Active learning (Ditto) under three labelling LLMs, "
                 "test-set leakage removed. \\textbf{Best} per row in bold, \\underline{second} underlined."),
        label="tab:llm-comparison-cleaned",
        group_field="labeller",
    )
    (LATEX_DIR / "table_llm_comparison.tex").write_text(t2 + "\n")

    # 3) Postfilter variants
    t3 = build_table(
        OUT / "plot_postfilter_variants.csv",
        col_specs=[
            ("Benchmark set", "Benchmark set (baseline)"),
            ("AL(Ditto)", "AL_Ditto"),
            ("+Relabel", "+Relabel"),
            ("+Relabel drop", "+Relabel drop"),
            ("+Closure drop", "+Closure drop"),
            ("+Closure \\& relabel drop", "+Closure AND relabel drop"),
            ("+Closure or relabel drop", "+Closure OR relabel drop"),
        ],
        caption=("Quality-improvement variants for Active learning (Ditto), test-set leakage removed. "
                 "Test F1 $\\pm$ standard deviation over three runs. "
                 "\\textbf{Best} per row in bold, \\underline{second} underlined."),
        label="tab:ditto-refinements-cleaned",
        group_field="variant",
    )
    (LATEX_DIR / "table_postfilter_variants.tex").write_text(t3 + "\n")

    for name in ["table_main_comparison", "table_llm_comparison", "table_postfilter_variants"]:
        print(f"Wrote {LATEX_DIR / (name + '.tex')}")
    print()
    print("=== table_main_comparison.tex ===")
    print(t1)
    print()
    print("=== table_postfilter_variants.tex ===")
    print(t3)


if __name__ == "__main__":
    main()
