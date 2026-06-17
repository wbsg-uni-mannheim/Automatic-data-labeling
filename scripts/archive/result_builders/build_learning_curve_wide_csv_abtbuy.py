#!/usr/bin/env python3
"""Re-shape plot_learning_curve_abtbuy.csv into a wide format (method × N).

Output: output/results_summary/plot_learning_curve_abtbuy_wide.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")
SRC = ROOT / "output/results_summary/plot_learning_curve_abtbuy.csv"
OUT = ROOT / "output/results_summary/plot_learning_curve_abtbuy_wide.csv"

df = pd.read_csv(SRC)
df["f1_mean_pct"] = (df["f1_mean"] * 100).round(2)
df["f1_std_pct"] = (df["f1_std"] * 100).round(2)
df["cell"] = df.apply(lambda r: f"{r['f1_mean_pct']:.2f} ± {r['f1_std_pct']:.2f}", axis=1)

wide = df.pivot(index="method_display", columns="N", values="cell")
wide.index.name = "Method"
wide = wide.reindex(["Similarity search", "Active learning (ML)", "Active learning (Ditto)"])

# also write a plain numeric version (only mean)
mean_wide = (df.pivot(index="method_display", columns="N", values="f1_mean") * 100).round(2)
mean_wide.index.name = "Method"
mean_wide = mean_wide.reindex(["Similarity search", "Active learning (ML)", "Active learning (Ditto)"])

# Stack the two tables one above the other for a single CSV
with OUT.open("w") as f:
    f.write("# F1 mean ± std (test set, % units; 6 seeds for sim+alditto, 3 seeds for AL ML)\n")
    wide.to_csv(f)
    f.write("\n# F1 mean only (% units)\n")
    mean_wide.to_csv(f)
    # Also dump n_runs per cell
    n_runs_wide = df.pivot(index="method_display", columns="N", values="n_runs")
    n_runs_wide.index.name = "Method"
    n_runs_wide = n_runs_wide.reindex(["Similarity search", "Active learning (ML)", "Active learning (Ditto)"])
    f.write("\n# n_runs per cell\n")
    n_runs_wide.to_csv(f)

print(f"Wrote {OUT}")
print()
print("=== F1 mean ± std (%) ===")
print(wide.to_string())
print()
print("=== F1 mean only (%) ===")
print(mean_wide.to_string())
