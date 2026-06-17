#!/usr/bin/env python3
"""Build output/by_method/ symlink tree organizing existing dirs into
similarity_search / active_learning_ml / active_learning_ditto.

Idempotent — re-run any time to refresh; existing symlinks are replaced.
Nothing is moved physically; scripts/configs that point at the original
paths continue to work.
"""
from __future__ import annotations
from pathlib import Path
import os

ROOT = Path("/work/aasteine/Automatic-data-labeling")
OUTPUT = ROOT / "output"
VIEW = OUTPUT / "by_method"

# Each entry: subdir-name-in-method-dir -> target-relative-to-output (string)
MAPPING = {
    "similarity_search": {
        "labelling_run":              "seed_round_only_profiles",
        "cleaned_training":           "labeling_cleaned/seed_round",
        "cleaned_no_val_training":    "labeling_cleaned_no_val/seed_round",
        "cleaned_llmvalid_training":  "labeling_cleaned_llmvalid/seed_round",
        "cleaned_size_scan_training": "labeling_cleaned_size_scan/seed_round",
        "trainings_test_cleaned":              "ditto_cleaned_seedround_runs",
        "trainings_test_cleaned_pubprofile":   "ditto_cleaned_seedround_published_runs",
        "trainings_no_val":                    "ditto_cleaned_seedround_no_val_runs",
        "trainings_llmvalid":                  "ditto_cleaned_seedround_llmvalid_runs",
        "trainings_size_scan_abtbuy_methods":  "ditto_size_scan_abtbuy_methods_runs",
    },
    "active_learning_ml": {
        "labelling_run":              "simple_active_learning_labeling",
        "cleaned_training":           "labeling_cleaned/simple_active",
        "cleaned_no_val_training":    "labeling_cleaned_no_val/simple_active",
        "cleaned_llmvalid_training":  "labeling_cleaned_llmvalid/simple_active",
        "cleaned_size_scan_training": "labeling_cleaned_size_scan/simple_active",
        "trainings_test_cleaned":              "ditto_cleaned_simpleactive_runs",
        "trainings_test_cleaned_pubprofile":   "ditto_cleaned_simpleactive_published_runs",
        "trainings_no_val":                    "ditto_cleaned_simpleactive_no_val_runs",
        "trainings_llmvalid":                  "ditto_cleaned_simpleactive_llmvalid_runs",
        "trainings_size_scan_abtbuy_methods":  "ditto_size_scan_abtbuy_methods_runs",
    },
    "active_learning_ditto": {
        # labelling runs per LLM labeller
        "labelling_run/gpt":   "three_phase_labeling_ditto_only_v2",
        "labelling_run/qwen":  "ditto_labeling/qwen",
        "labelling_run/kimi":  "ditto_labeling/kimi",
        # post-processing variant labelling runs (GPT only)
        "labelling_run_postfilter/relabel":              "three_phase_labeling_ditto_only_v2_relabel_batch_gpt-5-mini_agent_precision",
        "labelling_run_postfilter/relabel_drop":         "three_phase_labeling_ditto_only_v2_drop_changed",
        "labelling_run_postfilter/closure_drop":         "three_phase_labeling_ditto_only_v2_closure_bridge_drop",
        "labelling_run_postfilter/closure_and_relabel":  "three_phase_labeling_ditto_only_v2_closure_bridge_relabel_changed_drop",
        "labelling_run_postfilter/closure_or_relabel":   "three_phase_labeling_ditto_only_v2_closure_bridge_or_relabel_changed_drop",
        # cleaned data per labeller
        "cleaned_training/gpt":   "labeling_cleaned/gpt",
        "cleaned_training/qwen":  "labeling_cleaned/qwen",
        "cleaned_training/kimi":  "labeling_cleaned/kimi",
        # cleaned data per post-filter variant
        "cleaned_training_postfilter/v_relabel":           "labeling_cleaned/v_relabel",
        "cleaned_training_postfilter/v_relabel_drop":      "labeling_cleaned/v_relabel_drop",
        "cleaned_training_postfilter/v_closure_drop":      "labeling_cleaned/v_closure_drop",
        "cleaned_training_postfilter/v_closure_relabel":   "labeling_cleaned/v_closure_relabel",
        "cleaned_training_postfilter/v_closure_or_relabel":"labeling_cleaned/v_closure_or_relabel",
        # cleaned-no-val per labeller
        "cleaned_no_val_training/gpt":  "labeling_cleaned_no_val/gpt",
        "cleaned_no_val_training/qwen": "labeling_cleaned_no_val/qwen",
        "cleaned_no_val_training/kimi": "labeling_cleaned_no_val/kimi",
        # cleaned-llmvalid per labeller
        "cleaned_llmvalid_training/gpt":  "labeling_cleaned_llmvalid/gpt",
        "cleaned_llmvalid_training/qwen": "labeling_cleaned_llmvalid/qwen",
        "cleaned_llmvalid_training/kimi": "labeling_cleaned_llmvalid/kimi",
        # cleaned size-scan per labeller (abt-buy)
        "cleaned_size_scan_training/gpt":  "labeling_cleaned_size_scan/gpt",
        "cleaned_size_scan_training/qwen": "labeling_cleaned_size_scan/qwen",
        "cleaned_size_scan_training/kimi": "labeling_cleaned_size_scan/kimi",
        # downstream Ditto trainings
        "trainings_test_cleaned":   "ditto_cleaned_runs",
        "trainings_no_val":         "ditto_cleaned_no_val_runs",
        "trainings_llmvalid":       "ditto_cleaned_llmvalid_runs",
        "trainings_size_scan_abtbuy": "ditto_size_scan_abtbuy_runs",
        # post-filter variant trainings
        "trainings_postfilter/v_relabel":           "ditto_cleaned_v_relabel_runs",
        "trainings_postfilter/v_relabel_drop":      "ditto_cleaned_v_relabel_drop_runs",
        "trainings_postfilter/v_closure_drop":      "ditto_cleaned_v_closure_drop_runs",
        "trainings_postfilter/v_closure_relabel":   "ditto_cleaned_v_closure_relabel_runs",
        "trainings_postfilter/v_closure_or_relabel":"ditto_cleaned_v_closure_or_relabel_runs",
    },
}

READMES = {
    "similarity_search": (
        "# Similarity search (a.k.a. seed-round)\n\n"
        "FAISS-based similarity selection + GPT-5.2 labelling — the 'Embed.-based "
        "Selection' column in the paper. Each subdir below symlinks back into "
        "`output/` (no data is moved):\n\n"
        "- `labelling_run/`           — raw labelling runs (`seed_round_only_profiles/`)\n"
        "- `cleaned_training/`        — test-leak + dedup removed\n"
        "- `cleaned_no_val_training/` — test + validation pairs removed\n"
        "- `cleaned_llmvalid_training/` — test removed + 80/20 LLM-only val split\n"
        "- `cleaned_size_scan_training/` — abt-buy size-scan cleaned inputs\n"
        "- `trainings_*` — downstream Ditto training runs for each regime\n"
    ),
    "active_learning_ml": (
        "# Active learning (ML)\n\n"
        "FAISS candidates + simple-ML active-learning loop + GPT-5.2 labelling — the "
        "'Active learning (ML)' column in the paper. Same layout as `similarity_search/`.\n"
    ),
    "active_learning_ditto": (
        "# Active learning (Ditto)\n\n"
        "Three-phase Ditto-ensemble active learning + LLM labelling — the 'Active "
        "learning (Ditto)' column in the paper. We have 3 LLM labellers (GPT-5.2, "
        "Qwen 3.6+, Kimi K2.6) and 5 post-processing variants (relabel / relabel "
        "drop / closure drop / closure AND relabel / closure OR relabel).\n\n"
        "- `labelling_run/{gpt,qwen,kimi}/`     — raw labelling runs per labeller\n"
        "- `labelling_run_postfilter/*/`        — post-processing variant labelling runs (GPT only)\n"
        "- `cleaned_training/{gpt,qwen,kimi}/`  — test-leak + dedup removed\n"
        "- `cleaned_training_postfilter/*/`     — same for variants\n"
        "- `cleaned_no_val_training/*/`         — test + validation pairs removed\n"
        "- `cleaned_llmvalid_training/*/`       — test removed + 80/20 LLM-only val split\n"
        "- `cleaned_size_scan_training/*/`      — abt-buy size-scan cleaned inputs per labeller\n"
        "- `trainings_test_cleaned/`            — Ditto trainings (all 3 labellers)\n"
        "- `trainings_no_val/`, `trainings_llmvalid/` — alternative validation regimes\n"
        "- `trainings_size_scan_abtbuy/`        — abt-buy size-scan trainings (every profile, every labeller)\n"
        "- `trainings_postfilter/*/`            — variant trainings (GPT only)\n"
    ),
}


def symlink(src_abs: Path, link: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    rel = os.path.relpath(src_abs, start=link.parent)
    link.symlink_to(rel)


def main():
    if VIEW.exists():
        # Wipe previous links only (don't touch any other content)
        for root, dirs, files in os.walk(VIEW, topdown=False):
            for f in files:
                p = Path(root, f)
                if p.is_symlink() or p.suffix == ".md":
                    p.unlink()
            for d in dirs:
                try:
                    Path(root, d).rmdir()
                except OSError:
                    pass
        try:
            VIEW.rmdir()
        except OSError:
            pass

    VIEW.mkdir(parents=True, exist_ok=True)
    missing = []
    for method, entries in MAPPING.items():
        method_dir = VIEW / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for subpath, target in entries.items():
            src = OUTPUT / target
            if not src.exists():
                missing.append(f"{method}/{subpath} -> {target}")
                continue
            symlink(src, method_dir / subpath)
        # README
        (method_dir / "README.md").write_text(READMES[method])
        print(f"  {method}: {len(entries)} symlinks + README")

    if missing:
        print("\n[WARN] missing targets (skipped):")
        for m in missing:
            print(f"  - {m}")
    print(f"\nView at: {VIEW}")


if __name__ == "__main__":
    main()
