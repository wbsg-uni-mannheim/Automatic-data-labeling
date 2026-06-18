# Labeling Workflows

The three entry points that build machine-labeled training sets. Each selects record pairs and labels them with the teacher model.

| Script | What it does |
|---|---|
| `similarity_search.py` | Labels the top nearest-neighbor pairs from similarity search. |
| `active_learning_ml.py` | Active learning with a feature-based matcher. |
| `active_learning_ditto.py` | Seed labeling, feature-based active learning, then a Ditto-based phase. |

```bash
python scripts/labeling/similarity_search.py --help
python scripts/labeling/active_learning_ml.py --help
python scripts/labeling/active_learning_ditto.py --help
```

[`artifacts/USAGE.md`](../../artifacts/USAGE.md) walks through a full Abt-Buy example. Post-processing lives in [`../post_processing/`](../post_processing/).
