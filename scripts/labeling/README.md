# Labeling Workflows

This directory contains the public entry points for constructing machine-labeled entity-matching training sets.

## Entry Points

- `similarity_search.py`: constructs training pairs from nearest-neighbour similarity search and labels the selected pairs with the teacher model.
- `active_learning_ml.py`: runs feature-based active learning with traditional machine-learning matchers and labels selected pairs with the teacher model.
- `active_learning_ditto.py`: runs seed labeling, feature-based active learning, and a Ditto-based active-learning phase.

## Examples

```bash
python scripts/labeling/similarity_search.py --help
python scripts/labeling/active_learning_ml.py --help
python scripts/labeling/active_learning_ditto.py --help
```

Post-processing and materialization utilities are kept in `scripts/post_processing/`.