"""
Blocking and matching optimization using validation sets.

This module re-exports the optimization functions from their specialized modules
for backward compatibility. The actual implementations are in:
- blocking_optimization.py - Blocker optimization and evaluation
- matching_optimization.py - Matcher optimization
- labeled_set_generation.py - Training and validation set generation

Optimization Strategy
---------------------
The optimizer prioritizes HIGH REDUCTION RATIO (fewer candidates) while
maintaining a minimum PAIR COMPLETENESS (recall) threshold of 97%.

This means:
- We want to generate as few candidate pairs as possible (high reduction)
- But we must find at least 97% of true matches (high recall)
- Among configurations meeting the recall constraint, we pick the one
  with the highest reduction ratio

Why this strategy?
- Blocking is a filtering step before expensive matching
- Missing true matches at blocking stage cannot be recovered
- But having too many candidates wastes compute on the matcher
- 97% recall ensures we catch almost all matches while still filtering aggressively

Key concepts:
- Validation sets are cached as CSV files to avoid regeneration
- Uses EntityMatchingEvaluator for metrics (pair_completeness, pair_quality)
- pair_completeness = blocking recall (% of true matches found)
- reduction_ratio = 1 - (candidates / total_possible_pairs)
"""

from __future__ import annotations

# Re-export from blocking_optimization
from .blocking_optimization import (
    get_default_blocker_specs,
    optimize_blocking,
    evaluate_blocker_types,
)

# Re-export from matching_optimization
from .matching_optimization import (
    optimize_matching,
)

__all__ = [
    "get_default_blocker_specs",
    "optimize_matching",
    "optimize_blocking",
    "evaluate_blocker_types",
]
