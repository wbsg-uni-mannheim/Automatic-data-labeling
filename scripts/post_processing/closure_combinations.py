"""Intersect/union frozen closure removals with existing label-review disagreements."""
from scripts.post_processing.consistent_closure import negative_conflicts


def combine(original, reviewed, closure_indices, variant):
    if variant not in ('and', 'or'):
        raise ValueError('Expected and/or')
    if len(original) != len(reviewed):
        raise ValueError('Incomplete review')
    review = {row['pair_id']: row for row in reviewed}
    if len(review) != len(reviewed) or {r['pair_id'] for r in original} != set(review):
        raise ValueError('Review pair IDs do not match unique original pairs')
    changed = set()
    for i, row in enumerate(original):
        other = review[row['pair_id']]
        if {k: v for k, v in row.items() if k != 'label'} != {k: v for k, v in other.items() if k != 'label'}:
            raise ValueError('Review modified record content')
        if type(other['label']) is not int or other['label'] not in (0, 1):
            raise ValueError('Invalid reviewed label')
        if row['label'] != other['label']:
            changed.add(i)
    closure = set(closure_indices)
    if not closure.issubset(range(len(original))):
        raise ValueError('Closure indices out of bounds')
    dropped = closure & changed if variant == 'and' else closure | changed
    kept = [row for i, row in enumerate(original) if i not in dropped]
    conflicts = len(negative_conflicts(kept))
    if variant == 'or' and conflicts:
        raise ValueError('Union must retain no negative contradictions')
    return kept, {
        'rows_before': len(original), 'rows_after': len(kept), 'review_disagreements': len(changed),
        'closure_marked': len(closure), 'removed_positive': sum(original[i]['label'] == 1 for i in dropped),
        'removed_negative': sum(original[i]['label'] == 0 for i in dropped),
        'negative_conflicts_remaining': conflicts, 'dropped_indices': sorted(dropped),
    }
