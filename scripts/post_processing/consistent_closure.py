"""Drop positive bridges, then negative edges inside retained positive components.

No gold labels or teacher calls. Entity IDs are namespaced by source side.
Rows and their order are preserved; only explicitly audited rows are removed.
"""
from __future__ import annotations

import pandas as pd

from scripts.post_processing.build_closure_bridge_profiles import _detect_positive_bridge_edges


def pair_ids(row):
    if 'id_left' in row and 'id_right' in row:
        pair = str(row['id_left']), str(row['id_right'])
    else:
        pid = str(row['pair_id'])
        parts = pid.split('#') if '#' in pid else pid.split('__')
        if len(parts) not in (2, 3, 4):
            raise ValueError(f'Unrecognized pair ID: {pid}')
        pair = parts[0], parts[1]
    if not all(pair):
        raise ValueError('Empty entity ID')
    return pair


def negative_conflicts(rows):
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    nodes = [(('left', a), ('right', b)) for a, b in map(pair_ids, rows)]
    for row, (left, right) in zip(rows, nodes):
        if type(row['label']) is not int or row['label'] not in (0, 1):
            raise ValueError('Expected integer binary labels')
        if row['label'] == 1:
            parent[find(left)] = find(right)
    return [i for i, (row, (left, right)) in enumerate(zip(rows, nodes))
            if row['label'] == 0 and find(left) == find(right)]


def filter_rows(rows):
    if not rows:
        raise ValueError('Empty training set')
    before = negative_conflicts(rows)  # Also validates every label and entity ID.
    frame = pd.DataFrame([
        {'id1': a, 'id2': b, 'rid1': 'entity:' + a, 'rid2': 'entity:' + b, 'label': row['label']}
        for row, (a, b) in zip(rows, map(pair_ids, rows))])
    flagged, _ = _detect_positive_bridge_edges(frame, min_component_nodes=3)
    bridges = set(flagged['row_index'].astype(int)) if len(flagged) else set()
    assert all(rows[i]['label'] == 1 for i in bridges)
    after_bridges = [row for i, row in enumerate(rows) if i not in bridges]
    survivor_indices = [i for i in range(len(rows)) if i not in bridges]
    negatives = {survivor_indices[i] for i in negative_conflicts(after_bridges)}
    kept = [row for i, row in enumerate(rows) if i not in bridges | negatives]
    assert not negative_conflicts(kept)
    audit = {
        'rows_before': len(rows), 'rows_after': len(kept),
        'positive_bridges_removed': len(bridges),
        'negative_conflicts_before': len(before),
        'negative_conflicts_removed': len(negatives),
        'negative_conflicts_remaining': 0,
        'removed': [{'index': i, 'pair_id': rows[i]['pair_id'],
                     'reason': 'positive_bridge' if i in bridges else 'negative_in_positive_component'}
                    for i in sorted(bridges | negatives)],
    }
    return kept, audit
