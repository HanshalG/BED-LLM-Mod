"""Prospective metadata-only cohort for the nested-expression/search interface."""
import hashlib
import json
from pathlib import Path


def select(inventory, excluded):
    if len(inventory) != len(set(inventory)):
        raise ValueError('duplicate inventory')
    available = set(inventory) - set(excluded)
    if len(available) < 6:
        raise ValueError('insufficient fresh tasks')
    return sorted(available, key=lambda key: hashlib.sha256(('bed-rearc-mechanisms-v1:' + key).encode()).digest())[:6]


if __name__ == '__main__':
    root = Path('results/nonmyopic')
    first = json.loads((root / 'REARC_SOURCE_SCOPE_20260909.json').read_text())
    second = json.loads((root / 'REARC_FEEDBACK_COHORT_20260909.json').read_text())
    third = json.loads((root / 'REARC_EXPRESSION_COHORT_20260909.json').read_text())
    fourth = json.loads((root / 'REARC_SLOT_COHORT_20260909.json').read_text())
    excluded = first['selected_ids'] + second['selected_ids'] + third['selected_ids'] + fourth['selected_ids']
    result = {'source_commit': first['source_commit'], 'excluded_ids': sorted(excluded),
              'selected_ids': select(first['all_ids'], excluded),
              'selection': 'first6 SHA256(bed-rearc-mechanisms-v1:+id), no replacement',
              'source_seeds': [38000, 38001, 38002], 'demo_seeds': [38100],
              'query_seeds': [38200,38201],
              'target_seeds': list(range(38300, 38308)), 'calls': 0, 'paid_authorized': False}
    with (root / 'REARC_MECHANISM_COHORT_20260909.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


