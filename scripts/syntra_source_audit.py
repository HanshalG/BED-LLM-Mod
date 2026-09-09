"""Pinned source-shape audit only; never execute code or expose task labels."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = '40d0bdfac9a1dade0afe49fa8308fadff78035c7'
FILES = {
    'playgol_v2.jsonl': '11d197909e517432ca2914ed408d74f4b33a243aab676241dc56b77a608dd1aa',
    'mbpp_plus_51_cases.jsonl': '8072f8202d4f86a2df51b97d0cfc2355be670e4acd6bd9b788a2598f6409a576',
}


def summarize(name, raw):
    rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
    if not rows:
        raise ValueError('empty source')
    result = dict(rows=len(rows), sha256=hashlib.sha256(raw).hexdigest())
    if name == 'playgol_v2.jsonl':
        panels, splits, distinct = [], Counter(), Counter()
        for row in rows:
            if set(row) != {'idx', 'name', 'train', 'test'}:
                raise ValueError('unexpected Playgol schema')
            xs = []
            for split in ('train', 'test'):
                if type(row[split]) is not list:
                    raise ValueError('unexpected split type')
                for pair in row[split]:
                    if set(pair) != {'input', 'output'}:
                        raise ValueError('unexpected example schema')
                    # Deliberately never inspect output values, even for consistency checks.
                    xs.append(json.dumps(pair['input'], sort_keys=True))
            splits[f"{len(row['train'])}/{len(row['test'])}"] += 1
            distinct[str(len(set(xs)))] += 1
            panels.append(tuple(sorted(xs)))
        result.update(split_counts=dict(splits), distinct_input_counts=dict(distinct),
            unique_names=len({r['name'] for r in rows}), unique_ids=len({r['idx'] for r in rows}),
            duplicate_input_panels=len(rows)-len(set(panels)),
            eligible_for_one_initial_one_target_four_candidates=sum(
                n for count, n in distinct.items() if int(count) >= 6))
    elif name == 'mbpp_plus_51_cases.jsonl':
        counts = {s:Counter() for s in ('train','test')}
        for row in rows:
            if set(row) != {'code','idx','prompt','source_file','task_id','test','train'}:
                raise ValueError('unexpected MBPP schema')
            for split in counts:
                values = ast.literal_eval(row[split])
                if type(values) is not list:
                    raise ValueError('expected list literal')
                counts[split][str(len(values))] += 1
        result['split_lengths'] = {k:dict(v) for k,v in counts.items()}
        result['semantic_ambiguity_or_horizon_verified'] = False
    else:
        raise ValueError('unrecognized source')
    return result


def run():
    result = dict(source_commit=COMMIT, model_calls=0, cost_usd=0, paid_authority=False,
        scope='Full source payloads parsed locally; only schema/cardinality metadata emitted. '
              'No code executed or task predictions scored; not a sealed-data ingestion claim.', files={})
    for name, sha in FILES.items():
        url = f'https://raw.githubusercontent.com/klee972/SYNTRA/{COMMIT}/SYNTRA/data/{name}'
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read(3000001)
        if len(raw)>3000000 or hashlib.sha256(raw).hexdigest()!=sha:
            raise ValueError('source size/hash changed')
        result['files'][name] = summarize(name, raw)
    return result


if __name__=='__main__':
    result = run()
    path = Path('results/nonmyopic/SYNTRA_SOURCE_CARDINALITY_AUDIT_20260909.json')
    with path.open('x') as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write('\n')
    print(json.dumps(result,sort_keys=True))
