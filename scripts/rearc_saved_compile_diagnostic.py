"""Static saved-response audit; never execute or salvage rejected programs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.rearc_named_plan_interface import parse
from scripts.rearc_program_graph import exports
from scripts.herb_candidate_expression import to_graph
from scripts.rearc_source_scope import SOURCE, COMMIT

ROOT = Path('results/nonmyopic/rearc_named_plan_qualification_20260909')
RESULT_SHA = '82250825f92d239318503db9ce6c99c545c2027aea82357b7989ae8db56fa4a7'


def diagnose(text, dsl):
    """Expose structural error locations only, without evaluating any expression."""
    try:
        values = parse(text, 'compile')['hypotheses']
    except (ValueError, TypeError, KeyError) as error:
        return {'batch_valid': False, 'schema_error': str(error), 'slots': []}
    functions, constants = exports(dsl)
    rows = []
    for i, expression in enumerate(values):
        try:
            to_graph(expression, functions, constants)
        except (ValueError, TypeError, KeyError) as error:
            rows.append({'slot': i, 'valid': False, 'error': str(error)})
        else:
            rows.append({'slot': i, 'valid': True})
    return {'batch_valid': all(r['valid'] for r in rows), 'slots': rows}


def audit(root, expected_sha, dsl):
    raw = (root/'result.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('terminal identity')
    terminal = json.loads(raw)
    rows = []
    for i in range(6):
        for arm in ('contrasting', 'ordinary'):
            for stage in ('compile', 'repair'):
                name = f'{i}_{arm}_{stage}.response.json'
                raw = (root/name).read_bytes()
                if hashlib.sha256(raw).hexdigest() != terminal['artifact_sha256'][name]:
                    raise ValueError('response identity')
                text = json.loads(raw)['choices'][0]['message']['content']
                rows.append({'task': i, 'arm': arm, 'stage': stage, **diagnose(text, dsl)})
    counts = Counter(s['error'] for r in rows for s in r['slots'] if not s['valid'])
    return {'rows': rows, 'invalid_batches': sum(not r['batch_valid'] for r in rows),
            'slot_errors': dict(counts),
            'valid_slots_in_rejected_batches': sum(s['valid'] for r in rows
                if not r['batch_valid'] for s in r['slots']),
            'calls': 0, 'executions': 0, 'new_outputs': 0, 'cost_usd': 0,
            'rescored': False, 'depth_authorized': False}


if __name__ == '__main__':
    dsl = subprocess.check_output(['git', '-C', str(SOURCE), 'show', COMMIT+':dsl.py']).decode()
    result = audit(ROOT, RESULT_SHA, dsl)
    out = Path('results/nonmyopic/REARC_SAVED_COMPILE_DIAGNOSTIC_20260909.json')
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))
