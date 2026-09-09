"""Measure the frozen bounded full-grammar prefix on a handcrafted input."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
import tomllib

from scripts.rearc_graph_runtime import execute
from scripts.rearc_program_graph import exports, validate_graph


def check(path):
    raw = Path(path).read_bytes()
    data = tomllib.loads(raw.decode())
    assert data['status'] == 'bounded_enumeration_complete'
    source = subprocess.check_output(['git', '-C', '/private/tmp/bed-rearc-source-audit', 'show',
                                     'e5b7f1d06362a76f9d3b8c25154ff1fafca897ce:dsl.py'])
    functions, constants = exports(source.decode())
    assert len(data['rows']) == 256
    for row in data['rows']:
        validate_graph(row['graph'], functions, constants)
    records = []
    started = time.monotonic()
    for row in data['rows']:
        if time.monotonic() - started > 240:
            break
        records.append({'expression': row['expression'],
                        'result': execute(row['graph'], [[1, 2], [3, 4]])})
    fixture = execute(data['computed_callable_fixture'], [[1, 2], [3, 4]])
    assert fixture['status'] == 'ok' and fixture['output'] == [[1, 2], [3, 4]]
    counts = Counter(row['result']['status'] for row in records)
    outputs = {json.dumps(row['result']['output']) for row in records if row['result']['status'] == 'ok'}
    return {'status': 'complete' if len(records) == 256 else 'execution_cap',
            'candidate_sha256': hashlib.sha256(raw).hexdigest(),
            'executed': len(records), 'counts': dict(counts), 'distinct_grid_outputs': len(outputs),
            'records': records, 'computed_callable_fixture': fixture,
            'seconds': time.monotonic()-started, 'calls': 0, 'cost_usd': 0,
            'semantic_coverage_qualified': False, 'benchmark_examples': 0}


if __name__ == '__main__':
    import sys
    print(json.dumps(check(sys.argv[1]), indent=2))
