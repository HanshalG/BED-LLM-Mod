"""Check actual Herb candidate graphs in the existing isolated Hodel interpreter."""
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

from scripts.rearc_graph_runtime import execute
from scripts.rearc_program_graph import exports, validate_graph


def check(path):
    raw = Path(path).read_bytes()
    data = tomllib.loads(raw.decode())
    assert data['status'] == 'iterator_bridge_pass' and data['branch_replay_equal']
    source = subprocess.check_output([
        'git', '-C', '/private/tmp/bed-rearc-source-audit', 'show',
        'e5b7f1d06362a76f9d3b8c25154ff1fafca897ce:dsl.py'])
    functions, constants = exports(source.decode())
    for row in data['rows']:
        validate_graph(row['graph'], functions, constants)
    by_expr = {row['expression']: row['graph'] for row in data['rows']}
    fixtures = {
        'identity(I)': [[1, 2]],
        'vmirror(I)': [[2, 1]],
        'hconcat(I, I)': [[1, 2, 1, 2]],
        'apply(identity, I)': [[1, 2]],
        'apply(compose(identity, identity), I)': [[1, 2]],
        'repeat(first(I), TWO)': [[1, 2], [1, 2]],
        'paint(I, asobject(I))': [[1, 2]],
    }
    records = []
    for expression, expected in fixtures.items():
        result = execute(by_expr[expression], [[1, 2]])
        assert result['status'] == 'ok', result
        assert result['output'] == expected, result
        records.append({'expression': expression, 'result': result})
    demonstration = [execute(by_expr[expr], [[0, 0]]) for expr in ('identity(I)', 'vmirror(I)')]
    assert all(row['status'] == 'ok' and row['output'] == [[0, 0]] for row in demonstration)
    return {'status': 'bridge_fixture_pass', 'candidate_rows': len(data['rows']),
            'unique_graphs': len({json.dumps(row['graph'], sort_keys=True) for row in data['rows']}),
            'iterator_output_sha256': hashlib.sha256(raw).hexdigest(),
            'records': records, 'ambiguity_demonstration': demonstration,
            'calls': 0, 'cost_usd': 0,
            'full_dsl_search_qualified': False, 'depth_result': False}


if __name__ == '__main__':
    import sys
    print(json.dumps(check(sys.argv[1]), indent=2))
