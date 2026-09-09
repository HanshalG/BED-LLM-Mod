"""Recheck the opened engineering fixture after the ordering correction."""
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

from scripts.herb_support_collector import collect
from scripts.rearc_graph_runtime import execute
from scripts.rearc_program_graph import exports


def run(root):
    previous_path = root.parent / 'herb_guidance_mechanics_20260909' / 'result.json'
    previous_raw = previous_path.read_bytes()
    previous = json.loads(previous_raw)
    assert previous['status'] == 'engineering_complete' and previous['guide_is_handcrafted']
    cache = {}
    def key(graph, value):
        return json.dumps([graph, value], sort_keys=True)
    for arm in previous['arms'].values():
        for graph, predictions in zip(arm['graphs'], arm['predictions']):
            assert len(predictions) == 1
            prediction = predictions[0]
            cache[key(graph, [[0, 0]])] = {'status': 'ok', 'output': prediction} if prediction is not None else {'status': 'failed'}
        survivors = [g for g,w in zip(arm['graphs'], arm['posterior']['weights']) if w > 0]
        assert len(survivors) == len(arm['distinguishing_input_predictions'])
        for graph, row in zip(survivors, arm['distinguishing_input_predictions']):
            cache[key(graph, [[1, 2]])] = row['prediction']
    fresh, hits = 0, 0
    def cached(graph, value):
        nonlocal fresh, hits
        identifier = key(graph, value)
        if identifier not in cache:
            cache[identifier] = execute(graph, value)
            fresh += 1
        else:
            hits += 1
        return cache[identifier]
    source = subprocess.check_output(['git', '-C', '/private/tmp/bed-rearc-source-audit', 'show',
                                     'e5b7f1d06362a76f9d3b8c25154ff1fafca897ce:dsl.py'])
    functions, constants = exports(source.decode())
    full = tomllib.loads((root / 'full.toml').read_text())
    assert full['status'] == 'full_order_pass'
    results = {}
    for mode, candidate in full['arms'].items():
        assert candidate['count'] == 64 and candidate['root_grid_only']
        result = collect(candidate['expressions'], [{'input': [[0, 0]], 'output': [[0, 0, 0, 0]]}],
                         functions, constants, cached, attempts=64)
        result['future'] = [cached(g, [[1, 2]]) for g,w in zip(result['graphs'], result['posterior']['weights']) if w > 0]
        result['search_expansions'] = candidate['expansions']
        results[mode] = result
    return {'status': 'engineering_complete', 'arms': results,
            'previous_prediction_sha256': hashlib.sha256(previous_raw).hexdigest(),
            'fresh_executions': fresh, 'cached_requests': hits,
            'calls': 0, 'cost_usd': 0, 'guide_is_handcrafted': True, 'llm_efficacy_claim': False}


if __name__ == '__main__':
    import sys
    print(json.dumps(run(Path(sys.argv[1])), indent=2))
