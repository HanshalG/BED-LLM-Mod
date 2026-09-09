"""Handcrafted guide/collector integration, not an LLM or benchmark experiment."""
import ast
import json
import math
from pathlib import Path
import subprocess
import tomllib

from scripts.herb_support_collector import collect
from scripts.rearc_graph_runtime import execute
from scripts.rearc_program_graph import exports


def run(root):
    source = subprocess.check_output(['git', '-C', '/private/tmp/bed-rearc-source-audit', 'show',
                                     'e5b7f1d06362a76f9d3b8c25154ff1fafca897ce:dsl.py'])
    functions, constants = exports(source.decode())
    cache = {}
    def cached(graph, value):
        key = json.dumps([graph, value], sort_keys=True)
        if key not in cache:
            cache[key] = execute(graph, value)
        return cache[key]
    results = {}
    grammar_path = root.parent / 'herb_full_grammar_20260909' / 'grammar.jl'
    rules = [line.strip() for line in grammar_path.read_text().splitlines() if line.strip().startswith('Value = ')]
    for mode in ('base', 'guided'):
        candidates = tomllib.loads((root / f'{mode}_root_candidates.toml').read_text())
        assert candidates['root_grid_only'] and len(candidates['expressions']) == 64
        probabilities = tomllib.loads((root / f'{mode}.toml').read_text())['weights']
        probability = dict(zip(rules, probabilities))
        def score(node):
            if isinstance(node, ast.Name):
                return math.log(probability[f'Value = {node.id}'])
            return math.log(probability[f"Value = {node.func.id}({', '.join(['Value'] * len(node.args))})"]) + sum(score(a) for a in node.args)
        log_weights = [score(ast.parse(expr, mode='eval').body) for expr in candidates['expressions']]
        result = collect(candidates['expressions'], [{'input': [[0, 0]], 'output': [[0, 0, 0, 0]]}],
                         functions, constants, cached, attempts=64)
        future = []
        for graph, weight in zip(result['graphs'], result['posterior']['weights']):
            if weight > 0:
                future.append({'weight': weight, 'prediction': cached(graph, [[1, 2]])})
        result['distinguishing_input_predictions'] = future
        result['search_order_log_weights'] = log_weights
        result['nonmonotone_search_order_pairs'] = [i for i in range(1, len(log_weights)) if log_weights[i] > log_weights[i-1]+1e-12]
        results[mode] = result
    return {'status': 'engineering_complete', 'arms': results, 'cached_executions': len(cache),
            'llm_calls': 0, 'cost_usd': 0, 'benchmark_examples': 0,
            'guide_is_handcrafted': True, 'llm_efficacy_claim': False}


if __name__ == '__main__':
    import sys
    print(json.dumps(run(Path(sys.argv[1])), indent=2))
