"""Independent finite-tree oracle for saved real Julia iterator runs."""
from itertools import product
import ast
import math
from pathlib import Path
import tomllib
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'results/nonmyopic/herb_ordered_iterator_20260909'


def enumerate_trees(depth, unary, binary):
    rows = [('x', 1, (1, 0, 0, 0)), ('y', 1, (0, 1, 0, 0))]
    if depth == 1:
        return rows
    children = enumerate_trees(depth-1, unary, binary)
    if unary:
        rows += [(f'f({text})', size+1, (c[0], c[1], c[2]+1, c[3])) for text, size, c in children]
    if binary:
        for (a, sa, ca), (b, sb, cb) in product(children, repeat=2):
            if sa+sb+1 <= 7:
                counts = tuple(ca[i]+cb[i]+int(i == 3) for i in range(4))
                rows.append((f'pair({a}, {b})', sa+sb+1, counts))
    return rows


@pytest.mark.parametrize('index,depth,unary,binary,probabilities', [
    (0, 4, True, False, [.6, .1, .3, 1]),
    (1, 3, False, True, [.6, .1, 1, .3]),
    (2, 3, True, True, [.45, .15, .25, .15]),
    (3, 3, True, True, [.25, .25, .25, .25]),
])
def test_exact_independent_coverage_and_order(index, depth, unary, binary, probabilities):
    data = tomllib.loads((ROOT / 'audit.toml').read_text())
    assert data['status'] == 'exhaustive_order_pass'
    case = data['cases'][index]
    expected = enumerate_trees(depth, unary, binary)
    assert set(case['ordered_expressions']) == {r[0] for r in expected}
    assert case['count'] == len(expected)
    scores = sorted((sum(n*math.log(p) for n, p in zip(counts, probabilities)) for _, _, counts in expected), reverse=True)
    assert case['ordered_scores'] == pytest.approx(scores, abs=1e-12)


def test_full_prefix_weights_replay_and_limits():
    full = tomllib.loads((ROOT / 'full.toml').read_text())
    rules = [r.strip() for r in (ROOT.parent / 'herb_full_grammar_20260909' / 'grammar.jl').read_text().splitlines() if r.strip().startswith('Value = ')]
    for mode, arm in full['arms'].items():
        values = tomllib.loads((ROOT.parent / 'herb_guidance_mechanics_20260909' / f'{mode}.toml').read_text())['weights']
        probabilities = dict(zip(rules, values))
        def score(node):
            if isinstance(node, ast.Name):
                return math.log(probabilities[f'Value = {node.id}'])
            rule = f"Value = {node.func.id}({', '.join(['Value'] * len(node.args))})"
            return math.log(probabilities[rule]) + sum(score(a) for a in node.args)
        actual = [score(ast.parse(expr, mode='eval').body) for expr in arm['expressions']]
        assert actual == pytest.approx(arm['log_weights'], abs=1e-12)
        assert actual == pytest.approx(sorted(actual, reverse=True), abs=1e-12)
        assert len(actual) == 64 and arm['expansions'] <= 50000
    limits = tomllib.loads((ROOT / 'limits.toml').read_text())
    assert limits['status'] == 'limits_pass'
    assert limits['cap_preserved'] and limits['invalid_weights_rejected'] and limits['independent_counters']
