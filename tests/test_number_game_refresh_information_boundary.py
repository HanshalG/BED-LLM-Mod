"""Source-bound regression for truth-dependent controlled-support scoring."""
import ast
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path


@dataclass(frozen=True)
class Hypothesis:
    extension: tuple[bool, ...]


def scorer(*, predictive=False):
    path = Path('scripts/number_game_generator_aware_bed.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '1df1eab2d6e7dbbebb89e31d6c9863160d15ce519c8d9c55724de577a500a35a'
    names = {'binary_entropy', 'query_eig', 'best_query', 'merge_controlled_support', 'generator_aware_score'}
    if predictive:
        names |= {'hamming_error', 'evaluate_policy_root', 'predictive_bayes_risk_scores'}
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    body += [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(body) == len(names)+1
    namespace = {'math': math, 'DOMAIN': range(4)}
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, '<source-bound-scorer>', 'exec'), namespace)
    return namespace


def test_hidden_truth_changes_continuation_at_identical_history():
    code = scorer()
    first = Hypothesis((False, True, False, False))
    second = Hypothesis((False, False, True, False))
    generated = [Hypothesis((False, False, False, True))]
    assert first.extension[0] == second.extension[0]
    choices = [code['best_query'](code['merge_controlled_support'](truth, generated),
                                  excluded=(0,))[0] for truth in (first, second)]
    assert choices == [1, 2]
    assert code['best_query'](generated, excluded=(0,))[0] == 1


def test_controlled_future_entropy_not_realized_updater_entropy():
    code = scorer()
    truths = [Hypothesis((False, True, False, False)),
              Hypothesis((False, False, True, False))]
    generated = [Hypothesis((False, False, False, True))]
    branches = {(0, False): generated}
    score = code['generator_aware_score'](truths, 0, branches)
    assert math.isclose(score, math.log(2))
    assert code['query_eig'](truths, 0) == 0
    assert code['best_query'](generated, excluded=(0,))[1] == 0


def test_later_predictive_path_keeps_continuation_history_measurable():
    code = scorer(predictive=True)
    truths = [Hypothesis((False, True, False, False)),
              Hypothesis((False, False, True, False))]
    branches = {(0, False): [Hypothesis((False, False, False, True))]}
    before = tuple(branches[(0, False)])
    scores = code['predictive_bayes_risk_scores'](support=truths, roots=[0], branches=branches)
    rows = scores[0]['targets']
    assert [row['second_query'] for row in rows] == [1, 1]
    assert tuple(branches[(0, False)]) == before
    assert [row['branch_support_size'] for row in rows] == [1, 1]
    assert [row['survivor_count'] for row in rows] == [0, 1]
    assert all(not row['truth_extension_covered'] for row in rows)
    deployed = code['evaluate_policy_root'](policy='deployed', root=0,
        targets={f'particle_{i:02d}': truth for i, truth in enumerate(truths)}, branches=branches)
    assert deployed['targets'] == rows
    assert deployed['mean_posterior_predictive_brier'] == scores[0]['mean_posterior_predictive_brier']
