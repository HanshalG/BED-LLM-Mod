"""Source-bound regression for truth-dependent controlled-support scoring."""
import ast
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path


@dataclass(frozen=True)
class Hypothesis:
    extension: tuple[bool, ...]


def scorer():
    path = Path('scripts/number_game_generator_aware_bed.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '1df1eab2d6e7dbbebb89e31d6c9863160d15ce519c8d9c55724de577a500a35a'
    names = {'binary_entropy', 'query_eig', 'best_query', 'merge_controlled_support', 'generator_aware_score'}
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
