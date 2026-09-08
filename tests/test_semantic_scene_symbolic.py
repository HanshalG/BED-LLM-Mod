import copy
from itertools import islice
import json

import pytest

from environments.semantic_scene.rules import compile_rule, RuleError
from environments.semantic_scene.symbolic import candidates, propose
from environments.semantic_scene.belief import RuleBelief, UnsupportedHistory


def scene(color):
    return {"objects": [{"color": color, "shape": "block", "size": "small"}]}


def test_real_search_uses_history_and_fixed_work_budget():
    history = [(scene("red"), True), (scene("blue"), False)]
    saved = copy.deepcopy(history)
    aware = propose(history, max_candidates=192)
    repeated = propose(history, max_candidates=192)
    blind = propose([], max_candidates=192)
    reversed_labels = propose([(s, not y) for s, y in history], max_candidates=192)
    assert aware.selected_json == repeated.selected_json
    assert aware.records == repeated.records
    assert aware.selected_json != blind.selected_json != reversed_labels.selected_json
    assert aware.candidates_evaluated == 192 and aware.scene_evaluations == 384
    assert blind.scene_evaluations == 0 and not aware.grammar_exhausted
    assert history == saved
    for raw in aware.selected_json:
        assert all(compile_rule(raw).label(s) == y for s, y in history)
    pool = RuleBelief(
        [compile_rule(r) for r in aware.selected_json], [1] * len(aware.selected_json)
    )
    assert pool.condition(history).predict(scene("red")) == 1


def test_no_consistent_proposal_is_explicit_empty_not_a_guess():
    # First candidate is count(red) == 0, which contradicts this fixture.
    result = propose([(scene("red"), True)], max_candidates=1)
    assert result.selected_json == ()
    assert result.records[0][2] == 1
    assert not result.grammar_exhausted


def test_candidate_sizes_are_actual_nodes_and_ordered():
    def size(node):
        if isinstance(node, dict):
            return (1 if "op" in node else 0) + sum(size(v) for v in node.values())
        if isinstance(node, list):
            return sum(size(v) for v in node)
        return 0

    previous = 0
    operators = set()
    for declared, candidate in islice(candidates(), 2400):
        assert declared == size(candidate) and declared >= previous
        previous = declared
        operators.add(candidate["op"])
        compile_rule(json.dumps(candidate))
    assert {"count", "pair", "not", "and", "or"} <= operators


def test_input_guard_and_contradiction():
    with pytest.raises(UnsupportedHistory):
        propose([(scene("red"), True), (scene("red"), False)])
    for budget in (0, True, 50001, 1.5):
        with pytest.raises(RuleError):
            propose([], max_candidates=budget)
    with pytest.raises(RuleError):
        propose([(scene("red"), 1)])


def test_history_identity_is_scene_order_invariant_but_label_sensitive():
    mixed = {"objects": scene("red")["objects"] + scene("blue")["objects"]}
    flipped = {"objects": list(reversed(mixed["objects"]))}
    a = propose([(mixed, True)], max_candidates=5)
    b = propose([(flipped, True)], max_candidates=5)
    c = propose([(mixed, False)], max_candidates=5)
    assert a.history_sha256 == b.history_sha256 != c.history_sha256
    assert a.records == b.records


def test_full_declared_grammar_exhaustion_and_accounting():
    result = propose([], max_candidates=50000)
    # 8 attribute predicates, 24 count forms, 36 unordered predicate pairs.
    expected = 192 + 192 + 36 * 3 * 2 + 192 + 28 * 2 * 24 + 192 * 191
    assert result.candidates_evaluated == expected == 38808
    assert result.grammar_exhausted and result.scene_evaluations == 0
    assert len(result.records) == expected
    assert all(record[2] == 0 for record in result.records)
