import json
import pytest
from scripts.rearc_slot_panel import collect, checked_search

DSL = 'def identity(x: Any) -> Any:\n return x\ndef zero(x: Any) -> Any:\n return x\n'


def cases():
    return [{'inputs': [[[0]], [[1]], [[2]]], 'outputs': [[[0]], [[1]], [[2]]],
             'target_inputs': [[[3]] for _ in range(8)]} for _ in range(4)]


def graph(name):
    return {'steps': [{'id': 'x0', 'op': name, 'args': ['I']}], 'output': 'x0'}


def search_result(count, invalid=False):
    return {'status': 'complete', 'replacement_candidates': 0,
            'slots': [{'slot': i, 'graph': None if invalid or i%2 else graph('zero')}
                      for i in range(count)]}


def test_full_mixed_slot_path_positive_gate_privacy_and_sealing():
    requests, searches, updates, seals = {}, [], [], []
    def request(tag, body):
        requests[tag] = body
        name = 'identity' if '_aware_' in tag else 'zero'
        return json.dumps({'hypotheses': [f'{name}(I)']*4})
    def search(proposals, count, cap):
        searches.append((len(proposals), count, cap))
        return search_result(count)
    def evaluate(g, inputs):
        assert g is not None
        return inputs if g['steps'][0]['op'] == 'identity' else [[[0]] for _ in inputs]
    def targets():
        assert len(seals) == 1 and len(seals[0]) == 4
        for row in seals[0]:
            assert row['initial']['candidate_slots'] == 64
            assert row['aware']['candidate_slots'] == row['blind']['candidate_slots'] == 128
            assert row['direct']['candidate_slots'] == 16
            assert row['symbolic']['candidate_slots'] == 128
        return [[[[3]] for _ in range(8)] for _ in range(4)]
    result = collect(cases(), DSL, request, evaluate,
        lambda g,x: {'status': 'ok', 'output': evaluate(g,[x])[0]}, search,
        lambda name,row: updates.append(row), seals.append, targets)
    assert result['qualification_passed'] and result['task_wins'] == 4
    assert not result['depth_authorized']
    assert len(requests) == 24 and len(updates) == 12
    assert searches.count((8, 56, 50000)) == 12
    assert searches.count((0, 128, 100000)) == 4
    assert all(len(row['slots']) == 64 for row in updates)
    for i in range(4):
        blind, aware = requests[f'{i}_blind_repair'], requests[f'{i}_aware_repair']
        assert blind['seed'] == aware['seed'] == 36401+2*i
        assert len(json.loads(blind['messages'][1]['content'])['observations']) == 1
        feedback = json.loads(blind['messages'][-1]['content'])['public_execution_feedback']
        assert all(len(row) == 1 and row[0]['example_index'] == 0 for row in feedback['programs'])


def test_all_invalid_exact_initial_calls_no_execution_or_endpoint():
    calls, updates = [], []
    def request(tag, body):
        calls.append(tag)
        return 'bad json'
    def bomb(*args):
        raise AssertionError('invalid slot executed or closed endpoint opened')
    def search(proposals, n, cap):
        assert proposals == [None]*8
        return search_result(n, True)
    result = collect(cases(), DSL, request, bomb, bomb, search,
        lambda name,row: updates.append(row), bomb, bomb)
    assert result['status'] == 'initial_coverage_null'
    assert result['initial_covered'] == [False]*4
    assert len(calls) == 8 and len(updates) == 4
    assert all(len(row['slots']) == 64 for row in updates)


def test_failed_search_stops_before_extra_model_calls_or_targets():
    calls = []
    def request(tag, body):
        calls.append(tag)
        return 'bad json'
    def bomb(*args):
        raise AssertionError('unexpected continuation')
    with pytest.raises(ValueError, match='slot coverage'):
        collect(cases(), DSL, request, bomb, bomb,
            lambda *args: {'status': 'failed'}, bomb, bomb, bomb)
    assert len(calls) == 2


@pytest.mark.parametrize('change', ['short', 'reorder', 'refill'])
def test_search_coverage_rejected(change):
    result = search_result(56)
    if change == 'short':
        result['slots'].pop()
    elif change == 'reorder':
        result['slots'].reverse()
    else:
        result['replacement_candidates'] = 1
    with pytest.raises(ValueError, match='slot coverage'):
        checked_search(result, 56)
