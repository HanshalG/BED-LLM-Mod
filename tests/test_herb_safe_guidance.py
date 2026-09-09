import copy
import json
from pathlib import Path
import pytest
from scripts.herb_safe_guidance import prepare, search
from scripts.herb_search_guidance import weights


def metadata():
    root = Path(__file__).resolve().parents[1]/'results/nonmyopic/herb_full_grammar_20260909'
    data = json.loads((root/'source.json').read_text())
    data['rules'] = [r.strip() for r in (root/'grammar.jl').read_text().splitlines()
                     if r.strip().startswith('Value = ')]
    return data


def graph(op, args):
    return {'steps': [{'id': 'x0', 'op': op, 'args': args}], 'output': 'x0'}


def test_reject_whole_program_preserve_slots_and_valid_weights():
    data = metadata()
    good = graph('vmirror', ['I'])
    bad = {'steps': good['steps'] + [
        {'id': 'x1', 'op': 'lbind', 'args': ['recolor', 'THREE', 'outbox']}], 'output': 'x1'}
    slots = [bad, good, None, graph('eval', ['I']), good]
    before = copy.deepcopy(slots)
    result = prepare(data, slots)
    assert slots == before
    assert [r['eligible'] for r in result['slots']] == [False, True, False, False, True]
    assert result['slots'][0]['reason'] == 'unsupported_call_arity'
    assert result['graphs'] == [good, good]
    assert result['replacement_proposals'] == 0
    assert (result['base'], result['guided']) == weights(data, [good, good])
    assert all(g >= .5*b > 0 for b, g in zip(result['base'], result['guided']))


def test_every_export_arity_and_all_invalid_fallback():
    data = metadata()
    for signature in data['signatures']:
        for arity in range(5):
            op = signature['name']
            rule = f"Value = {op}({', '.join(['Value']*arity)})"
            result = prepare(data, [graph(op, ['I']*arity)])
            assert result['slots'][0]['eligible'] == (rule in data['rules'])
    result = prepare(data, [None, graph('lbind', ['I', 'I', 'I'])])
    assert result['guided'] == weights(data, [])[0]
    assert len(result['slots']) == 2


def test_computed_callable_supported_and_zero_arg_rejected():
    good = {'steps': [{'id': 'x0', 'op': 'lbind', 'args': ['recolor', 'THREE']},
                     {'id': 'x1', 'op': 'x0', 'args': ['I']}], 'output': 'x1'}
    assert prepare(metadata(), [good])['slots'][0]['eligible']
    good['steps'][1]['args'] = []
    assert not prepare(metadata(), [good])['slots'][0]['eligible']


def test_search_does_not_refill_or_increase_budget(monkeypatch):
    calls = []
    def fake(graphs, count, cap):
        calls.append((graphs, count, cap))
        return {'status': 'complete'}
    monkeypatch.setattr('scripts.herb_search_runtime.search', fake)
    result = search([None]*8, 56, 50000)
    assert calls == [([], 56, 50000)]
    assert len(result['guidance_slots']) == 8
    assert result['replacement_proposals'] == 0


def test_corrupt_metadata_remains_fatal():
    data = metadata()
    data['rules'].append(data['rules'][0])
    with pytest.raises(ValueError, match='duplicate'):
        prepare(data, [None])


def test_search_failure_is_not_retried_or_silently_replaced(monkeypatch):
    calls = []
    def fail(*args):
        calls.append(args)
        raise ValueError('unknown callable or forward reference')
    monkeypatch.setattr('scripts.herb_search_runtime.search', fail)
    with pytest.raises(ValueError, match='unknown callable'):
        search([None], 56, 50000)
    assert calls == [([], 56, 50000)]


def test_full_grammar_computed_nongrid_terminal_is_not_convertible():
    from scripts.herb_candidate_expression import to_graph
    data = metadata()
    functions = {s['name'] for s in data['signatures']}
    terminals = {r.removeprefix('Value = ') for r in data['rules'] if '(' not in r}
    assert 'Value = __bed_call1(Value, Value)' in data['rules']
    with pytest.raises(ValueError, match='unknown callable'):
        to_graph('__bed_call1(I, I)', functions, terminals-functions-{'I'})
