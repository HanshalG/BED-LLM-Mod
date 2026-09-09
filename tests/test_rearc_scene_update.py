import json
import pytest

from scripts.rearc_scene_update import scene_update

DSL = 'def identity(x: Any) -> Any:\n return x\n'
CODE = 'def transform(g): return g'


@pytest.mark.parametrize('order', [('raw', 'inventory'), ('inventory', 'raw')])
def test_equal_calls_slots_history_and_observed_only_facts(order):
    prompts = {}
    bank = []
    def request(stage, prompt, fmt):
        prompts[stage] = prompt
        return json.dumps({f'p{i}': 'identity' for i in range(4)}
            if stage.endswith('plan') else {'hypotheses': [CODE] * 8})
    result = scene_update(inputs=[[[1]], [[2]], [[3]]],
        observations=[{'index': 0, 'output': [[1]]}, {'index': 1, 'output': [[2]]}],
        dsl_source=DSL, request=request,
        diagnose=lambda code, x: {'status': 'ok', 'output': x},
        bank_update=lambda arm, value: bank.append(arm), order=order)
    assert len(prompts) == result['calls'] == 6
    assert bank == list(order)
    assert all(len(r['slots']) == 16 for r in result['arms'].values())
    for stage in ('plan', 'compile', 'repair'):
        raw = json.loads(prompts['raw_' + stage][1]['content'])
        inv = json.loads(prompts['inventory_' + stage][1]['content'])
        facts = inv.pop('public_scene_facts')
        assert inv == raw
        assert [r['index'] for r in facts['observed_outputs']] == [0, 1]
        assert [r['index'] for r in facts['inputs']] == [0, 1, 2]


def test_invalid_history_never_dispatches():
    calls = []
    with pytest.raises(ValueError):
        scene_update(inputs=[[[1]]], observations=[{'index': 5, 'output': [[9]]}],
            dsl_source=DSL, request=lambda *a: calls.append(a),
            diagnose=lambda *a: None, bank_update=lambda *a: None)
    assert not calls


def test_oversize_inventory_stops_before_http(monkeypatch):
    import scripts.rearc_scene_update as module
    monkeypatch.setattr(module, 'grid_inventory', lambda g: {'large': 'x' * 65536})
    calls = []
    with pytest.raises(ValueError, match='message budget'):
        scene_update(inputs=[[[1]]], observations=[{'index': 0, 'output': [[1]]}],
            dsl_source=DSL, request=lambda *a: calls.append(a),
            diagnose=lambda *a: None, bank_update=lambda *a: None,
            order=('inventory', 'raw'))
    assert not calls
