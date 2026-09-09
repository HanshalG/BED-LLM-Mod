import json
from scripts.rearc_expression_panel import collect
from scripts.rearc_expression_update import update

DSL = 'def identity(x: Any) -> Any:\n return x\n'


def cases():
    return [{'inputs': [[[0]], [[1]], [[2]]], 'outputs': [[[0]], [[1]], [[2]]],
             'target_inputs': [[[3]] for _ in range(8)]} for _ in range(4)]


def test_initial_null_never_opens_targets_or_refresh():
    calls = []
    def request(tag, body):
        calls.append(tag)
        return '{"hypotheses":["identity(I)","identity(I)","identity(I)","identity(I)"]}'
    def bomb(*args):
        raise AssertionError('closed endpoint opened')
    result = collect(cases(), DSL, request, lambda g,x: [None]*len(x),
        lambda g,x: {'status': 'runtime_failed', 'returncode': 1},
        lambda g,n,cap: ['identity(I)']*n, lambda *args: None, bomb, bomb)
    assert result['status'] == 'initial_coverage_null'
    assert len(calls) == 8 and all('_initial_' in tag for tag in calls)


def test_complete_call_accounting_blind_privacy_and_sealing():
    requests, searches, seals = {}, [], []
    def request(tag, body):
        requests[tag] = body
        return json.dumps({'hypotheses': ['identity(I)']*4})
    def search(graphs, count, cap):
        searches.append((count, cap))
        return ['identity(I)']*count
    def targets():
        assert len(seals) == 1 and len(seals[0]) == 4
        return [[[[3]] for _ in range(8)] for _ in range(4)]
    result = collect(cases(), DSL, request, lambda g,x: x,
        lambda g,x: {'status': 'ok', 'output': x}, search,
        lambda *args: None, seals.append, targets)
    assert len(requests) == 24 and len(searches) == 16
    assert searches.count((56, 50000)) == 12 and searches.count((128, 100000)) == 4
    assert result['status'] == 'complete' and not result['qualification_passed']
    assert not result['depth_authorized']
    for i in range(4):
        blind = requests[f'{i}_blind_repair']
        assert blind['seed'] == requests[f'{i}_aware_repair']['seed']
        assert len(json.loads(blind['messages'][1]['content'])['observations']) == 1
        feedback = json.loads(blind['messages'][-1]['content'])['public_execution_feedback']
        assert all(len(row) == 1 and row[0]['example_index'] == 0 for row in feedback['programs'])


def test_malformed_batch_still_uses_exact_fixed_repair_slots():
    calls = []
    def request(phase, messages):
        calls.append(phase)
        return 'not json'
    result = update(inputs=[[[0]]], observations=[{'index': 0, 'output': [[0]]}],
        dsl_source=DSL, request=request,
        diagnose=lambda *args: (_ for _ in ()).throw(AssertionError('invalid graph executed')))
    assert calls == ['proposal', 'repair']
    assert result['slots'] == [None]*8 and result['graphs'] == []
