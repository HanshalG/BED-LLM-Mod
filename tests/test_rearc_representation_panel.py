import copy
import json
import pytest
from scripts.rearc_representation_panel import collect

DSL = 'def identity(x: Any) -> Any:\n return x\n'
CODE = 'def transform(grid):\n return grid\n'


@pytest.mark.parametrize('fits', [True, False])
def test_real_controller_forecast_seals_before_labels_and_exact_budget(fits):
    case = {'inputs':[[[1]]], 'outputs':[[[1]]], 'query_inputs':[[[2]],[[3]]],
            'target_inputs':[[[4]]]*8, 'target_hashes':['sealed']*10}
    cases = [copy.deepcopy(case) for _ in range(6)]
    calls, updates, events = [], [], []
    def request(name, payload):
        calls.append((name,payload))
        if name.endswith('_plan'):
            return json.dumps({f'p{i}':'identity' for i in range(4)})
        return json.dumps({'hypotheses':[CODE if '_python_' in name else 'identity(I)']*8})
    def diagnose(p,x):
        assert x == [[1]]
        return {'status':'ok','output':x if fits else [[0]]}
    def evaluate(p,inputs):
        return inputs if fits else [[[0]] for _ in inputs]
    def seal(rows):
        assert len(updates)==6 and len(calls)==30
        assert all(len(row[a]['outputs'])==10 for row in rows for a in ('python','dsl'))
        events.append('sealed')
    def targets():
        assert events == ['sealed']
        events.append('labels')
        return [c['query_inputs']+c['target_inputs'] for c in cases]
    result = collect(cases,DSL,request,evaluate,evaluate,diagnose,diagnose,
        lambda name,value:updates.append((name,value)),seal,targets)
    assert len(calls)==30 and len(updates)==6
    for i in range(6):
        subset = {n:b for n,b in calls if n.startswith(str(i)+'_')}
        assert subset[f'{i}_python_compile']['seed']==subset[f'{i}_dsl_compile']['seed']
        assert subset[f'{i}_python_repair']['seed']==subset[f'{i}_dsl_repair']['seed']
        assert all(b['reasoning']['effort']=='medium' for b in subset.values())
    assert result['endpoints_opened'] == fits
    assert not result['depth_authorized']
    if fits:
        assert events == ['sealed','labels']
        assert result['means']['python']['all']['whole_grid_brier'] == 0
        assert not result['qualification_passed']  # Ties/saturation are not a win.
    else:
        assert not events and result['status']=='initial_coverage_null'


def test_invalid_coverage_opens_nothing():
    with pytest.raises(ValueError, match='coverage'):
        collect([],DSL,None,None,None,None,None,None,None,None)
