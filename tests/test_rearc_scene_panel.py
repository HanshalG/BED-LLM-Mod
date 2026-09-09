import json
import pytest
from scripts.rearc_scene_panel import collect, score

DSL = 'def identity(x: Any) -> Any:\n return x\n'
CODE = 'def transform(g): return g'


@pytest.mark.parametrize('fits', [True, False])
def test_calls_evidence_seed_pairing_and_sealed_null(fits):
    case = {'inputs': [[[1]], [[2]]], 'outputs': [[[1]], [[2 if fits else 8]]],
            'query_inputs': [[[3]]]*2, 'target_inputs': [[[3]]]*7,
            'target_hashes': ['sealed']*9}
    calls = {}; updates = []; events = []
    def request(tag, payload):
        calls[tag] = payload
        return json.dumps({f'p{i}': 'identity' for i in range(4)}
                          if tag.endswith('plan') else {'hypotheses': [CODE]*8})
    def seal(f):
        assert len(calls) == 24 and len(updates) == 8
        events.append('seal')
    def targets():
        assert events == ['seal']
        events.append('targets')
        return [[[[3]]]*9 for _ in range(4)]
    result = collect([case]*4, DSL, request, lambda code, xs: xs,
        lambda code, x: {'status': 'ok', 'output': x},
        lambda tag, value: updates.append(tag), seal, targets)
    assert len(calls) == 24
    for i in range(4):
        for j, stage in enumerate(('plan', 'compile', 'repair')):
            assert calls[f'{i}_raw_{stage}']['seed'] == calls[f'{i}_inventory_{stage}']['seed'] == 51400+3*i+j
    assert events == (['seal', 'targets'] if fits else [])
    assert result['endpoints_opened'] == fits
    assert not result['qualification_passed'] and not result['depth_authorized']


def test_support_and_proper_score_pass_does_not_authorize_depth():
    good = {'weights': [.5, .5], 'outputs': [[[[1]], [[2]]]]*9}
    bad = {'weights': [1.], 'outputs': [[None]]*9}
    result = score([{'inventory': good, 'raw': bad}]*4, [[[[1]]]*9]*4)
    assert result['qualification_passed'] and result['covered_query_answers'] == 8
    assert result['task_wins'] == 4 and not result['depth_authorized']
    assert result['brier_gain_decomposition'] == {'truth_mass': .5, 'concentration': .25}
    assert result['means']['raw']['all']['failure_probability'] == 1


def test_diffuse_wrong_predictions_cannot_pass_support():
    diffuse = {'weights': [.5, .5], 'outputs': [[[[2]], [[3]]]]*9}
    wrong = {'weights': [1.], 'outputs': [[[[2]]]]*9}
    result = score([{'inventory': diffuse, 'raw': wrong}]*4, [[[[1]]]*9]*4)
    assert result['gates']['paired_score']
    assert not result['qualification_passed'] and result['covered_query_answers'] == 0
    assert result['brier_gain_decomposition']['truth_mass'] == 0


def test_bad_case_never_dispatches():
    with pytest.raises(ValueError):
        collect([], DSL, None, None, None, None, None, None)
