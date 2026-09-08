import hashlib
import json
import pytest

from environments.program_induction import independent_joint_screen as s


def test_full_gate_and_failures_before_labels(tmp_path):
    law=dict(interpretation='restricted_syntax_prior_not_full_posterior',worlds=[
        dict(key='p',weight=1.,answers=['1'],targets=['2']*32)])
    good=dict(distributions=[{'2':1.}]*32)
    bad=dict(distributions=[{'3':1.}]*32)
    panel={str(i):dict(case=dict(query=[[1],[2]],targets=[[[1],[2]]]*32),
        observation=dict(inputs=[[1],[2]],output=1),teachers={t:law for t in s.TEACHERS},
        forecasts={a:bad if a=='repeat' else good for a in s.UPDATERS}) for i in range(8)}
    labels={k:dict(target_inputs=r['case']['targets'],outputs=[2]*32) for k,r in panel.items()}
    path=tmp_path/'panel.json'
    def score():
        path.write_text(json.dumps(panel))
        return s.score_sealed(path,hashlib.sha256(path.read_bytes()).hexdigest(),lambda:labels)
    result=score()
    assert result['joint_gate'] and result['updater_ranking_gate'] and result['multi_query_screen_allowed']
    assert not result['depth_authorized'] and not result['scientific_pass']
    panel['0']['teachers']['ab']=None
    assert not score()['joint_gate']
    def bomb():
        raise AssertionError('labels opened')
    with pytest.raises(ValueError,match='seal'):
        s.score_sealed(path,'wrong',bomb)
    panel.pop('0')
    path.write_text(json.dumps(panel))
    with pytest.raises(ValueError,match='eight'):
        s.score_sealed(path,hashlib.sha256(path.read_bytes()).hexdigest(),bomb)
