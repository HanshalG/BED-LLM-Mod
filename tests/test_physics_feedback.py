import json
import numpy as np
import pytest
from environments.program_induction.physics_feedback import calibrate, feedback, guards, revision_messages, calibrated_prediction


def test_evidence_matches_dense_normal():
    r=np.array([.1,.3,-.2])
    cov=.05**2*np.eye(3)+4*np.ones((3,3))
    expected=-.5*(3*np.log(2*np.pi)+np.linalg.slogdet(cov)[1]+r@np.linalg.solve(cov,r))
    assert calibrate(r)['log_evidence']==pytest.approx(expected,abs=1e-9)


def test_guard_exposes_unobserved_domain_failure():
    report=feedback(['sqrt(x0-4)'],['N'],[({'N':5},0.)])[0]
    assert report['invalid_history_count']==0
    assert report['invalid_guard_count']>0
    assert not report['domain_guarantee']
    assert all(type(p['N']) is int for p in guards(['N']))


def test_revision_uses_only_supplied_history():
    args=(['a'],'context',{'a':'scale'})
    a=revision_messages(*args,[({'a':1.},0.)],['x0'])
    b=revision_messages(*args,[({'a':1.},1.)],['x0'])
    x,y=[json.loads(v[1]['content']) for v in (a,b)]
    assert x['initial_proposal_diagnostics']!=y['initial_proposal_diagnostics']
    assert x['public_domain']==y['public_domain']


def test_calibration_and_invalid_support():
    a=calibrated_prediction(['1','(1)'],['a'],[({'a':1.},2.)],[{'a':2.}])
    assert len(a['weights'])==1
    assert a['mean'][0]==pytest.approx(2*4/(4+.05**2))
    assert calibrated_prediction(['-1'],['a'],[({'a':1.},0.)],[{'a':2.}])['status']=='empty_support'
