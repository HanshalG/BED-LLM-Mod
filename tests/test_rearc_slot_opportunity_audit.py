import math
from pathlib import Path
import pytest
from scripts.rearc_slot_opportunity_audit import distribution, audit, ROOT


def test_deterministic_wrong_prediction_is_not_uncertainty():
    result = distribution([[[0]],[[0]]],[[1]])
    assert result['entropy_nats']==0
    assert result['observed_output_probability']==0
    assert result['distinct_outputs']==1


def test_disagreement_and_failure_mass():
    result = distribution([[[0]],None],[[0]])
    assert result['entropy_nats']==pytest.approx(math.log(2))
    assert result['observed_output_probability']==.5
    assert result['failure_probability']==.5
    assert distribution([],[[0]])=={'status':'empty_support'}


def test_bank_audit_never_executes_or_reads_target_values(monkeypatch):
    import subprocess
    import socket
    def bomb(*args,**kwargs):
        raise AssertionError('new execution/network')
    monkeypatch.setattr(subprocess,'run',bomb)
    monkeypatch.setattr(subprocess,'check_output',bomb)
    monkeypatch.setattr(socket,'create_connection',bomb)
    original = Path.read_text
    def read(self,*args,**kwargs):
        if self.name=='targets.json':
            raise AssertionError('target labels read')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    result = audit(ROOT)
    assert len(result['rows'])==4
    assert result['new_program_executions']==result['new_model_calls']==0
    assert not result['qualification_passed'] and not result['depth_authorized']
