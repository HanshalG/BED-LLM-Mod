import pytest
from scripts.rearc_expression_run import ExpressionBank, ExpressionProbe
from scripts.rearc_expression_interface import response_format


def test_probe_caps_before_parent_transport(monkeypatch):
    monkeypatch.setattr('scripts.rearc_expression_run.FeedbackProbe.request',lambda *args: pytest.fail('HTTP path opened'))
    probe = object.__new__(ExpressionProbe)
    probe.report = {'calls':24}
    with pytest.raises(ValueError,match='cap'):
        probe.request('x',{'response_format':response_format()})
    probe.report = {'calls':0}
    with pytest.raises(ValueError,match='cap'):
        probe.request('x',{'response_format':{}})


def test_failed_operation_banked_and_replayed_without_execution(tmp_path):
    bank = ExpressionBank(tmp_path)
    def fail():
        raise RuntimeError('fixed failure')
    with pytest.raises(RuntimeError,match='fixed failure'):
        bank.saved('search_test',fail)
    assert (tmp_path/'search_test.error.json').exists()
    replay = ExpressionBank(tmp_path,replay=True)
    with pytest.raises(RuntimeError,match='fixed failure'):
        replay.saved('search_test',lambda: pytest.fail('operation repeated'))
