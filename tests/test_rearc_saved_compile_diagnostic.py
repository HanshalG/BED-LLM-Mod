import json
from pathlib import Path
import pytest
from scripts.rearc_saved_compile_diagnostic import diagnose, audit, ROOT, RESULT_SHA

DSL = 'def identity(x: Any) -> Any:\n return x\n'


def test_precise_slot_errors_without_execution():
    result = diagnose(json.dumps({'hypotheses':[
        'identity(I)', 'identity(', '__bed_call1(I)', 'unknown(I)',
        'identity(I)', 'identity(I)', 'identity(I)', 'identity(I)']}), DSL)
    assert not result['batch_valid']
    assert [s.get('error') for s in result['slots'][:4]] == [
        None, 'expression syntax', 'computed call arity', 'unknown function']


def test_invalid_schema_does_not_partial_parse():
    result = diagnose('{"hypotheses":["identity(I)"]}', DSL)
    assert not result['batch_valid'] and result['slots'] == []


def test_audit_reads_only_terminal_and_fixed_raw_responses(monkeypatch):
    original = Path.read_bytes
    def guarded(path):
        assert path.parent == ROOT
        assert path.name == 'result.json' or path.name.endswith('.response.json')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    result = audit(ROOT, RESULT_SHA, DSL)
    assert len(result['rows']) == 24 and result['executions'] == 0
    assert not result['rescored']
    with pytest.raises(ValueError, match='terminal identity'):
        audit(ROOT, 'wrong', DSL)
