import pytest

from scripts.bugsinpy_boundary_audit import inspect_functions


def test_inspection_does_not_execute_top_level():
    raw = b'raise RuntimeError("never execute")\n@decorator\ndef f(command, settings):\n return command.script.lower()\n'
    result = inspect_functions(raw, ['f'])['f']
    assert result['unused_arguments_syntactically'] == ['settings']
    assert result['command_attributes'] == ['script']
    assert result['decorators'] == ['decorator']
    assert result['call_targets'] == ['command.script.lower']


def test_missing_function_rejected():
    with pytest.raises(ValueError):
        inspect_functions(b'x = 2', ['missing'])
