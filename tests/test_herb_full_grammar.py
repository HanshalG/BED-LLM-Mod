import pytest
import json
from pathlib import Path
import tomllib
from scripts.herb_full_grammar import compile_grammar


def test_permissive_exports_and_computed_calls():
    grammar, report = compile_grammar('K = 2\ndef identity(x: Any) -> Any:\n return x\ndef compose(a: Callable, b: Callable) -> Callable:\n return a\n')
    assert 'Value = compose(Value, Value)' in grammar
    assert 'Value = compose\n' in grammar
    assert 'Value = K\n' in grammar
    assert 'Value = __bed_call4(Value, Value, Value, Value, Value)' in grammar
    assert report['function_count'] == 2
    assert report['direct_one_call_combinations'] == 4 + 16


def test_no_silent_signature_drop():
    with pytest.raises(ValueError, match='signature'):
        compile_grammar('def f(x: Any, *args) -> Any:\n return x\n')


def test_full_export_artifact_and_computed_callable():
    root = Path(__file__).resolve().parents[1] / 'results/nonmyopic/herb_full_grammar_20260909'
    source = json.loads((root / 'source.json').read_text())
    grammar = (root / 'grammar.jl').read_text()
    assert source['function_count'] == len(source['signatures']) == 160
    assert source['constant_count'] == 28
    for signature in source['signatures']:
        assert f"Value = {signature['name']}\n" in grammar
        assert f"Value = {signature['name']}(" in grammar
    result = tomllib.loads((root / 'candidates.toml').read_text())
    assert len(result['rows']) == 256
    fixture = result['computed_callable_fixture']
    assert fixture['steps'] == [
        {'id': 'x0', 'op': 'compose', 'args': ['identity', 'identity']},
        {'id': 'x1', 'op': 'x0', 'args': ['I']},
    ]
