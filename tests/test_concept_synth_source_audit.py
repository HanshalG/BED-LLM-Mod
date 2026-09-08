import pytest

from scripts.concept_synth_source_audit import inspect_source


def test_source_is_not_executed():
    source = '''
raise RuntimeError("must not execute")
def formula_library():
    return [FormulaTemplate("family", "key", "description", unknown())]
def _base_problem_description():
    return {"hiddenTarget": {"formula": secret()}, "seed": 1}
'''
    result = inspect_source(source)
    assert result['template_count'] == 1
    assert result['metadata_keys'] == ['formula', 'hiddenTarget', 'seed']
    assert result['source_executed'] is False
    assert result['endpoint_labels_loaded'] == 0


def test_changed_generator_shape_rejected():
    with pytest.raises(ValueError):
        inspect_source('def formula_library():\n    return generate_random_programs()')
