import pytest

from scripts.newtonbench_observation_contract_audit import functions


def test_function_extraction_does_not_execute_module_level_effects():
    source = "raise RuntimeError('unwanted import')\ndef selected(x: MissingAnnotation):\n    return x + 1\n"
    selected = functions(source, {"selected"}, {})["selected"]
    assert selected(2) == 3


def test_missing_function_fails_closed():
    with pytest.raises(ValueError, match="missing"):
        functions("def other(): pass", {"selected"}, {})


def test_namespace_is_copied_and_unselected_functions_absent():
    namespace = {"value": 4}
    selected = functions(
        "def chosen(): return value\ndef unused(): raise RuntimeError()",
        {"chosen"},
        namespace,
    )
    assert selected["chosen"]() == 4
    assert namespace == {"value": 4}
    assert "unused" not in selected["chosen"].__globals__
