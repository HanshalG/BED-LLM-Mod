import pytest

from scripts.zendoworld_source_contract_audit import extract_methods, reproduce


def test_extraction_does_not_execute_imports_or_other_methods():
    source = """
raise RuntimeError("top level executed")
import nonexistent_model_client
class ZendoStateGameMaster:
    def __init__(self, program, index, dataset, paths, dsl, cfg, images=False):
        self.remaining_examples = [((dataset[0][0], False), paths[0])]
        print("Label mismatch")
    def label_input(self, scene):
        return True
    def paid_call(self):
        raise RuntimeError("not allowed")
"""
    master = extract_methods(source)
    assert not hasattr(master, "paid_call")
    assert reproduce(master)["disagreement_retained"]


def test_missing_method_rejected():
    with pytest.raises(ValueError, match="missing required"):
        extract_methods("class ZendoStateGameMaster:\n def __init__(self): pass")


def test_changed_contract_rejected():
    class Consistent:
        def __init__(self, *args, **kwargs):
            self.remaining_examples = [((None, True), None)]

        def label_input(self, scene):
            return True

    with pytest.raises(AssertionError, match="differs"):
        reproduce(Consistent)
