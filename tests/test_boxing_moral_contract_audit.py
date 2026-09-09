import pytest
from scripts.boxing_moral_contract_audit import audit


def test_changed_source_rejected_before_execution():
    with pytest.raises(ValueError, match='source mismatch'):
        audit(b"raise RuntimeError('must not execute')")
