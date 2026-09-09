import pytest

from scripts.bugsinpy_contract_audit import inventory, legacy_accepts_output


def test_mixed_outcome_is_not_reliable_success():
    assert legacy_accepts_output('1 failed, 3 passed in 0.10s')
    assert legacy_accepts_output('3 passed in 0.10s')
    assert not legacy_accepts_output('1 failed in 0.10s')


def test_inventory_never_needs_task_contents():
    tree = {'truncated': False, 'tree': [
        {'path': 'projects/p/bugs/1/bug.info', 'type': 'blob'},
        {'path': 'projects/p/bugs/1/run_test.sh', 'type': 'blob'},
        {'path': 'LICENSE', 'type': 'blob'}]}
    result = inventory(tree)
    assert result['bug_count'] == 1
    assert result['projects'] == ['p']
    assert result['root_license_paths'] == ['LICENSE']
    with pytest.raises(ValueError, match='incomplete'):
        inventory({'truncated': True})
