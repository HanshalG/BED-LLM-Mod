import pytest

from scripts.black_prefixed_runtime_smoke import source_paths


def test_only_preselected_runtime_files_not_test_bank():
    paths = ['black.py', 'LICENSE', 'blib2to3/pgen2/driver.py', 'blib2to3/Grammar.txt',
             'tests/test_black.py', 'setup.py']
    tree = {'truncated': False, 'tree': [{'type': 'blob', 'path': p} for p in paths]}
    assert source_paths(tree) == sorted(paths[:4])
    tree['tree'].append({'type': 'blob', 'path': 'blib2to3/../../escape.py'})
    with pytest.raises(ValueError):
        source_paths(tree)


def test_reject_incomplete_inventory():
    with pytest.raises(ValueError):
        source_paths({'truncated': True})
