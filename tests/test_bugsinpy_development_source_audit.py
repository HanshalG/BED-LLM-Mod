import pytest

from scripts.bugsinpy_development_source_audit import parse_info, select


def test_selection_is_order_invariant_and_one_per_project():
    paths = [f'projects/{p}/bugs/{i}/bug.info'
             for p in ('black', 'cookiecutter', 'thefuck') for i in (1, 2, 3)]
    assert select(paths) == select(list(reversed(paths)))
    assert len(select(paths)) == 3
    with pytest.raises(ValueError):
        select([])


def test_metadata_not_shell():
    assert parse_info(b'a="$(echo never executed)"') == {'a': '$(echo never executed)'}
    with pytest.raises(ValueError):
        parse_info(b'a=1\na=2')
