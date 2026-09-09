from scripts.author_strings_overlap_audit import choose, compare


def test_selection_is_order_independent_and_excludes_opened():
    ids = list(map(str, range(1, 50)))
    selected = choose(ids)
    assert len(selected) == 12
    assert selected == choose(list(reversed(ids)))
    assert not {'1', '10'} & set(selected)


def test_overlap_requires_panel_not_single_generic_input():
    result = compare(['a', 'b', 'c', 'd', 'e'],
                     {'full': {'a', 'b', 'c', 'd'}, 'partial': {'a', 'z'}, 'tiny': {'a'}})
    assert result['exact_derivative_subsets'] == ['full']
    assert result['maximum_shared_inputs'] == 4
