from scripts.robustfill_coverage_audit import search


def test_joint_concat_witness():
    result = search({('ab', 'x'): 'a', ('c', 'yz'): 'b'}, ('abc', 'xyz'))
    assert result['status'] == 'witness'
    assert result['path'] == ('a', 'b')


def test_cannot_mix_programs_across_examples():
    assert search({('a', 'wrong'): 'a', ('wrong', 'b'): 'b'}, ('a', 'b'))['status'] == 'no_witness_in_declared_subset'


def test_empty_and_bounded_search():
    assert search({('', ''): 'empty'}, ('x', 'y'))['status'] == 'no_witness_in_declared_subset'
    assert search({('a',): 'a'}, ('aa',), max_parts=1)['status'] == 'no_witness_in_declared_subset'
    assert search({('a',): 'a'}, ('aaa',), max_states=1)['status'] == 'incomplete_state_cap'
