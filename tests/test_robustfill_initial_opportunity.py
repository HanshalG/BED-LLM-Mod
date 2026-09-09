import random
from scripts.robustfill_initial_opportunity import draw_rows


def test_exact_length_weights_and_concatenation():
    groups = {'a': [(('x',), 1)], 'b': [(('y',), 1)]}
    rows, meta = draw_rows(groups, 'ab', 2, random.Random(1), count=8)
    assert rows == [['xy']] * 8
    assert meta['length_integer_weights'] == [0, 2, 0]


def test_empty_pieces_counted_and_reproducible():
    groups = {'': [(('',), 1)], 'a': [(('x',), 1)]}
    a = draw_rows(groups, 'a', 2, random.Random(42))
    assert a == draw_rows(groups, 'a', 2, random.Random(42))
    assert a[1]['length_integer_weights'] == [4, 4, 3]
    assert a[0] == [['x']] * 256


def test_no_fallback_and_syntax_multiplicity():
    assert draw_rows({'a': [(('x',), 1)]}, 'b', 1, random.Random(0))[0] == []
    rows, meta = draw_rows({'a': [(('x',), 2), (('y',), 1)]}, 'a', 3, random.Random(0))
    assert meta['length_integer_weights'] == [27, 0, 0]
    assert set(map(tuple, rows)) == {('x',), ('y',)}
