from scripts.deepcoder_opportunity import load_dsl
from environments.program_induction.constrained import decode
from environments.program_induction.local_support import expand, evaluate
from environments.program_induction.prior import program_probability


def test_one_edit_support_is_source_valid_and_retains_roots():
    dsl = load_dsl()
    roots = decode(dsl, '{"programs":[{"statement":"x2 = Reverse x0","next":{"statement":"x3 = Head x2","next":null}}]}')
    history = [{'inputs': [[1], [2]], 'output': 1}]
    pool, work = expand(dsl, roots, history)
    assert str(roots[0]) in {str(p) for p in pool}
    assert len(pool) > 1 and work['compatible_roots'] == 1
    for p in pool:
        assert program_probability(dsl, p) > 0
        assert evaluate(p, history[0]['inputs']) == 1
        assert len(p.statements) == 2
        assert sum(str(a) != str(b) for a, b in zip(p.statements, roots[0].statements)) <= 1
    assert expand(dsl, list(roots)*2, history)[1] == work
    assert expand(dsl, roots, [{'inputs': [[1], [2]], 'output': 42}])[0] == []
