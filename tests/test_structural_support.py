from environments.program_induction import constrained
from environments.program_induction.structural_support import insertions
from environments.program_induction.prior import program_probability
from scripts.deepcoder_opportunity import load_dsl, sample_program


def test_inserted_stage_and_later_references_are_valid():
    d = load_dsl()
    root, = constrained.decode(d, '{"programs":[{"statement":"x2 = ZipWith (*) x0 x1","next":{"statement":"x3 = Sum x2","next":null}}]}')
    pool = list(insertions(d, root))
    target, = constrained.decode(d, '{"programs":[{"statement":"x2 = ZipWith (*) x0 x1","next":{"statement":"x3 = Scanl1 (min) x2","next":{"statement":"x4 = Sum x3","next":null}}}]}')
    assert str(target) in {str(p) for p in pool}
    for seed in range(20):
        p = sample_program(d, 22000000+seed)
        edits = list(insertions(d, p))
        if len(p.statements) == 4:
            assert edits == []
        for edit in edits:
            assert len(edit.statements) == len(p.statements)+1
            assert program_probability(d, edit) > 0
            assert str(constrained.decode(d, __import__('json').dumps(constrained.encode(d,[edit])))[0]) == str(edit)
