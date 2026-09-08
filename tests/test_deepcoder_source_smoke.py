"""Hash-pinned source smoke on explicit fixtures, never the scientific seeds."""
from scripts.deepcoder_opportunity import load_dsl, outcome, sample_program, sample_input


def test_upstream_interpreter_fixture():
    dsl = load_dsl()
    program = dsl.Program(['x0', 'x1'], [dsl.Statement.from_str('x2 = Reverse x0')])
    assert outcome(program, [[1, 2, 3], [9]]) == '{"type":"list","value":[3,2,1]}'
    program = dsl.Program(['x0', 'x1'], [dsl.Statement.from_str('x2 = Head x0')])
    assert outcome(program, [[], [9]]) == 'ERROR'


def test_sampling_structural_only():
    dsl = load_dsl()
    for seed in range(10):
        program = sample_program(dsl, seed)
        assert str(program) == str(sample_program(dsl, seed))
        assert 2 <= len(program.statements) <= 4
        for previous, current in zip(program.statements, program.statements[1:]):
            assert previous.variable in current.args
        assert sample_input(seed) == sample_input(seed)
        assert all(1 <= len(xs) <= 5 and all(-10 <= x <= 10 for x in xs)
                   for xs in sample_input(seed))
