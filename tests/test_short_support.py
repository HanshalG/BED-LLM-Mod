from environments.program_induction.short_support import enumerate_two_statement
from environments.program_induction.local_support import evaluate
from environments.program_induction.prior import program_probability, statement_options
from scripts.deepcoder_opportunity import load_dsl


def test_exhaustive_source_count_and_mass():
    d=load_dsl()
    ps,work=enumerate_two_statement(d,[])
    initial=[('x0',list),('x1',list)]
    expected=sum(len(statement_options(d,initial+[('x2',op.output_type)],True))
                 for op,_ in statement_options(d,initial,False))
    assert len(ps)==len({str(p) for p in ps})==work['candidates']==expected
    # This is exactly one of three equiprobable source lengths, not all lengths.
    assert sum(program_probability(d,p) for p in ps)==__import__('fractions').Fraction(1,3)


def test_identity_history_retains_type_changing_counterhypothesis():
    d=load_dsl()
    h=[dict(inputs=[[-7,-8],[1]],output=[-7,-8])]
    ps,work=enumerate_two_statement(d,h)
    counter=[p for p in ps if str(p)=='x0 = INPUT | x1 = INPUT | x2 = Last x0 | x3 = Drop x2 x0']
    assert len(counter)==1
    assert evaluate(counter[0],[[4,1],[1]])==[1]
    assert all(evaluate(p,h[0]['inputs'])==h[0]['output'] for p in ps)
    assert work['history_executions']==work['candidates']
