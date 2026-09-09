from environments.string_induction.finite import operations, support


def test_complete_prior_multiplicities_and_all_total():
    ops=operations()
    assert len({n for n,f in ops})==len(ops)
    for _,f in ops:
        assert isinstance(f(''),str)
        assert isinstance(f('A, b@2.-/_'),str)
    rows,work=support(('',''),['A, b@2'])
    assert work['attempted']==len(ops)+len(ops)**2
    assert work['prior_units']==work['unconditioned_prior_units']==2*len(ops)**2
    assert len(rows)==2*len(ops)**2


def test_conditioning_only_initial_pair():
    rows,work=support(('Alice Brown','AB'),['Bob Cole','c d'])
    assert rows and work['complete_declared_grammar']
    assert ['BC','cd'] in rows
    assert not work['complete_human_task_prior']
