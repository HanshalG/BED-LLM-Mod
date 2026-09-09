from scripts.author_strings_opportunity import actual,gate,measure
from environments.program_induction.reference import ProgramReference


def test_zero_uncertainty_is_null_and_complete_budget():
    r=measure([['a']*9],['a']*6,['a']*3)
    assert r['status']=='complete' and set(r['expected_risk'].values())=={0.}
    assert all(len(p['trace'])==3 and p['brier']==0 for p in r['actual_paths'].values())
    assert not gate({'1':r,'10':r})['passed']


def test_missing_actual_answer_abstains_without_invented_branch():
    ref=ProgramReference([['a']*9],6)
    try:
        r=actual(ref,['b']*6,['a']*3,2)
        assert r['abstained'] and r['brier']==1 and len(r['trace'])==1
    finally:
        ref.clear()


def test_empty_reference_does_not_pass():
    r=measure([],['a']*6,['a']*3)
    assert not gate({'1':r,'10':r})['passed']
