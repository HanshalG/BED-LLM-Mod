import pytest
from scripts.robustfill_cohort_opportunity import public_case, summarize


def data():
    bk = ''.join(f"in(e{i},1,'{chr(64+i)}').width(e{i},1)." for i in range(1,12))
    exs = ''.join(f"pos(out(e{i},1,'x'))." for i in range(1,12))
    return bk, exs


def test_numeric_roles_and_no_later_label_in_public_case():
    bk, exs = data()
    a = public_case(bk, exs)
    assert a == public_case(bk, exs.replace("out(e2,1,'x')", "out(e2,1,'y')"))
    assert a['example_ids'] == [f'e{i}' for i in range(1,11)]
    assert a['extra_examples'] == 1


def test_invalid_source_fails_and_incomplete_not_dropped():
    bk, exs = data()
    with pytest.raises(ValueError):
        public_case(bk, exs+"neg(out(e2,1,'x')).")
    assert summarize({'a': {'status':'empty_support'}})['full_cohort_means'] is None


def test_reversal_retained():
    result = summarize({'a':dict(status='complete', expected_risk=dict(h1=.1,h2=.2,h3=.05,random=.3,openloop=.1))})
    assert result['nonmonotonic_tasks'] == ['a']
    assert result['strict_h2_improvements'] == 0
    assert result['strict_h3_improvements'] == 1
