import pytest
from scripts.bed_synthesis_equivalence_audit import diagnostic


def test_distinct_programs_survive_identical_demonstrations():
    r=diagnostic()
    assert r['retained_programs']==2
    assert r['retained_weights']==[.5,.5]
    assert r['correct_prior_expected_brier']==pytest.approx(.25)
    assert r['correct_after_query_expected_brier']==0
    assert r['query_value']==pytest.approx(.25)


def test_demonstration_merge_produces_false_certainty():
    r=diagnostic()
    assert r['merged_predicted_brier']==0
    assert r['merged_actual_expected_brier']==pytest.approx(.5)
