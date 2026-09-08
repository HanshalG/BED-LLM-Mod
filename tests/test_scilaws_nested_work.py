import pytest

from environments.scilaws.adaptive_reference import AdaptiveReference
from scripts.scilaws_mixed_refinement_audit import fixture
from scripts.scilaws_nested_work_audit import TracedReference, summarize


def test_tracing_preserves_value_and_evaluation_counts():
    m, state = fixture(8, ())
    plain = AdaptiveReference(m, predictive_coordinates=True)
    traced = TracedReference(m, predictive_coordinates=True)
    assert traced.terminal(state, 0) == pytest.approx(plain.terminal(state, 0))
    assert traced.evaluations == plain.evaluations
    assert traced.records[0]['evaluations'] == traced.evaluations
    assert traced.level == 0


def test_summarize_does_not_invent_missing_action_scores():
    m, _ = fixture(8, ())
    ref = TracedReference(m)
    ref.records = [dict(outer_callback=1, observation=0, action=0, value=1,
                        evaluations=30, status='completed')]
    result = summarize(ref)
    assert result['completed_terminal_calls'] == 1
    assert not result['decisions']
    assert not result['sampled_switch_intervals']
