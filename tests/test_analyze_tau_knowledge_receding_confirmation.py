from scripts.analyze_tau_knowledge_receding_confirmation import _interval


def test_interval_uses_two_sided_95_percent_quantiles():
    values = list(range(101))
    assert _interval(values) == [2.5, 97.5]
