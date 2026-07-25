from scripts.tau_knowledge_proposal_pooling_development import _average_ranks


def test_average_ranks_handles_ties() -> None:
    assert _average_ranks([10, 20, 20, 40]) == [1.0, 2.5, 2.5, 4.0]
