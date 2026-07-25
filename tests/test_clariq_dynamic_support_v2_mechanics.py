from scripts.clariq_dynamic_support_v2_mechanics import (
    BRANCH_PROFILE_DIVERSITY_MINIMUM,
    BRANCH_SUPPORT_CHANGE_MINIMUM,
    EXPECTED_REQUESTS,
    MAX_COST_USD,
    POSITIVE_CONTINUATION_MINIMUM,
)


def test_v2_mechanics_frozen_counts_and_caps() -> None:
    assert EXPECTED_REQUESTS == 91
    assert BRANCH_SUPPORT_CHANGE_MINIMUM == 80
    assert BRANCH_PROFILE_DIVERSITY_MINIMUM == 75
    assert POSITIVE_CONTINUATION_MINIMUM == 75
    assert MAX_COST_USD == 1.25
