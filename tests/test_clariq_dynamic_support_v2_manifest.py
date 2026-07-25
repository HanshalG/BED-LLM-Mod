from scripts.clariq_dynamic_support_manifest import (
    DEVELOPMENT_TOPIC_IDS,
    SEALED_HOLDOUT_TOPIC_IDS,
)
from scripts.clariq_dynamic_support_v2_manifest import (
    EXPECTED_BRANCH_COUNT,
    EXPECTED_FULL_TREE_REQUESTS,
    EXPECTED_ROOT_COUNT,
    V1_MECHANICS_TOPIC_ID,
    V2_MECHANICS_TOPIC_ID,
)


def test_v2_uses_a_fresh_mechanics_topic_and_preserves_sealed_cells() -> None:
    assert V2_MECHANICS_TOPIC_ID == "60"
    assert V2_MECHANICS_TOPIC_ID != V1_MECHANICS_TOPIC_ID
    assert V2_MECHANICS_TOPIC_ID not in DEVELOPMENT_TOPIC_IDS
    assert V2_MECHANICS_TOPIC_ID not in SEALED_HOLDOUT_TOPIC_IDS
    assert EXPECTED_ROOT_COUNT == 15
    assert EXPECTED_BRANCH_COUNT == 90
    assert EXPECTED_FULL_TREE_REQUESTS == 91
