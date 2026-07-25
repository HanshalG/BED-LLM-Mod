from scripts.ambigdocs_gpt54_stability_smoke import (
    EXPECTED_REQUESTS,
    QUESTION_COUNT,
    REPEAT_COUNT,
)


def test_stability_request_budget_is_ten() -> None:
    assert QUESTION_COUNT == 4
    assert REPEAT_COUNT == 3
    assert EXPECTED_REQUESTS == 10
