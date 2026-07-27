from scripts.hotpot_link_restricted_v2_split import (
    V2_CONFIRMATION_SIZE,
    V2_DEVELOPMENT_SIZE,
)


def test_v2_split_sizes_preserve_two_thousand_rows() -> None:
    assert V2_DEVELOPMENT_SIZE + V2_CONFIRMATION_SIZE == 2_000
