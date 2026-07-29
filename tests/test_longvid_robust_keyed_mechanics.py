from __future__ import annotations

import pytest

from scripts import longvid_robust_keyed_mechanics as robust
from scripts.longvid_contrastive_path_belief_mechanics import layout_hash_for


def _support_rows() -> list[str]:
    return [
        (
            f"H{index}|{10 + index}|anchor{index}|"
            f"anchor{index} query {index}|distinct hypothesis {index}"
        )
        for index in range(1, 7)
    ]


def test_keyed_support_accepts_arbitrary_complete_order() -> None:
    support = robust.parse_keyed_support("\n".join(reversed(_support_rows())))
    assert [row["weight"] for row in support] == [11, 12, 13, 14, 15, 16]
    assert support[0]["anchor"] == "anchor1"


@pytest.mark.parametrize(
    "text",
    [
        "\n".join(_support_rows()[:-1]),
        "\n".join([*_support_rows()[:-1], _support_rows()[0]]),
        "\n".join(_support_rows()).replace("H6|16", "H7|16"),
        "\n".join(_support_rows()).replace("H6|16", "H6|0"),
    ],
)
def test_keyed_support_rejects_incomplete_or_invalid_rows(text: str) -> None:
    with pytest.raises(ValueError):
        robust.parse_keyed_support(text)


def test_keyed_rank_parser_is_exact() -> None:
    assert robust.parse_keyed_rank("R|B|79|missing causal bridge") == {
        "choice": "B",
        "confidence": 79,
        "unresolved_need": "missing causal bridge",
    }
    with pytest.raises(ValueError):
        robust.parse_keyed_rank("B|79|missing causal bridge")


def test_mechanics_layout_is_stable_and_fresh() -> None:
    assert (
        layout_hash_for(robust.MECHANICS_TASK_LAYOUT)
        == robust.MECHANICS_TASK_LAYOUT_HASH
    )
    assert not set(robust.EXCLUDED_PRIOR_ROWS) & {
        row for row, _ in robust.MECHANICS_TASK_LAYOUT
    }


def test_dry_smoke_uses_exact_ten_requests(tmp_path) -> None:
    from helpers import Config

    config = Config()
    config.openrouter_concurrency = 8
    model = robust.KeyedFixtureModel()
    payload = robust.smoke.run_smoke(
        config,
        raw_path=tmp_path / "raw.json",
        model=model,
        support_message_builder=robust.keyed_support_messages,
        rank_message_builder=robust.smoke_rank_messages,
        support_parser=robust.parse_keyed_support,
        rank_parser=robust.parse_keyed_rank,
        interface_version=robust.INTERFACE_VERSION,
        response_format_name="order_insensitive_keyed_pipe_rows",
        max_transport_retries=robust.MAX_TRANSPORT_RETRIES,
    )

    assert payload["status"] == "passed"
    assert payload["usage"]["physical_requests"] == 10
    assert payload["gates"]["all_pass"]
