from __future__ import annotations

from scripts.longvid_contrastive_line_mechanics import (
    EXCLUDED_PRIOR_ROWS,
    TASK_LAYOUT,
    TASK_LAYOUT_HASH,
    line_rank_messages,
    line_support_messages,
    parse_line_rank,
    parse_line_support,
)
from scripts.longvid_contrastive_path_belief_mechanics import (
    initial_messages,
    layout_hash_for,
)


def _support_text(anchor: str = "QUESTION") -> str:
    return "\n".join(
        (
            f"H{index}|{10 + index}|{anchor}|"
            f"{anchor} distinct query {index}|Distinct chain hypothesis {index}"
        )
        for index in range(1, 7)
    )


def test_line_layout_hash_and_exclusions_are_stable() -> None:
    assert layout_hash_for(TASK_LAYOUT) == TASK_LAYOUT_HASH
    assert not set(EXCLUDED_PRIOR_ROWS) & {
        row_index for row_index, _ in TASK_LAYOUT
    }


def test_line_support_accepts_unicode_anchor() -> None:
    support = parse_line_support(_support_text("€4bn"))
    assert len(support) == 6
    assert support[0]["anchor"] == "€4bn"


def test_line_rank_parser() -> None:
    assert parse_line_rank("B|79|missing causal bridge") == {
        "choice": "B",
        "confidence": 79,
        "unresolved_need": "missing causal bridge",
    }


def test_line_prompts_replace_object_instruction() -> None:
    support = line_support_messages(initial_messages("fixture?"))
    assert "exactly six lines" in support[0]["content"]
    rank = line_rank_messages(
        [
            {"role": "system", "content": "Return only the schema-conforming object."},
            {"role": "user", "content": "fixture"},
        ]
    )
    assert "choice|confidence|unresolved_need" in rank[0]["content"]
