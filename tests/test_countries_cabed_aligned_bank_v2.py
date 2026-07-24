import json
import math

import pytest

from scripts.countries_cabed_aligned_bank_v2 import (
    BANK_PROPOSALS,
    BANK_WIDTH,
    COUNTRIES,
    FORMAL_STYLES,
    SMOKE_STYLES,
    _build_tree,
    _evaluate_tree,
    bank_generation_messages,
    parse_bank_questions,
    summarize,
)


def _binary_table(bit: int) -> tuple[str, ...]:
    return tuple(
        "Yes" if ((index >> bit) & 1) else "No"
        for index in range(len(COUNTRIES))
    )


def _test_tree() -> dict:
    questions = tuple(f"Does property {index} apply?" for index in range(BANK_WIDTH))
    tables = tuple(_binary_table(index % 6) for index in range(BANK_WIDTH))
    return _build_tree(
        tree_index=0,
        style="test",
        questions=questions,
        tables=tables,
    )


def test_protocol_sizes_and_styles_are_frozen():
    assert BANK_PROPOSALS == 40
    assert BANK_WIDTH == 32
    assert len(COUNTRIES) == 64
    assert len(SMOKE_STYLES) == 2
    assert len(FORMAL_STYLES) == 12


def test_bank_parser_requires_exact_unique_nondirect_questions():
    questions = [
        f"Does property {index} apply?" for index in range(BANK_PROPOSALS)
    ]
    assert parse_bank_questions(
        json.dumps({"questions": questions}),
        count=BANK_PROPOSALS,
    ) == tuple(questions)
    questions[-1] = questions[0]
    with pytest.raises(ValueError, match="duplicate"):
        parse_bank_questions(
            json.dumps({"questions": questions}),
            count=BANK_PROPOSALS,
        )
    questions[-1] = "Is it Canada?"
    with pytest.raises(ValueError, match="direct"):
        parse_bank_questions(
            json.dumps({"questions": questions}),
            count=BANK_PROPOSALS,
        )


def test_generation_prompt_is_target_blind_and_requests_balanced_bank():
    prompt = bank_generation_messages(
        count=BANK_PROPOSALS,
        style=SMOKE_STYLES[0],
    )[-1]["content"].lower()
    assert "at least four" in prompt
    assert "do not calculate information gain" in prompt
    assert "hidden target" not in prompt
    assert "exactly 40" in prompt


def test_build_tree_uses_every_other_bank_row_as_each_followup_menu():
    tree = _test_tree()
    assert len(tree["bank"]) == BANK_WIDTH
    assert len(tree["roots"]) == BANK_WIDTH
    for root_index, root in enumerate(tree["roots"]):
        for branch in root["branches"]:
            assert len(branch["candidate_bank_indices"]) == BANK_WIDTH - 1
            assert root_index not in branch["candidate_bank_indices"]
            assert branch["selected_bank_index"] != root_index


def test_build_tree_rejects_too_few_balanced_rows():
    questions = tuple(
        f"Does property {index} apply?" for index in range(BANK_WIDTH)
    )
    tables = tuple(
        _binary_table(index % 6)
        if index < BANK_WIDTH - 1
        else tuple("No" for _ in COUNTRIES)
        for index in range(BANK_WIDTH)
    )
    with pytest.raises(ValueError, match="fewer than 32"):
        _build_tree(
            tree_index=0,
            style="test",
            questions=questions,
            tables=tables,
        )


def test_exact_depth_two_selection_cannot_underperform_matched_depth_one():
    record = _evaluate_tree(_test_tree())
    assert record["mean_entropy_gain_depth_two_vs_one"] >= -1.0e-12
    assert record["target_loss_count_vs_one"] >= 0
    assert all(
        target["depth_two"]["truth_nll"]
        == target["depth_two"]["final_entropy"]
        for target in record["targets"]
    )


def test_known_adaptive_bank_produces_strict_nonmyopic_root():
    # Root 0 is the strongest one-step split. Root 1 is weaker immediately,
    # but rows 2 and 3 specialize to its two branches.
    questions = tuple(
        f"Does synthetic property {index} apply?"
        for index in range(BANK_WIDTH)
    )
    balanced_yes = {
        4, 5, 9, 15, 16, 17, 21, 23, 31, 32, 34, 35, 36, 37, 39, 41,
        44, 45, 46, 49, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    }
    balanced = tuple(
        "Yes" if index in balanced_yes else "No" for index in range(64)
    )
    setup = tuple("Yes" if index < 24 else "No" for index in range(64))
    yes_branch_specialist = tuple(
        "Yes" if index < 12 else "No" for index in range(64)
    )
    no_branch_specialist = tuple(
        "Yes" if 24 <= index < 44 else "No" for index in range(64)
    )
    tables = [
        balanced,
        setup,
        yes_branch_specialist,
        no_branch_specialist,
    ]
    tables.extend([balanced] * (BANK_WIDTH - len(tables)))
    tree = _build_tree(
        tree_index=0,
        style="test",
        questions=questions,
        tables=tables,
    )
    record = _evaluate_tree(tree)
    assert record["selections"]["depth_one"] == 0
    assert record["selections"]["depth_two"] == 1
    assert record["mean_entropy_gain_depth_two_vs_one"] > 0.0


def test_smoke_summary_requires_exact_82_nonreasoning_requests():
    records = [_evaluate_tree(_test_tree()), _evaluate_tree(_test_tree())]
    usage = {"physical_requests": 82, "reasoning_tokens": 0}
    summary = summarize(records, usage, stage="serving_smoke")
    assert summary["gates"]["exact_physical_request_count"]
    assert summary["gates"]["zero_reasoning_tokens"]
    assert summary["num_targets_per_tree"] == 64
    assert math.isfinite(summary["mean_entropy_gain_depth_two_vs_one"])
