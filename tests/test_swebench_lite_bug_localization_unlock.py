from __future__ import annotations

from scripts.swebench_lite_bug_localization_unlock import (
    code_tokens,
    has_direct_target_leak,
    root_queries,
    summarize,
    target_file_from_patch,
)


def test_target_file_parser_requires_single_file() -> None:
    patch = "--- a/pkg/old.py\n+++ b/pkg/new_name.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert target_file_from_patch(patch) == "pkg/new_name.py"


def test_direct_leak_checks_path_basename_and_stem() -> None:
    target = "src/project/parser_helpers.py"
    assert has_direct_target_leak("fix parser_helpers behavior", target)
    assert has_direct_target_leak("see src/project/parser_helpers.py", target)
    assert not has_direct_target_leak("fix malformed token handling", target)


def test_code_tokens_split_paths_snake_case_and_camel_case() -> None:
    assert code_tokens("src/ParserTools/bad_token.py") == [
        "src",
        "parser",
        "tools",
        "bad",
        "token",
        "py",
    ]


def test_root_queries_are_distinct_and_target_blind() -> None:
    problem = (
        "Parser crashes on malformed aliases\n\n"
        "Calling `parse_alias()` raises UnexpectedTokenError for nested input."
    )
    roots = root_queries(problem)
    assert len(roots) >= 3
    assert len({root.casefold() for root in roots}) == len(roots)


def test_summary_enforces_nonmyopic_opportunity_gates() -> None:
    records = []
    for index in range(9):
        records.append(
            {
                "num_roots": 4,
                "distinct_root_top1": 3,
                "best_immediate_value": int(index >= 3),
                "pair_gain": int(index < 3),
                "oracle_first_differs_from_greedy": index < 2,
                "nonmyopic_gap": int(index < 2),
            }
        )
    summary = summarize(records)
    assert summary["gates"]["all_pass"]
