from __future__ import annotations

from scripts import logdx_agent_chain_source_audit as audit


def test_literal_regex_alternatives_drop_generic_terms() -> None:
    assert audit._literal_regex_alternatives(
        r"error|FAILED pandas|tests/foo_test.py::test_shape|Traceback"
    ) == [
        "FAILED pandas",
        "tests/foo_test.py::test_shape",
    ]


def test_dependencies_require_prior_observation_and_absence_from_initial() -> None:
    call = {
        "tool": "grep",
        "args": {"pattern": "tests/foo_test.py::test_shape"},
    }
    dependencies = audit.dependencies_for_call(
        call,
        initial_context="A short unrelated excerpt.",
        prior_observations=[
            "120: FAILED tests/foo_test.py::test_shape - AssertionError"
        ],
    )
    assert dependencies == [
        {
            "dependency_type": "file_or_path",
            "argument_field": "pattern",
        }
    ]

    assert audit.dependencies_for_call(
        call,
        initial_context="FAILED tests/foo_test.py::test_shape",
        prior_observations=[
            "120: FAILED tests/foo_test.py::test_shape - AssertionError"
        ],
    ) == []


def test_dependencies_exclude_literals_echoed_from_prior_tool_arguments() -> None:
    call = {
        "tool": "grep",
        "args": {"pattern": "tests/foo_test.py::test_shape"},
    }
    assert audit.dependencies_for_call(
        call,
        initial_context="A short unrelated excerpt.",
        prior_observations=[
            "grep pattern='tests/foo_test.py::test_shape'\n"
            "120: FAILED tests/foo_test.py::test_shape"
        ],
        prior_calls=[
            {
                "tool": "grep",
                "args": {"pattern": "tests/foo_test.py::test_shape"},
            }
        ],
    ) == []


def test_dependencies_exclude_tool_header_literals() -> None:
    assert audit.dependencies_for_call(
        {
            "tool": "grep",
            "args": {"pattern": "showing"},
        },
        initial_context="A short unrelated excerpt.",
        prior_observations=[
            "grep pattern='specific' matches=3 (showing 1 merged range)\n"
            "120: actual log content"
        ],
        prior_calls=[
            {
                "tool": "grep",
                "args": {"pattern": "specific"},
            }
        ],
    ) == []


def test_line_dependency_uses_exact_prior_line_prefix() -> None:
    call = {
        "tool": "view_log_lines",
        "args": {"center_line": 842, "radius": 20},
    }
    assert audit.dependencies_for_call(
        call,
        initial_context="No numbered context.",
        prior_observations=["840: before\n842: panic in worker\n843: after"],
    ) == [
        {
            "dependency_type": "line_number",
            "argument_field": "center_line",
        }
    ]
    assert audit.dependencies_for_call(
        call,
        initial_context="842: already visible",
        prior_observations=["842: panic in worker"],
    ) == []


def _row(
    case_id: str,
    *,
    calls: int,
    gain: float,
    dependency_type: str | None = None,
) -> dict:
    dependencies = []
    if dependency_type is not None:
        dependencies = [
            {
                "call_index": 1,
                "tool": "grep",
                "dependency_type": dependency_type,
                "argument_field": "pattern",
            }
        ]
    return {
        "split": "dev",
        "case_id": case_id,
        "context_method": "tail",
        "agent_score": 0.5 + gain,
        "single_shot_score": 0.5,
        "score_gain": gain,
        "tool_call_count": calls,
        "tool_names": ["tail"] * calls,
        "dependencies": dependencies,
    }


def test_aggregate_uses_distinct_case_units() -> None:
    rows = []
    for index in range(25):
        dependency_type = None
        gain = 0.01
        calls = 0
        if index < 12:
            calls = 1
        if index < 8:
            calls = 2
        if index < 6:
            dependency_type = (
                "line_number" if index % 2 == 0 else "test_or_symbol"
            )
            gain = 0.2 if index < 5 else 0.05
        rows.append(
            _row(
                f"case-{index:02d}",
                calls=calls,
                gain=gain,
                dependency_type=dependency_type,
            )
        )
        rows.append(
            _row(
                f"case-{index:02d}",
                calls=0,
                gain=0.01,
            )
        )

    result = audit.aggregate_rows(rows)

    assert result["metrics"]["matched_case_count"] == 25
    assert result["metrics"]["tool_case_count"] == 12
    assert result["metrics"]["multi_tool_case_count"] == 8
    assert result["metrics"]["dependency_case_count"] == 6
    assert result["metrics"]["dependency_improved_case_count"] == 5
    assert all(result["gates"].values())


def test_bound_source_and_manifests() -> None:
    assert audit.verify_source()["commit"] == audit.SOURCE_COMMIT
    case_paths, summary = audit.verify_manifests()
    assert summary["case_count"] == 35
    assert len(case_paths) == 35
    assert all(audit.verify_leakage_boundary().values())
