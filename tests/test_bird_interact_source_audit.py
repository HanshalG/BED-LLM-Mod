from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "bird_interact_source_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "bird_interact_source_audit",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(
    instance_id: str,
    database: str,
    term: str,
    snippet: str,
    *,
    follow_up=None,
):
    return {
        "instance_id": instance_id,
        "selected_database": database,
        "amb_user_query": "ambiguous request",
        "sol_sql": [],
        "test_cases": [],
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {
                    "term": term,
                    "sql_snippet": snippet,
                    "is_mask": True,
                    "type": "semantic_ambiguity",
                }
            ]
        },
        "knowledge_ambiguity": [],
        "follow_up": follow_up,
    }


def test_summarize_rows_detects_competing_same_database_interpretations():
    rows = [
        _row("a", "db", "risk score", "risk > 0.5"),
        _row("b", "db", " Risk   Score ", "risk > 0.8"),
        _row("c", "other", "risk score", "risk > 0.2"),
    ]

    summary = MODULE.summarize_rows(rows)

    assert summary["rows"] == 3
    assert summary["distinct_databases"] == 2
    assert summary["empty_solution_sql_rows"] == 3
    alternatives = summary["released_alternative_interpretations"]
    assert alternatives["same_database_repeated_term_pairs"] == 1
    assert alternatives[
        "same_database_term_pairs_with_distinct_sql_snippets"
    ] == 1
    assert alternatives["task_term_pairs_with_distinct_sql_snippets"] == 0
    assert not alternatives["explicit_world_prior_field_present"]


def test_summarize_rows_counts_ambiguity_structure():
    row = _row("a", "db", "status", "status = 'active'")
    row["user_query_ambiguity"]["critical_ambiguity"].extend(
        [
            {
                "term": "recent",
                "sql_snippet": "date > cutoff",
                "is_mask": False,
                "type": "semantic_ambiguity",
            },
            {
                "term": "top",
                "sql_snippet": "limit 5",
                "is_mask": True,
                "type": "intent_ambiguity",
            },
        ]
    )
    row["knowledge_ambiguity"] = [{"term": "a"}, {"term": "b"}]

    summary = MODULE.summarize_rows([row])

    assert summary["critical_ambiguities"]["total"] == 3
    assert summary["critical_ambiguities"]["masked"] == 2
    assert summary["critical_ambiguities"]["tasks_with_at_least_three"] == 1
    assert summary["knowledge_ambiguities"]["tasks_with_at_least_two"] == 1
