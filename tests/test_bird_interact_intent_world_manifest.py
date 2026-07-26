from __future__ import annotations

from scripts import bird_interact_intent_world_manifest as manifest


def _row(index: int, database: str | None = None) -> dict:
    critical = [
        {
            "term": "risk score",
            "sql_snippet": "risk * severity",
            "is_mask": True,
            "type": "knowledge_linking_ambiguity",
        },
        {
            "term": "recent",
            "sql_snippet": "date > cutoff",
            "is_mask": True,
            "type": "semantic_ambiguity",
        },
        {
            "term": "top results",
            "sql_snippet": "limit 5",
            "is_mask": False,
            "type": "intent_ambiguity",
        },
    ]
    return {
        "instance_id": f"task_{index}",
        "selected_database": database or f"db_{index % 5}",
        "amb_user_query": "ambiguous request",
        "sol_sql": [],
        "test_cases": [],
        "follow_up": None,
        "user_query_ambiguity": {"critical_ambiguity": critical},
        "knowledge_ambiguity": [{"term": "risk"}],
    }


def test_eligibility_requires_mixed_hidden_ambiguities():
    row = _row(1)
    assert manifest.eligible_task(row)

    row["knowledge_ambiguity"] = []
    assert not manifest.eligible_task(row)
    row = _row(2)
    for item in row["user_query_ambiguity"]["critical_ambiguity"]:
        item["type"] = "knowledge_linking_ambiguity"
    assert not manifest.eligible_task(row)
    row = _row(3)
    row["user_query_ambiguity"]["critical_ambiguity"][1]["is_mask"] = False
    assert not manifest.eligible_task(row)


def test_split_is_reproducible_and_content_blind():
    rows = [_row(index) for index in range(100)]
    splits, eligible = manifest.split_tasks(rows)

    assert splits == manifest.split_tasks(rows)[0]
    assert len(eligible) == 100
    assert {name: len(values) for name, values in splits.items()} == {
        "mechanics": 3,
        "opportunity": 24,
        "development": 12,
        "holdout": 61,
    }


def test_exposed_task_ids_are_excluded():
    row = _row(1)
    row["instance_id"] = "alien_1"
    assert not manifest.eligible_task(row)
