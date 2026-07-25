from scripts.hil_bench_progressive_opportunity import (
    DEVELOPMENT_IDS,
    HOLDOUT_IDS,
    OPPORTUNITY_IDS,
    aggregate,
    analyze_task,
    extract_problem,
    frozen_split,
)


def test_frozen_split_is_disjoint_and_reproducible():
    opportunity, development, holdout = frozen_split()
    assert opportunity == OPPORTUNITY_IDS
    assert development == DEVELOPMENT_IDS
    assert holdout == HOLDOUT_IDS
    assert len(set(opportunity + development + holdout)) == 89
    assert not set(opportunity) & set(development)
    assert not set(opportunity) & set(holdout)


def test_extract_problem_excludes_tool_instructions():
    instruction = """
Answer the following question:
Which customers qualify for the special tier?

# Available Commands
- ask_human
"""
    assert extract_problem(instruction) == (
        "Which customers qualify for the special tier?"
    )


def test_task_analysis_counts_novel_business_evidence():
    blockers = [
        {
            "type": "question",
            "description": "The special tier cutoff is unclear.",
        },
        {
            "type": "business info",
            "description": (
                "The business definition mentions platinum classification but "
                "does not identify the database attribute."
            ),
        },
        {
            "type": "schema",
            "description": "The relevant status column is unclear.",
        },
    ]
    record = analyze_task(
        "sql_fixture",
        "Which customers qualify for the special tier?",
        blockers,
        [
            "Platinum classification is maintained by the loyalty team.",
            "An unrelated reporting convention.",
        ],
    )
    assert record["source_counts"] == {
        "business info": 1,
        "question": 1,
        "schema": 1,
    }
    assert record["business_evidence"][0]["novel_token_count"] >= 1


def test_aggregate_passes_a_synthetic_gate_sized_cohort():
    records = []
    for index in range(40):
        records.append(
            {
                "task_id": f"sql_{index}",
                "blocker_count": 3,
                "source_counts": {
                    "business info": 1,
                    "question": 1,
                    "schema": 1,
                },
                "business_info_document_count": 25,
                "business_evidence": [
                    {
                        "novel_token_count": 1,
                        "novel_recall": 0.1,
                        "best_total_recall": 0.2,
                    }
                ],
            }
        )
    result = aggregate(records)
    assert result["gate_passed"] is True
    assert all(result["gate_checks"].values())
