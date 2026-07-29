from __future__ import annotations

from scripts.icae_bench_source_opportunity_audit import (
    content_tokens,
    lexical_unlock_edges,
    substantive_constraints,
    trigger_idf,
)


def _record() -> dict:
    return {
        "fuzzy_prd": "Build a compact event processor.",
        "oracle_data": {
            "hidden_constraints": [
                {
                    "constraint_id": "C001",
                    "trigger_keywords": ["which event formats are supported"],
                    "oracle_response": (
                        "Support the CloudEvent envelope and preserve its "
                        "extension attributes."
                    ),
                },
                {
                    "constraint_id": "C002",
                    "trigger_keywords": [
                        "how are cloudevent extension attributes validated"
                    ],
                    "oracle_response": "Reject extension keys containing spaces.",
                },
                {
                    "constraint_id": "ARCH001",
                    "trigger_keywords": ["how should code be organized"],
                    "oracle_response": "Keep adapters separate.",
                },
            ]
        },
    }


def test_content_tokens_remove_common_words() -> None:
    assert content_tokens("What is the CloudEvent envelope format?") == {
        "cloudevent",
        "envelope",
        "format",
    }


def test_substantive_constraints_exclude_boilerplate_entries() -> None:
    rows = substantive_constraints(_record()["oracle_data"]["hidden_constraints"])
    assert [identifier for identifier, _ in rows] == ["C001", "C002"]


def test_lexical_unlock_requires_new_answer_introduced_terms() -> None:
    record = _record()
    idf = trigger_idf([record])
    edges = lexical_unlock_edges(
        record,
        idf=idf,
        minimum_idf_score=2.0,
    )
    assert edges == [("C001", "C002")]


def test_lexical_unlock_does_not_credit_terms_already_in_fuzzy_prd() -> None:
    record = _record()
    record["fuzzy_prd"] += " Use CloudEvent extension attributes."
    idf = trigger_idf([record])
    assert lexical_unlock_edges(
        record,
        idf=idf,
        minimum_idf_score=2.0,
    ) == []
