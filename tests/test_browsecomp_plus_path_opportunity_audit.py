from __future__ import annotations

from scripts.browsecomp_plus_path_opportunity_audit import analyze_task


def test_bridge_requires_observation_enabled_tokens_and_new_evidence():
    result = analyze_task(
        question="Which musician released the hidden album?",
        events=[
            (
                "hidden album musician",
                [
                    {
                        "docid": "d0",
                        "snippet": "A profile mentions Avery North and Moon Archive.",
                    }
                ],
            ),
            (
                "Avery North Moon Archive discography",
                [
                    {
                        "docid": "e1",
                        "snippet": "The evidence page.",
                    }
                ],
            ),
        ],
        evidence_doc_ids={"e1"},
        gold_doc_ids={"e1"},
    )

    assert result["has_bridge_evidence"]
    assert result["has_bridge_gold"]
    assert result["has_bridge_after_miss"]
    assert result["first_evidence_search_index"] == 2
    assert not result["has_delayed_evidence"]


def test_question_terms_alone_do_not_count_as_bridge():
    result = analyze_task(
        question="Which Avery North album is hidden?",
        events=[
            (
                "unrelated query",
                [{"docid": "d0", "snippet": "Avery North profile."}],
            ),
            (
                "Avery North album",
                [{"docid": "e1", "snippet": "Evidence."}],
            ),
        ],
        evidence_doc_ids={"e1"},
        gold_doc_ids=set(),
    )

    assert not result["has_bridge_evidence"]
    assert result["evidence_seen_count"] == 1


def test_retrieving_seen_evidence_does_not_create_second_hit():
    result = analyze_task(
        question="Which record?",
        events=[
            (
                "initial clue",
                [{"docid": "d0", "snippet": "Avery North Moon Archive."}],
            ),
            (
                "Avery North Moon Archive",
                [{"docid": "e1", "snippet": "Evidence one."}],
            ),
            (
                "Avery North Moon Archive details",
                [{"docid": "e1", "snippet": "Evidence one repeated."}],
            ),
        ],
        evidence_doc_ids={"e1"},
        gold_doc_ids=set(),
    )

    assert result["bridge_evidence_hit_count"] == 1
