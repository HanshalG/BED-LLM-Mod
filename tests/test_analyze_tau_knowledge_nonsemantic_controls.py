from scripts.analyze_tau_knowledge_nonsemantic_controls import (
    _argmax,
    _tokens,
    _unique_documents,
)


def test_argmax_uses_original_order_for_ties():
    assert _argmax([3.0, 5.0, 5.0, 1.0]) == 1


def test_tokens_remove_short_terms_and_frozen_stopwords():
    assert _tokens("The ATM fee is due in 2 days") == {
        "atm",
        "fee",
        "due",
        "days",
    }


def test_unique_documents_keep_max_bm25_score():
    first = [{"id": "a", "bm25_score": 2.0}]
    followup = [
        {"id": "a", "bm25_score": 3.0},
        {"id": "b", "bm25_score": 1.0},
    ]
    documents = _unique_documents(first, followup)
    assert [(row["id"], row["bm25_score"]) for row in documents] == [
        ("a", 3.0),
        ("b", 1.0),
    ]
