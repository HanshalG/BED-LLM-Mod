from __future__ import annotations

from scripts import extract_browsecomp_plus_open_mechanics as extract


def test_decrypt_value_preserves_query_id_only(monkeypatch):
    monkeypatch.setattr(
        extract,
        "decrypt_string",
        lambda value: f"open:{value}",
    )
    encrypted = {
        "query_id": "42",
        "query": "q",
        "gold_docs": [
            {"docid": "d", "text": "t", "url": "u"},
        ],
    }

    assert extract.decrypt_value(encrypted) == {
        "query_id": "42",
        "query": "open:q",
        "gold_docs": [
            {
                "docid": "open:d",
                "text": "open:t",
                "url": "open:u",
            }
        ],
    }


def test_key_derivation_has_requested_length():
    assert len(extract.derive_key("secret", 0)) == 0
    assert len(extract.derive_key("secret", 97)) == 97
