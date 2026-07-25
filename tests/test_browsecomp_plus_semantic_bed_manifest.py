from __future__ import annotations

from scripts import browsecomp_plus_semantic_bed_manifest as manifest


def _fixture_counts() -> dict[str, int]:
    return {
        **{str(index): 4 for index in range(100, 160)},
        **{str(index): 6 for index in range(200, 280)},
        **{str(index): 9 for index in range(300, 370)},
    }


def test_split_is_deterministic_disjoint_and_stratified():
    counts = _fixture_counts()
    first = manifest.split_ids(counts, verify_frozen=False)
    second = manifest.split_ids(
        dict(reversed(list(counts.items()))),
        verify_frozen=False,
    )

    assert first == second
    flattened = [query_id for values in first.values() for query_id in values]
    assert len(flattened) == len(set(flattened)) == len(counts)
    assert {name: len(values) for name, values in first.items()} == {
        "mechanics": 5,
        "opportunity": 120,
        "development": 40,
        "holdout": 45,
    }
    assert {
        name: manifest.evidence_bin(counts[query_id])
        for name, query_id in zip(
            ("low", "mid", "high"),
            (
                first["mechanics"][0],
                first["mechanics"][1],
                first["mechanics"][3],
            ),
        )
    } == {"low": "low", "mid": "mid", "high": "high"}


def test_parse_qrels_rejects_duplicate_and_nonbinary_rows(tmp_path):
    path = tmp_path / "qrels.txt"
    path.write_text("1 Q0 d1 1\n1 Q0 d1 1\n", encoding="utf-8")
    try:
        manifest.parse_qrels(path)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate qrel was accepted")

    path.write_text("1 Q0 d1 2\n", encoding="utf-8")
    try:
        manifest.parse_qrels(path)
    except ValueError as exc:
        assert "invalid qrel fields" in str(exc)
    else:
        raise AssertionError("nonbinary qrel was accepted")


def test_ordered_hash_is_order_sensitive():
    assert manifest.ordered_hash(["1", "2"]) != manifest.ordered_hash(["2", "1"])
