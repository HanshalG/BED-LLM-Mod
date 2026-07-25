from __future__ import annotations

import json

from scripts import multihop_rag_semantic_bed_manifest as manifest


def _rows() -> list[dict[str, object]]:
    rows = [
        {"question_type": "null_query", "evidence_list": []}
        for _ in range(manifest.EXPECTED_SOURCE_ROWS)
    ]
    for index in range(
        manifest.PUBLIC_PREVIEW_ROWS,
        manifest.PUBLIC_PREVIEW_ROWS + manifest.EXPECTED_ELIGIBLE_ROWS,
    ):
        rows[index] = {
            "question_type": (
                "inference_query" if index % 2 else "comparison_query"
            ),
            "evidence_list": [{}, {}],
        }
    return rows


def test_split_is_content_blind_and_disjoint(monkeypatch):
    rows = _rows()
    eligible = [
        str(index)
        for index in range(
            manifest.PUBLIC_PREVIEW_ROWS,
            manifest.PUBLIC_PREVIEW_ROWS + manifest.EXPECTED_ELIGIBLE_ROWS,
        )
    ]
    import random

    random.Random(manifest.SELECTION_SEED).shuffle(eligible)
    expected = {
        "mechanics": eligible[:5],
        "opportunity": eligible[5:405],
        "development": eligible[405:445],
        "holdout": eligible[445:],
    }
    monkeypatch.setattr(
        manifest,
        "SPLIT_HASHES",
        {
            name: manifest.ordered_hash(values)
            for name, values in expected.items()
        },
    )
    monkeypatch.setattr(
        manifest,
        "EXPECTED_MECHANICS_IDS",
        tuple(expected["mechanics"]),
    )
    actual = manifest.split_ids(rows)
    assert actual == expected
    assert len(set().union(*map(set, actual.values()))) == len(eligible)
    assert all(
        int(task_id) >= manifest.PUBLIC_PREVIEW_ROWS
        for values in actual.values()
        for task_id in values
    )


def test_manifest_emits_no_content(tmp_path, monkeypatch):
    query_path = tmp_path / "queries.json"
    corpus_path = tmp_path / "corpus.json"
    rows = _rows()
    corpus = [{} for _ in range(manifest.EXPECTED_CORPUS_ROWS)]
    query_path.write_text(json.dumps(rows), encoding="utf-8")
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    monkeypatch.setattr(
        manifest,
        "QUERY_SHA256",
        manifest.sha256_file(query_path),
    )
    monkeypatch.setattr(
        manifest,
        "CORPUS_SHA256",
        manifest.sha256_file(corpus_path),
    )
    eligible = [
        str(index)
        for index in range(
            manifest.PUBLIC_PREVIEW_ROWS,
            manifest.PUBLIC_PREVIEW_ROWS + manifest.EXPECTED_ELIGIBLE_ROWS,
        )
    ]
    import random

    random.Random(manifest.SELECTION_SEED).shuffle(eligible)
    expected = {
        "mechanics": eligible[:5],
        "opportunity": eligible[5:405],
        "development": eligible[405:445],
        "holdout": eligible[445:],
    }
    monkeypatch.setattr(
        manifest,
        "SPLIT_HASHES",
        {
            name: manifest.ordered_hash(values)
            for name, values in expected.items()
        },
    )
    monkeypatch.setattr(
        manifest,
        "EXPECTED_MECHANICS_IDS",
        tuple(expected["mechanics"]),
    )
    result = manifest.build_manifest(query_path, corpus_path)
    assert result["content_emitted"] is False
    assert result["answers_emitted"] is False
    assert result["document_ids_emitted"] is False

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(
                *(keys(child) for child in value.values())
            )
        if isinstance(value, list):
            return set().union(*(keys(child) for child in value))
        return set()

    assert not {
        "query",
        "answer",
        "evidence_list",
        "url",
        "title",
        "body",
        "fact",
    } & keys(result)
