from __future__ import annotations

import copy

from scripts import discoverllm_priority_world_manifest as v1
from scripts import discoverllm_priority_world_manifest_v2 as manifest


def _root(node_id: str) -> dict:
    return {
        "id": node_id,
        "text": f"criterion {node_id}",
        "aware": 0,
        "children": [
            {
                "id": f"{node_id}.1",
                "text": "child",
                "aware": 0,
                "children": [
                    {
                        "id": f"{node_id}.1.1",
                        "text": "leaf",
                        "aware": 0,
                        "children": [],
                    }
                ],
            }
        ],
    }


def _domain_rows(domain_index: int, count: int) -> list[dict]:
    rows = []
    for index in range(count):
        artifact_id = f"artifact_{domain_index * 1000 + index + 500}"
        state = [{"hierarchy": [_root(str(root)) for root in range(1, 6)]}]
        history = [state]
        rows.extend(
            [
                {
                    "artifact_id": artifact_id,
                    "assistant_index": 0,
                    "prompt": [],
                    "completion": "Draft.",
                    "criteria_history": copy.deepcopy(history),
                },
                {
                    "artifact_id": artifact_id,
                    "assistant_index": 1,
                    "prompt": [],
                    "completion": "Which direction?",
                    "criteria_history": copy.deepcopy(history),
                },
            ]
        )
    return rows


def test_all_domain_selection_is_reproducible_and_target_blind():
    rows = {
        "creative_writing": _domain_rows(0, 100),
        "technical_writing": _domain_rows(1, 100),
        "svg_drawing": _domain_rows(2, 100),
    }
    eligible = manifest.eligible_artifacts(rows)
    assert len(eligible) == 300
    splits = manifest.split_artifacts(eligible)
    assert {name: len(values) for name, values in splits.items()} == {
        "mechanics": 3,
        "opportunity": 60,
        "development": 30,
        "holdout": 207,
    }
    assert all(
        len(details["world_ids"]) == v1.NUM_WORLDS
        for details in eligible.values()
    )
    assert splits == manifest.split_artifacts(eligible)


def test_manifest_does_not_emit_world_ids_or_semantic_content(
    tmp_path,
    monkeypatch,
):
    paths = {}
    rows_by_domain = {}
    for index, domain in enumerate(manifest.SOURCE_SHA256):
        path = tmp_path / f"{domain}.parquet"
        path.write_bytes(domain.encode("ascii"))
        paths[domain] = path
        rows_by_domain[domain] = _domain_rows(index, 100)
    monkeypatch.setattr(
        manifest,
        "SOURCE_SHA256",
        {domain: v1.sha256_file(path) for domain, path in paths.items()},
    )
    monkeypatch.setattr(
        manifest,
        "_load_earliest_turn_rows",
        lambda path: rows_by_domain[path.stem],
    )
    result = manifest.build_manifest(paths, enforce_frozen=False)
    assert result["gates"]["at_least_250_eligible_artifacts"] is True
    assert result["selected_world_ids_emitted"] is False
    assert result["released_scores_read"] is False
    assert result["released_winner_labels_read"] is False

    forbidden = {
        "prompt",
        "criterion",
        "completion",
        "score",
        "winner",
        "source_metadata",
        "world_ids",
    }

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(child) for child in value))
        return set()

    assert not forbidden & keys(result)
