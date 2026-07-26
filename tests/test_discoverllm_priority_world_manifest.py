from __future__ import annotations

import copy

from scripts import discoverllm_priority_world_manifest as manifest


def _node(
    node_id: str,
    *,
    aware: int = 0,
    children: list[dict] | None = None,
) -> dict:
    return {
        "id": node_id,
        "text": f"criterion {node_id}",
        "aware": aware,
        "children": children or [],
    }


def _deep_root(node_id: str) -> dict:
    return _node(
        node_id,
        children=[_node(f"{node_id}.1", children=[_node(f"{node_id}.1.1")])],
    )


def _rows(count: int = 60) -> list[dict]:
    rows = []
    for index in range(count):
        artifact_id = f"artifact_{1000 + index}"
        state = [{"hierarchy": [_deep_root(str(root)) for root in range(1, 6)]}]
        history = [state]
        rows.extend(
            [
                {
                    "artifact_id": artifact_id,
                    "assistant_index": 0,
                    "prompt": [{"role": "user", "content": "request"}],
                    "completion": "A complete draft.",
                    "criteria_history": copy.deepcopy(history),
                },
                {
                    "artifact_id": artifact_id,
                    "assistant_index": 1,
                    "prompt": [{"role": "user", "content": "request"}],
                    "completion": "Which direction should I take?",
                    "criteria_history": copy.deepcopy(history),
                },
            ]
        )
    return rows


def test_world_root_filter_requires_hidden_deep_subtree():
    state = [
        {
            "hierarchy": [
                _deep_root("eligible"),
                _node("aware", aware=1, children=[_node("aware.1")]),
                _node("shallow"),
                _node("small", children=[_node("small.1")]),
            ]
        }
    ]
    assert [
        root["id"] for root in manifest.eligible_world_roots(state)
    ] == ["eligible"]


def test_split_and_world_selection_are_reproducible_and_disjoint():
    eligible = manifest.eligible_artifacts(_rows())
    first = manifest.split_artifacts(eligible)
    second = manifest.split_artifacts(eligible)
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "mechanics": 3,
        "opportunity": 30,
        "development": 20,
        "holdout": 7,
    }
    all_ids = [artifact_id for values in first.values() for artifact_id in values]
    assert len(all_ids) == len(set(all_ids))
    assert all(
        len(details["world_ids"]) == manifest.NUM_WORLDS
        for details in eligible.values()
    )


def test_manifest_emits_no_semantic_content_or_source_scores(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "creative.parquet"
    source.write_bytes(b"fixture")
    rows = _rows(120)
    monkeypatch.setattr(
        manifest,
        "CREATIVE_WRITING_SHA256",
        manifest.sha256_file(source),
    )
    monkeypatch.setattr(manifest, "_load_turn_one_rows", lambda _: rows)
    result = manifest.build_manifest(source, enforce_frozen=False)

    assert result["gates"]["at_least_100_eligible_artifacts"] is True
    assert result["prompt_content_emitted"] is False
    assert result["criterion_content_emitted"] is False
    assert result["completion_content_emitted"] is False
    assert result["source_metadata_emitted"] is False
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
