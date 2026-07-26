from __future__ import annotations

import hashlib
import json

import pytest

from scripts import infoquest_llm_bed_manifest as manifest


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_rows():
    seeds = []
    settings = []
    traits = []
    for record_id in range(manifest.EXPECTED_RECORDS):
        personas = [
            {"id": record_id * 3 + index, "persona": f"persona {record_id} {index}"}
            for index in range(1, 4)
        ]
        seed_message = f"ambiguous request {record_id}"
        seeds.append(
            {
                "id": record_id,
                "persona1": personas[0],
                "persona2": personas[1],
                "persona3": personas[2],
                "prompt": f"seed prompt {record_id}",
                "prompt_tokens": 10,
                "seed_message": seed_message,
                "seed_message_tokens": 3,
            }
        )
        setting_rows = []
        for index in (1, 2):
            setting_rows.append(
                {
                    "description": f"description {record_id} {index}",
                    "goal": f"goal {record_id} {index}",
                    "obstacle": f"obstacle {record_id} {index}",
                    "constraints": [
                        f"constraint {record_id} {index} {item}"
                        for item in range(5)
                    ],
                    "solution": f"solution {record_id} {index}",
                    "checklist": [
                        f"checklist {record_id} {index} {item}"
                        for item in range(5)
                    ],
                    "persona": personas[index - 1]["persona"],
                    "setting_tokens": 40,
                    "prompt_tokens": 50,
                }
            )
        settings.append(
            {
                "id": record_id,
                "setting1": setting_rows[0],
                "setting2": setting_rows[1],
                "seed_message": seed_message,
            }
        )
        traits.append(
            {
                "id": record_id,
                "traits1": f"traits {record_id} 1",
                "traits1_tokens": 4,
                "traits2": f"traits {record_id} 2",
                "traits2_tokens": 4,
            }
        )
    return {
        "seed_messages": seeds,
        "settings": settings,
        "traits": traits,
    }


def _write_source(tmp_path, monkeypatch):
    rows_by_name = _source_rows()
    digests = {}
    for name, filename in manifest.SOURCE_FILES.items():
        path = tmp_path / filename
        _write_jsonl(path, rows_by_name[name])
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(manifest, "SOURCE_SHA256", digests)
    monkeypatch.setattr(
        manifest,
        "_git_revision",
        lambda source_root: manifest.SOURCE_REVISION,
    )
    return rows_by_name


def test_frozen_splits_are_reproducible_and_content_blind():
    splits = manifest.split_record_ids()
    assert {name: len(ids) for name, ids in splits.items()} == {
        "mechanics": 2,
        "opportunity": 80,
        "development": 30,
        "holdout": 388,
    }
    assert splits["mechanics"] == [0, 1]
    assert {
        name: manifest.canonical_sha256(ids) for name, ids in splits.items()
    } == manifest.EXPECTED_SPLIT_HASHES
    assert (
        manifest.canonical_sha256(splits)
        == manifest.EXPECTED_COMBINED_SPLITS_HASH
    )
    flattened = [record_id for ids in splits.values() for record_id in ids]
    assert len(flattened) == len(set(flattened)) == manifest.EXPECTED_RECORDS


def test_manifest_validates_sources_without_emitting_semantic_content(
    tmp_path,
    monkeypatch,
):
    rows_by_name = _write_source(tmp_path, monkeypatch)
    result = manifest.build_manifest(tmp_path)
    assert all(result["gates"].values())
    assert result["content_emitted"] is False
    assert len(result["source"]["record_sha256_by_id"]) == 500

    serialized = json.dumps(result, sort_keys=True)
    forbidden_values = {
        rows_by_name["seed_messages"][0]["seed_message"],
        rows_by_name["seed_messages"][0]["persona1"]["persona"],
        rows_by_name["settings"][0]["setting1"]["goal"],
        rows_by_name["settings"][0]["setting1"]["checklist"][0],
        rows_by_name["traits"][0]["traits1"],
    }
    assert not any(value in serialized for value in forbidden_values)
    assert result["openrouter_calls"] == 0
    assert result["oatml_jobs"] == 0


def test_manifest_rejects_cross_file_mismatch(tmp_path, monkeypatch):
    rows_by_name = _write_source(tmp_path, monkeypatch)
    rows_by_name["settings"][10]["seed_message"] = "different"
    settings_path = tmp_path / manifest.SOURCE_FILES["settings"]
    _write_jsonl(settings_path, rows_by_name["settings"])
    monkeypatch.setitem(
        manifest.SOURCE_SHA256,
        "settings",
        hashlib.sha256(settings_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="seed message does not match"):
        manifest.build_manifest(tmp_path)


def test_manifest_rejects_non_five_item_checklist(tmp_path, monkeypatch):
    rows_by_name = _write_source(tmp_path, monkeypatch)
    rows_by_name["settings"][10]["setting2"]["checklist"].pop()
    settings_path = tmp_path / manifest.SOURCE_FILES["settings"]
    _write_jsonl(settings_path, rows_by_name["settings"])
    monkeypatch.setitem(
        manifest.SOURCE_SHA256,
        "settings",
        hashlib.sha256(settings_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="exactly five items"):
        manifest.build_manifest(tmp_path)
