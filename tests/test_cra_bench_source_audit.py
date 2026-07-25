from __future__ import annotations

from scripts import cra_bench_source_audit as audit


def _row(base_user: int, split: str, target: str) -> dict[str, object]:
    return {
        "base_user_index": base_user,
        "domain": "domain",
        "user_profile": {"user": base_user},
        "recsys_profile": {"visible": base_user},
        "task": {
            "initial_query": f"query {split}",
            "behavior_profile": {"patience_budget": 2},
        },
        "fuzzy_gt": {
            "evaluation_only": {
                "target_asin": target,
            }
        },
    }


def test_analyze_rows_counts_repeated_difficulty_worlds():
    splits = {
        split: [
            _row(0, split, "target-0"),
            _row(1, split, "target-1"),
        ]
        for split in ("easy", "medium", "hard")
    }
    result = audit.analyze_rows(splits)
    assert result["split_rows"] == {
        "easy": 2,
        "medium": 2,
        "hard": 2,
    }
    assert result["base_users"] == 2
    assert result["unique_targets"] == 2
    assert result["hard_domain_counts"] == {"domain": 2}
    assert result["hard_patience_budget_counts"] == {"2": 2}
    assert result["cross_difficulty"] == {
        "base_users_with_three_variants": 2,
        "same_user_profile": 2,
        "same_recommender_profile": 2,
        "same_target": 2,
    }


def test_audit_contract_emits_no_target_content(tmp_path, monkeypatch):
    root = tmp_path / "cra"
    (root / "data").mkdir(parents=True)
    rows = {
        split: [_row(0, split, "private-target")]
        for split in ("easy", "medium", "hard")
    }
    hashes = {}
    for split, split_rows in rows.items():
        path = root / "data" / f"{split}.jsonl"
        path.write_text(
            "\n".join(
                __import__("json").dumps(row) for row in split_rows
            )
            + "\n",
            encoding="utf-8",
        )
        hashes[split] = audit.sha256_file(path)
    monkeypatch.setattr(audit, "SPLIT_SHA256", hashes)
    monkeypatch.setattr(audit, "EXPECTED_SPLIT_ROWS", 1)
    monkeypatch.setattr(audit, "EXPECTED_UNIQUE_TARGETS", 1)
    monkeypatch.setattr(audit, "EXPECTED_DOMAINS", {"domain": 1})
    monkeypatch.setattr(
        audit,
        "EXPECTED_RELEASE_FILES",
        {f"data/{split}.jsonl" for split in rows},
    )
    result = audit.build_audit(root)
    serialized = __import__("json").dumps(result)
    assert "private-target" not in serialized
    assert result["task_content_emitted"] is False
    assert result["target_ids_emitted"] is False
