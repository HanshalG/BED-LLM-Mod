from __future__ import annotations

import json

import pytest

from scripts import number_game_qwen_first_link_confirmation64 as confirm
from scripts import number_game_qwen_first_link_serving_smoke as smoke


def test_fresh_seed_and_budget_constants() -> None:
    assert len(confirm.TREE_SEEDS) == 64
    assert len(confirm.TARGET_SEEDS) == 64
    assert confirm.EXPECTED_REQUESTS == 3_712
    assert confirm.BOOTSTRAP_SAMPLES == 20_000
    assert smoke.MODEL_ID == "qwen/qwen3.7-plus"
    assert smoke.MODEL_SEED == 60_000


def test_smoke_validator_requires_exact_linked_interface(tmp_path) -> None:
    path = tmp_path / "RESULT.json"
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "protocol": {
                    "interface_version": smoke.INTERFACE_VERSION,
                    "model": smoke.MODEL_ID,
                    "expected_requests": 10,
                    "efficacy_used_for_authorization": False,
                },
                "gates": {"all_pass": True},
            }
        )
    )
    assert confirm.validate_smoke_result(path)["status"] == "passed"

    payload = json.loads(path.read_text())
    payload["protocol"]["model"] = "wrong/model"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="model changed"):
        confirm.validate_smoke_result(path)


def test_first_link_summary_uses_selected_root_pair(monkeypatch) -> None:
    monkeypatch.setattr(confirm, "BOOTSTRAP_SAMPLES", 100)
    trees = []
    for index in range(12):
        predicted = 0.01 + index * 0.001
        realized = 0.02 + index * 0.002
        trees.append(
            {
                "tree_seed": index,
                "selection": {
                    "crossfit_depth_three_root": 1,
                    "myopic_root": 2,
                    "crossfit_depth_three_brier": {
                        "1": 0.1,
                        "2": 0.1 + predicted,
                    },
                },
                "per_root_endpoint_brier": {
                    "1": 0.2,
                    "2": 0.2 + realized,
                },
            }
        )

    summary = confirm.summarize_first_link(trees)

    assert summary["root_differences"] == 12
    assert summary["wins"] == 12
    assert summary["losses"] == 0
    assert summary["score_to_realized_advantage_spearman"] == pytest.approx(
        1.0
    )
