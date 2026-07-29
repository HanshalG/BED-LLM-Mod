import json

import pytest

from scripts import number_game_glm_planner_depth_three as glm
from scripts import number_game_glm_planner_serving_smoke as smoke
from scripts import number_game_qwen_planner_depth_three as engine


def test_glm_smoke_contract_is_distinct() -> None:
    assert smoke.MODEL_ID == "z-ai/glm-5.1"
    assert smoke.MODEL_SEED == 46_000


def test_formal_seed_blocks_are_disjoint() -> None:
    validation = [
        seed
        for index in range(32)
        for seed in engine.validation_seeds_for_tree(
            index,
            start=glm.FORMAL_VALIDATION_SEED_START,
        )
    ]
    endpoints = [
        seed
        for index in range(32)
        for seed in engine.extra_endpoint_seeds_for_tree(
            index,
            start=glm.FORMAL_EXTRA_ENDPOINT_SEED_START,
        )
    ]
    seeds = [
        *glm.FORMAL_TREE_SEEDS,
        *glm.FORMAL_TARGET_SEEDS,
        *validation,
        *endpoints,
    ]

    assert len(seeds) == len(set(seeds))
    assert glm.FORMAL_EXPECTED_REQUESTS == 32 * engine.REQUESTS_PER_TREE


def test_validate_smoke_requires_frozen_glm_contract(tmp_path) -> None:
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

    assert glm.validate_smoke_result(path)["status"] == "passed"

    value = json.loads(path.read_text())
    value["gates"]["all_pass"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        glm.validate_smoke_result(path)


def test_balance_check_is_fail_closed() -> None:
    with pytest.raises(RuntimeError):
        glm.require_starting_balance(7.49)
    glm.require_starting_balance(7.50)
