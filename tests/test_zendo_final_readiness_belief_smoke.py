from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.zendo_final_readiness_belief_smoke import (
    EXPECTED_REQUESTS,
    POOL_SIZE,
    ROOT_COUNT,
    best_continuation,
    deterministic_scene_pool,
    deterministic_random_audit_bank,
    parse_particle_population,
    parse_scorer,
    run_gate,
    select_root_scenes,
)
from helpers import load_config
from scripts.zendo_path_dependent_belief_gate import (
    evaluate_rule,
    posterior_weights,
    raw_official_scene,
    validate_rule,
)


def _hypothesis(index: int, rule: dict[str, object]) -> dict[str, object]:
    return {
        "id": f"H{index:02d}",
        "rule_text": f"rule {index}",
        "rule": validate_rule(rule),
    }


def _cases() -> list[dict[str, object]]:
    return json.loads(
        Path(
            "external/doing-experiments-and-revising-rules/data/zendo_cases.json"
        ).read_text(encoding="utf-8")
    )


def _diverse_hypotheses() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 1
    for attribute, values in (
        ("color", ("blue", "red", "green")),
        ("size", ("small", "medium", "large")),
        ("orientation", ("upright", "left", "right")),
    ):
        for value in values:
            rows.append(
                _hypothesis(
                    index,
                    {
                        "op": "exists",
                        "predicate": {
                            "op": "attribute",
                            "attribute": attribute,
                            "value": value,
                        },
                    },
                )
            )
            index += 1
    for value in range(1, 4):
        rows.append(
            _hypothesis(
                index,
                {
                    "op": "count",
                    "predicate": {"op": "any"},
                    "comparison": "eq",
                    "value": value,
                },
            )
        )
        index += 1
    return rows


def test_deterministic_scene_pool_is_stable_and_distinct() -> None:
    initial = raw_official_scene(_cases()[1]["t"][0])
    first = deterministic_scene_pool(initial)
    second = deterministic_scene_pool(initial)
    assert first == second
    assert len(first) == POOL_SIZE
    assert len({json.dumps(scene, sort_keys=True) for scene in first}) == POOL_SIZE
    assert json.dumps(initial, sort_keys=True) not in {
        json.dumps(scene, sort_keys=True) for scene in first
    }


def test_root_selection_is_target_blind_informative_and_distinct() -> None:
    initial = raw_official_scene(_cases()[1]["t"][0])
    hypotheses = _diverse_hypotheses()
    weights = posterior_weights(hypotheses, [(initial, True)])
    pool = deterministic_scene_pool(initial)
    roots, rows = select_root_scenes(hypotheses, weights, pool)
    assert len(roots) == ROOT_COUNT
    assert len(rows) == ROOT_COUNT
    assert len(
        {tuple(row["prediction_signature"]) for row in rows}
    ) == ROOT_COUNT
    assert all(
        min(row["initial_probability_yes"], 1 - row["initial_probability_yes"])
        >= 0.10
        for row in rows
    )
    assert rows[0]["initial_eig"] == max(
        row["initial_eig"] for row in rows
    )


def test_best_continuation_maximizes_eig_and_excludes_root() -> None:
    initial = raw_official_scene(_cases()[1]["t"][0])
    hypotheses = _diverse_hypotheses()
    weights = posterior_weights(hypotheses, [(initial, True)])
    pool = deterministic_scene_pool(initial)
    roots, _ = select_root_scenes(hypotheses, weights, pool)
    pool_index, continuation, score = best_continuation(
        hypotheses, weights, pool, excluded_scene=roots[0]
    )
    assert pool[pool_index] == continuation
    assert continuation != roots[0]
    assert math.isfinite(score)
    assert score > 0
    signature = tuple(
        evaluate_rule(hypothesis["rule"], continuation)
        for hypothesis in hypotheses
    )
    assert len(signature) == len(hypotheses)


def test_scorer_parser_is_strict() -> None:
    assert parse_scorer('{"root_scores":[10,20,30,40]}') == [10, 20, 30, 40]
    with pytest.raises(ValueError):
        parse_scorer('{"root_scores":["10","20","30","40"]}')
    with pytest.raises(json.JSONDecodeError):
        parse_scorer('```json\n{"root_scores":[10,20,30,40]}\n```')
    with pytest.raises(ValueError):
        parse_scorer('{"root_scores":[10,20,30,40],"reason":"extra"}')


def test_particle_multiset_parser_preserves_duplicate_multiplicity() -> None:
    hypotheses = _diverse_hypotheses()
    hypotheses[-1]["rule"] = hypotheses[0]["rule"]
    response = json.dumps({"hypotheses": hypotheses})
    particles = parse_particle_population(
        response, allow_duplicate_asts=True
    )
    assert len(particles) == 12
    assert particles[-1]["rule"] == particles[0]["rule"]
    with pytest.raises(ValueError, match="unique"):
        parse_particle_population(response, allow_duplicate_asts=False)


def test_protocol_uses_exactly_ten_requests() -> None:
    assert EXPECTED_REQUESTS == 1 + 2 * ROOT_COUNT + 1


def test_random_only_audit_bank_stays_within_executable_dsl() -> None:
    bank = deterministic_random_audit_bank(
        task_index=6, selection_seed=24370
    )
    assert len(bank) == 512
    assert max(len(scene["blocks"]) for scene in bank) <= 6


class _FakeAdapter:
    def __init__(self, hypothesis_response: str) -> None:
        self.hypothesis_response = hypothesis_response
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        messages: list[list[dict[str, str]]],
        **_: object,
    ) -> list[str]:
        self.requests += len(messages)
        if len(messages) == 8:
            return [self.hypothesis_response] * 8
        if self.requests == 1:
            return [self.hypothesis_response]
        return ['{"root_scores":[10,20,30,40]}']

    def usage_snapshot(self) -> dict[str, object]:
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.01,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
        }


def test_full_gate_freezes_ten_calls_before_external_analysis(
    tmp_path: Path,
) -> None:
    response = json.dumps({"hypotheses": _diverse_hypotheses()})
    adapter = _FakeAdapter(response)
    config = load_config(
        "configs/config_zendo_path_dependent_belief_openrouter.yaml"
    )
    payload = run_gate(
        config,
        source_dir=Path(
            "external/doing-experiments-and-revising-rules"
        ),
        raw_checkpoint_path=tmp_path / "raw.json",
        model_adapter=adapter,
    )
    assert adapter.requests == EXPECTED_REQUESTS
    assert payload["protocol"][
        "truth_hidden_until_all_branches_and_scores_frozen"
    ]
    assert len(payload["root_rows"]) == ROOT_COUNT
    assert payload["usage"]["physical_requests"] == EXPECTED_REQUESTS


def test_full_gate_accepts_particle_multiplicity_on_fresh_task(
    tmp_path: Path,
) -> None:
    hypotheses = _diverse_hypotheses()
    hypotheses[-1]["rule"] = hypotheses[0]["rule"]
    adapter = _FakeAdapter(json.dumps({"hypotheses": hypotheses}))
    config = load_config(
        "configs/config_zendo_path_dependent_belief_openrouter.yaml"
    )
    payload = run_gate(
        config,
        source_dir=Path(
            "external/doing-experiments-and-revising-rules"
        ),
        raw_checkpoint_path=tmp_path / "multiset-raw.json",
        model_adapter=adapter,
        task_name="mu",
        interface_version="test-multiset",
        selection_seed=24370,
        allow_duplicate_particles=True,
        audit_mode="random_only",
    )
    assert adapter.requests == EXPECTED_REQUESTS
    assert payload["protocol"]["particle_semantics"] == "multiset"
    assert payload["protocol"]["task_name"] == "mu"
    assert payload["summary"]["population_unique_ast_counts"]["initial"] == 11
