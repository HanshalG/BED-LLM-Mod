from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scripts import number_game_qwen_fully_fresh_source_control32 as run


def _source_result(
    *,
    mechanics: bool = True,
    opportunity: int = 21,
    myopic_science: bool = True,
    dynamic_science: bool = True,
) -> dict:
    return {
        "status": (
            "passed"
            if mechanics and myopic_science and dynamic_science
            else "gated_null"
        ),
        "usage": {
            "adapter_requests": run.SOURCE_EXPECTED_REQUESTS,
            "run_cost_usd": 4.7,
        },
        "mechanics_gates": {"mechanics": mechanics},
        "myopic_policy_gates": {"myopic": myopic_science},
        "dynamic_support": {
            "root_differences": opportunity,
            "gates": {"dynamic": dynamic_science},
            "comparison": {},
        },
    }


def _control_result(
    *,
    mechanics: bool = True,
    science: bool = True,
) -> dict:
    return {
        "status": "passed" if mechanics and science else "gated_null",
        "usage": {
            "adapter_requests": run.CONTROL_EXPECTED_REQUESTS,
            "run_cost_usd": 3.2,
        },
        "mechanics_gates": {"mechanics": mechanics},
        "analysis": {
            "scientific_gates": {"first_link": science},
        },
        "second_draw_novelty": {
            "used_as_gate": False,
        },
    }


def _write_source_artifacts(output_dir, result) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "RESULT.json").write_text(json.dumps(result))
    (output_dir / "TREES.json").write_text(json.dumps({"trees": []}))
    (output_dir / "TARGETS.json").write_text(
        json.dumps({"targets": []})
    )


def _write_control_artifacts(output_dir, result) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "RESULT.json").write_text(json.dumps(result))
    (output_dir / "CONTROLS.json").write_text(
        json.dumps({"trees": []})
    )


def _install_fake_stages(
    monkeypatch,
    *,
    source_result,
    control_result,
):
    calls = {"source": 0, "control": 0}

    def fake_source(*, output_dir, run_id):
        del run_id
        calls["source"] += 1
        _write_source_artifacts(output_dir, source_result)
        return deepcopy(source_result)

    def fake_control(
        *,
        source_dir,
        output_dir,
        run_id,
        adapter,
        remaining_credit,
        bootstrap_samples,
    ):
        del source_dir, run_id, adapter, remaining_credit, bootstrap_samples
        calls["control"] += 1
        _write_control_artifacts(output_dir, control_result)
        return deepcopy(control_result)

    monkeypatch.setattr(run, "validate_predecessors", lambda: {})
    monkeypatch.setattr(run, "run_fresh_source", fake_source)
    monkeypatch.setattr(run, "run_fresh_control", fake_control)
    return calls


def test_frozen_counts_and_balance_gate() -> None:
    assert run.SOURCE_EXPECTED_REQUESTS == 3_680
    assert run.CONTROL_EXPECTED_REQUESTS == 3_072
    assert run.COMPOSITE_EXPECTED_REQUESTS == 6_752
    assert run.COMPOSITE_RUN_BUDGET_USD == 9.5
    assert run.TREE_SEEDS == tuple(range(100_000, 100_032))
    assert run.TARGET_SEEDS == tuple(range(100_100, 100_132))
    assert run.CONTROL_SEED_START == 10_000_000

    run.require_starting_balance(10.25)
    with pytest.raises(RuntimeError, match="below the frozen"):
        run.require_starting_balance(10.249)


def test_predecessor_results_are_hash_bound() -> None:
    predecessors = run.validate_predecessors()

    assert predecessors["source"]["protocol"]["tree_count"] == 96
    assert predecessors["control"]["status"] == "passed"


def test_source_context_uses_fresh_schedule_and_restores() -> None:
    names = (
        "INTERFACE_VERSION",
        "TREE_SEEDS",
        "TARGET_SEEDS",
        "VALIDATION_SEED_START",
        "TREE_COUNT",
        "EXPECTED_REQUESTS",
        "RUN_BUDGET_USD",
        "BOOTSTRAP_SEED",
        "mechanics_gates",
        "finalize_result",
    )
    originals = {name: getattr(run.source, name) for name in names}

    with run.configured_source_module():
        assert run.source.INTERFACE_VERSION == run.SOURCE_INTERFACE_VERSION
        assert run.source.TREE_SEEDS == run.TREE_SEEDS
        assert run.source.TARGET_SEEDS == run.TARGET_SEEDS
        assert run.source.TREE_COUNT == 32
        assert run.source.EXPECTED_REQUESTS == 3_680
        assert run.source.mechanics_gates is run.source_mechanics_gates
        assert run.source.finalize_result is run.finalize_source_result

    assert {name: getattr(run.source, name) for name in names} == originals


def test_source_mechanics_accept_exact_clean_run() -> None:
    scored = [
        {
            "mechanics": {
                "initial_valid": 24,
                "minimum_first_branch_valid": 12,
                "minimum_retained_second_branch_valid": 8,
                "validation_support_count": 16,
                "minimum_validation_support_valid": 16,
            }
        }
        for _ in range(32)
    ]
    usage = {
        "adapter_requests": 3_680,
        "http_attempts": 3_681,
        "retry_count": 1,
        "provider_error_retries": 1,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 4.8,
    }
    targets = [
        type("Target", (), {"extension": (index,)})()
        for index in range(33)
    ]
    parse_summary = {
        "parse_events": run.SOURCE_EXPECTED_PARSE_EVENTS,
        "pooled_parse_events": (
            run.SOURCE_EXPECTED_POOLED_PARSE_EVENTS
        ),
        "provider_draws_parsed": 3_680,
        "item_salvaged_draws": 0,
    }

    gates = run.source_mechanics_gates(
        scored_trees=scored,
        usage=usage,
        targets=targets,
        parse_summary=parse_summary,
        fallback_events=[],
    )

    assert all(gates.values())


def test_source_finalization_uses_32_tree_gates(monkeypatch) -> None:
    passing = {
        "relative_brier_reduction": 0.09,
        "tree_cluster_brier_difference_95pct_bootstrap": [
            -0.02,
            -0.001,
        ],
        "brier_tree_wins": 21,
    }
    monkeypatch.setattr(
        run.p96,
        "comparison_with_frozen_bootstrap",
        lambda *args, **kwargs: deepcopy(passing),
    )
    monkeypatch.setattr(
        run.source.pooled,
        "parser_accounting",
        lambda _: {"parse_events": run.SOURCE_EXPECTED_PARSE_EVENTS},
    )
    result = {
        "aggregate": {"comparisons": {}},
        "trees": [
            {
                "selection": {
                    "crossfit_depth_three_root": index,
                    "fixed_support_depth_three_root": index + 1,
                }
            }
            for index in range(32)
        ],
        "protocol": {},
        "primary_gates": {"old": True},
        "mechanics_gates": {"mechanics": True},
    }

    run.finalize_source_result(
        result,
        parse_events=[],
        fallback_events=[],
    )

    assert result["status"] == "passed"
    assert all(result["myopic_policy_gates"].values())
    assert all(result["dynamic_support"]["gates"].values())
    assert result["dynamic_support"]["root_differences"] == 32
    assert "primary_gates" not in result


def test_control_context_loads_fresh_source_shape(
    tmp_path,
) -> None:
    source_result = json.loads(
        run.quality.SOURCE_RESULT.read_text(encoding="utf-8")
    )
    source_trees = json.loads(
        run.quality.SOURCE_TREES.read_text(encoding="utf-8")
    )
    source_targets = json.loads(
        run.quality.SOURCE_TARGETS.read_text(encoding="utf-8")
    )
    result_path = tmp_path / "RESULT.json"
    trees_path = tmp_path / "TREES.json"
    targets_path = tmp_path / "TARGETS.json"
    result_path.write_text(
        json.dumps({**source_result, "trees": source_result["trees"][:32]})
    )
    trees_path.write_text(
        json.dumps({**source_trees, "trees": source_trees["trees"][:32]})
    )
    targets_path.write_text(json.dumps(source_targets))
    hashes = {
        "result": run.sha256_file(result_path),
        "trees": run.sha256_file(trees_path),
        "targets": run.sha256_file(targets_path),
    }

    with run.configured_control_source(
        source_result_path=result_path,
        source_trees_path=trees_path,
        source_targets_path=targets_path,
        source_result_sha256=hashes["result"],
        source_trees_sha256=hashes["trees"],
        source_targets_sha256=hashes["targets"],
        tree_seeds=tuple(range(80_000, 80_032)),
    ):
        loaded_result, loaded_trees, targets = (
            run.quality.load_and_validate_sources(
                result_path=result_path,
                trees_path=trees_path,
                targets_path=targets_path,
            )
        )
        manifest = run.control.request_manifest(
            loaded_trees["trees"]
        )

    assert len(loaded_result["trees"]) == 32
    assert len(targets) == 33
    assert len(manifest) == 3_072
    assert manifest[0]["seed"] == 10_000_000


def test_source_science_does_not_control_stage_b(
    tmp_path,
    monkeypatch,
) -> None:
    source_result = _source_result(
        myopic_science=False,
        dynamic_science=False,
    )
    control_result = _control_result()
    calls = _install_fake_stages(
        monkeypatch,
        source_result=source_result,
        control_result=control_result,
    )

    result = run.run_combined(
        output_dir=tmp_path,
        run_id="science-null-still-runs-control",
        remaining_credit=10.25,
        control_remaining_credit=5.0,
        bootstrap_samples=10,
    )

    assert calls == {"source": 1, "control": 1}
    assert result["status"] == "gated_null"
    assert result["decision"] == "complete_composite_endpoint"
    assert (
        result["protocol"][
            "source_science_did_not_authorize_control_stage"
        ]
        is True
    )


def test_source_mechanics_failure_stops_before_control(
    tmp_path,
    monkeypatch,
) -> None:
    calls = _install_fake_stages(
        monkeypatch,
        source_result=_source_result(mechanics=False),
        control_result=_control_result(),
    )

    result = run.run_combined(
        output_dir=tmp_path,
        run_id="mechanics-stop",
        remaining_credit=10.25,
    )

    assert calls == {"source": 1, "control": 0}
    assert result["status"] == "mechanics_failed"
    assert result["decision"] == "stop_before_control_source_mechanics"


def test_changed_root_opportunity_failure_stops_before_control(
    tmp_path,
    monkeypatch,
) -> None:
    calls = _install_fake_stages(
        monkeypatch,
        source_result=_source_result(opportunity=19),
        control_result=_control_result(),
    )

    result = run.run_combined(
        output_dir=tmp_path,
        run_id="opportunity-stop",
        remaining_credit=10.25,
    )

    assert calls == {"source": 1, "control": 0}
    assert result["status"] == "opportunity_failed"
    assert result["decision"] == (
        "stop_before_control_changed_root_floor"
    )


def test_all_component_gates_pass_composite(
    tmp_path,
    monkeypatch,
) -> None:
    calls = _install_fake_stages(
        monkeypatch,
        source_result=_source_result(),
        control_result=_control_result(),
    )

    result = run.run_combined(
        output_dir=tmp_path,
        run_id="all-pass",
        remaining_credit=10.25,
        control_remaining_credit=5.0,
        bootstrap_samples=10,
    )

    assert calls == {"source": 1, "control": 1}
    assert result["status"] == "passed"
    assert all(result["composite_gates"].values())
    assert result["usage"]["total_requests"] == 6_752
    assert result["usage"]["total_cost_usd"] == pytest.approx(7.9)


def test_control_mechanics_failure_is_not_generic_failure(
    tmp_path,
    monkeypatch,
) -> None:
    calls = _install_fake_stages(
        monkeypatch,
        source_result=_source_result(),
        control_result=_control_result(mechanics=False),
    )

    result = run.run_combined(
        output_dir=tmp_path,
        run_id="control-mechanics-failure",
        remaining_credit=10.25,
        control_remaining_credit=5.0,
        bootstrap_samples=10,
    )

    assert calls == {"source": 1, "control": 1}
    assert result["status"] == "mechanics_failed"
    assert result["decision"] == "stop_at_control_mechanics"
