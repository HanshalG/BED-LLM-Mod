from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import number_game_qwen_fully_fresh_control_verify as verify


def _comparison(*, reduction: float) -> dict:
    return {
        "relative_brier_reduction": reduction,
        "tree_cluster_brier_difference_95pct_bootstrap": [-0.02, -0.001],
        "brier_tree_wins": 20,
        "brier_tree_ties": 2,
        "brier_tree_losses": 10,
    }


def _analysis() -> dict:
    return {
        "scientific_gates": {"quality": True, "calibration": True},
        "stages": {
            "second": {
                "conditional": {
                    "posterior_predictive_mse": 0.03,
                    "truth_extension_coverage": 0.6,
                },
                "history_blind": {
                    "posterior_predictive_mse": 0.04,
                    "truth_extension_coverage": 0.4,
                },
                "differences": {
                    "conditional_minus_history_blind_predictive_mse": -0.01,
                    "conditional_minus_history_blind_truth_coverage": 0.2,
                },
            }
        },
        "selected_root_prompt_conditioning": {
            "prompt_benefit_contrast_to_realized_advantage_spearman": 0.5,
        },
        "bootstrap": {
            "prompt_benefit_contrast_to_realized_spearman_95pct": [0.1, 0.8],
        },
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _build_run(tmp_path: Path) -> tuple[Path, dict, dict, dict]:
    run_dir = tmp_path / "run"
    source_dir = run_dir / "source"
    control_dir = run_dir / "control"
    source_result = {
        "status": "gated_null",
        "protocol": {"source_calendar_date": "2026-08-06"},
        "usage": {
            "adapter_requests": 3680,
            "run_cost_usd": 4.1,
        },
        "mechanics_gates": {"mechanics": True},
        "myopic_policy_gates": {"myopic": True},
        "dynamic_support": {
            "root_differences": 29,
            "gates": {"reduction": False},
            "comparison": _comparison(reduction=0.028),
        },
        "aggregate": {
            "comparisons": {
                "myopic_eig": _comparison(reduction=0.157),
            }
        },
    }
    _write_json(source_dir / "RESULT.json", source_result)
    _write_json(source_dir / "TREES.json", {"trees": [{"tree": 1}]})
    _write_json(source_dir / "TARGETS.json", {"targets": [{"target": 1}]})
    source_hashes = {
        "result": verify.sha256_file(source_dir / "RESULT.json"),
        "trees": verify.sha256_file(source_dir / "TREES.json"),
        "targets": verify.sha256_file(source_dir / "TARGETS.json"),
    }

    controls = {"trees": [{"branches": {}}]}
    _write_json(control_dir / "CONTROLS.json", controls)
    novelty = {
        "branch_slot_count": 1,
        "minimum": 2,
        "maximum": 2,
        "mean": 2.0,
        "fraction_at_least_two": 1.0,
        "distribution": {"2": 1},
        "used_as_gate": False,
    }
    control_result = {
        "status": "passed",
        "usage": {
            "adapter_requests": 3072,
            "run_cost_usd": 3.1,
        },
        "mechanics_gates": {"mechanics": True},
        "analysis": _analysis(),
        "trees": [{"tree": 1}],
        "second_draw_novelty": novelty,
        "protocol": {
            "controls_sha256": verify.sha256_file(
                control_dir / "CONTROLS.json"
            ),
            "source_result_sha256": source_hashes["result"],
            "source_trees_sha256": source_hashes["trees"],
            "source_targets_sha256": source_hashes["targets"],
            "control_seed_start": verify.base.CONTROL_SEED_START,
            "bootstrap_seed": verify.base.CONTROL_BOOTSTRAP_SEED,
            "bootstrap_samples": verify.base.BOOTSTRAP_SAMPLES,
        },
    }
    _write_json(control_dir / "RESULT.json", control_result)
    control_hashes = {
        "result": verify.sha256_file(control_dir / "RESULT.json"),
        "controls": verify.sha256_file(control_dir / "CONTROLS.json"),
    }

    authorization = {
        "source_artifacts": source_hashes,
        "source_calendar_date": "2026-08-06",
        "source_science_was_not_an_authorization_input": True,
        "structural_authorization": verify.base.source_control_authorization(
            source_result
        ),
    }
    _write_json(run_dir / "CONTROL_AUTHORIZATION.json", authorization)
    source_stage = {
        "source_artifacts": source_hashes,
        "control_authorization_sha256": verify.sha256_file(
            run_dir / "CONTROL_AUTHORIZATION.json"
        ),
    }
    _write_json(run_dir / "SOURCE_STAGE.json", source_stage)

    gates, usage, status = verify._expected_composite(
        source_result=source_result,
        control_result=control_result,
    )
    composite = {
        "status": status,
        "decision": "complete_composite_endpoint",
        "protocol": {
            "source_artifacts": source_hashes,
            "control_artifacts": control_hashes,
            "source_calendar_date": "2026-08-06",
            "control_calendar_date": "2026-08-07",
            "control_authorization_sha256": verify.sha256_file(
                run_dir / "CONTROL_AUTHORIZATION.json"
            ),
            "control_authorization": authorization[
                "structural_authorization"
            ],
            "source_science_was_not_an_authorization_input": True,
        },
        "source": verify._expected_source_summary(source_result),
        "control": verify._expected_control_summary(control_result),
        "composite_gates": gates,
        "usage": usage,
    }
    _write_json(run_dir / "RESULT.json", composite)
    _write_json(
        run_dir / "CONTROL_STAGE.json",
        {
            "status": composite["status"],
            "decision": composite["decision"],
            "result_sha256": verify.sha256_file(run_dir / "RESULT.json"),
        },
    )
    return run_dir, source_result, control_result, controls


def test_complete_public_run_replays_and_writes_verification(
    tmp_path: Path,
) -> None:
    run_dir, _, control_result, controls = _build_run(tmp_path)
    replay_calls = []

    def replay_fn(**kwargs):
        replay_calls.append(kwargs)
        return {
            "trees": deepcopy(control_result["trees"]),
            "analysis": deepcopy(control_result["analysis"]),
        }

    result = verify.verify_completed_control(
        run_dir=run_dir,
        replay_fn=replay_fn,
        novelty_fn=lambda value: (
            deepcopy(control_result["second_draw_novelty"])
            if value == controls
            else None
        ),
    )

    assert result["status"] == "verified"
    assert result["provider_calls"] == 0
    assert result["verification_cost_usd"] == 0.0
    assert all(result["checks"].values())
    assert result["summary"]["composite_status"] == "gated_null"
    assert result["recomputed_composite_usage"]["total_requests"] == 6752
    assert len(replay_calls) == 1
    assert (run_dir / "CONTROL_VERIFICATION.json").exists()
    markdown = (run_dir / "CONTROL_VERIFICATION.md").read_text(
        encoding="utf-8"
    )
    assert "Status: **verified**" in markdown
    assert "model calls / cost for verification: `0 / $0`" in markdown


def test_source_artifact_tamper_fails_before_replay(tmp_path: Path) -> None:
    run_dir, _, _, _ = _build_run(tmp_path)
    source = run_dir / "source" / "TREES.json"
    source.write_text(source.read_text(encoding="utf-8") + "\n")
    replay_calls = []

    with pytest.raises(ValueError, match="source_hashes"):
        verify.verify_completed_control(
            run_dir=run_dir,
            replay_fn=lambda **kwargs: replay_calls.append(kwargs),
            novelty_fn=lambda _: {},
        )

    assert replay_calls == []


def test_control_stage_hash_tamper_fails_before_replay(tmp_path: Path) -> None:
    run_dir, _, _, _ = _build_run(tmp_path)
    stage = json.loads((run_dir / "CONTROL_STAGE.json").read_text())
    stage["result_sha256"] = "0" * 64
    _write_json(run_dir / "CONTROL_STAGE.json", stage)

    with pytest.raises(ValueError, match="control_stage_hash"):
        verify.verify_completed_control(
            run_dir=run_dir,
            replay_fn=lambda **_: {},
            novelty_fn=lambda _: {},
        )


def test_replayed_analysis_drift_is_rejected(tmp_path: Path) -> None:
    run_dir, _, control_result, _ = _build_run(tmp_path)
    changed = deepcopy(control_result["analysis"])
    changed["scientific_gates"]["quality"] = False

    with pytest.raises(ValueError, match="replayed_analysis"):
        verify.verify_completed_control(
            run_dir=run_dir,
            replay_fn=lambda **_: {
                "trees": deepcopy(control_result["trees"]),
                "analysis": changed,
            },
            novelty_fn=lambda _: deepcopy(
                control_result["second_draw_novelty"]
            ),
        )


def test_novelty_drift_is_rejected(tmp_path: Path) -> None:
    run_dir, _, control_result, _ = _build_run(tmp_path)
    changed = deepcopy(control_result["second_draw_novelty"])
    changed["mean"] = 99.0

    with pytest.raises(ValueError, match="second_draw_novelty"):
        verify.verify_completed_control(
            run_dir=run_dir,
            replay_fn=lambda **_: {
                "trees": deepcopy(control_result["trees"]),
                "analysis": deepcopy(control_result["analysis"]),
            },
            novelty_fn=lambda _: changed,
        )


def test_composite_gate_drift_is_rejected(tmp_path: Path) -> None:
    run_dir, _, control_result, _ = _build_run(tmp_path)
    composite_path = run_dir / "RESULT.json"
    composite = json.loads(composite_path.read_text())
    composite["composite_gates"]["source_dynamic_support_passed"] = True
    _write_json(composite_path, composite)
    stage = json.loads((run_dir / "CONTROL_STAGE.json").read_text())
    stage["result_sha256"] = verify.sha256_file(composite_path)
    _write_json(run_dir / "CONTROL_STAGE.json", stage)

    with pytest.raises(ValueError, match="composite_gates"):
        verify.verify_completed_control(
            run_dir=run_dir,
            replay_fn=lambda **_: {
                "trees": deepcopy(control_result["trees"]),
                "analysis": deepcopy(control_result["analysis"]),
            },
            novelty_fn=lambda _: deepcopy(
                control_result["second_draw_novelty"]
            ),
        )
