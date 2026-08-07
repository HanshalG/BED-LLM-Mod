from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import regretbench_deepseek_confirmation_result_verify as verify
from scripts import regretbench_deepseek_dynamic_depth2_confirmation as confirmation
from scripts import regretbench_deepseek_dynamic_depth2_policy as policy
from scripts import regretbench_deepseek_support_recovery as recovery
from tests.test_regretbench_deepseek_dynamic_depth2_policy import _FixtureAdapter


class _ConfirmationFixtureAdapter(_FixtureAdapter):
    def __init__(self) -> None:
        super().__init__()
        for index, cig in enumerate(confirmation.load_confirmation_cigs()):
            facets = [facet.replace("_", " ") for facet in cig.semantic_facets]
            self.facets[cig.cig_id] = facets
            _, truth = recovery.sample_truth(
                cig, confirmation.TRUTH_SEED_START + index
            )
            self.truth_aliases[cig.cig_id] = str(
                (truth.slots or {})["answer_aliases"]
            ).split("|")[0]
            self.truth_replies[cig.cig_id] = {
                facet.replace("_", " "): str(
                    (truth.slots or {}).get(facet, "")
                )
                for facet in cig.semantic_facets
            }


def _authorization() -> dict:
    return {
        "status": "authorized",
        "development_status": "passed",
        "independent_replay_status": "verified",
        "development_result_sha256": "1" * 64,
        "development_verification_sha256": "2" * 64,
        "development_daily_result_sha256": "3" * 64,
    }


def test_confirmation_scope_is_disjoint_and_restores_development_globals() -> None:
    original_seed = policy.INITIAL_SEED_START
    original_loader = recovery.load_stage_cigs
    development_ids = [cig.cig_id for cig in original_loader("development")]

    with confirmation.confirmation_scope():
        confirmation_ids = [
            cig.cig_id for cig in recovery.load_stage_cigs("development")
        ]
        assert policy.INITIAL_SEED_START == confirmation.INITIAL_SEED_START

    assert policy.INITIAL_SEED_START == original_seed
    assert recovery.load_stage_cigs is original_loader
    assert set(confirmation_ids).isdisjoint(development_ids)
    assert len(confirmation_ids) == 64
    seed_sets = confirmation.expected_seed_sets()
    assert len(set().union(*seed_sets.values())) == 1345


def test_confirmation_replay_does_not_import_experiment_producers() -> None:
    source = Path(verify.__file__).read_text(encoding="utf-8")

    assert "regretbench_deepseek_dynamic_depth2_confirmation as" not in source
    assert "regretbench_deepseek_dynamic_depth2_policy as" not in source


def test_confirmation_refuses_nonpassing_development_before_adapter_calls(
    tmp_path,
) -> None:
    adapter = _ConfirmationFixtureAdapter()
    authorization = _authorization()
    authorization["development_status"] = "gated_null"

    with pytest.raises(ValueError, match="literal verified development pass"):
        confirmation.run_confirmation(
            output_dir=tmp_path / "confirmation",
            run_id="fixture-confirmation",
            support_smoke_result=tmp_path / "support-smoke.json",
            support_development_result=tmp_path / "support-development.json",
            policy_smoke_result=tmp_path / "policy-smoke.json",
            development_authorization=authorization,
            adapter=adapter,
            bootstrap_samples=20,
        )

    assert adapter.requests == 0


def test_exact_scale_confirmation_replays_from_raw_and_detects_tampering(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        policy,
        "validate_support_predecessors",
        lambda **kwargs: {"support": "fixture"},
    )
    monkeypatch.setattr(
        policy,
        "validate_policy_smoke",
        lambda path: {"path": str(path), "sha256": "fixture"},
    )
    output = tmp_path / "confirmation"
    adapter = _ConfirmationFixtureAdapter()

    result = confirmation.run_confirmation(
        output_dir=output,
        run_id="fixture-confirmation",
        support_smoke_result=tmp_path / "support-smoke.json",
        support_development_result=tmp_path / "support-development.json",
        policy_smoke_result=tmp_path / "policy-smoke.json",
        development_authorization=_authorization(),
        adapter=adapter,
        bootstrap_samples=50,
    )

    assert result["status"] != "mechanics_failed"
    assert result["confirmation_opened"] is True
    assert result["development_opened"] is False
    assert result["protocol"]["source_split"] == "confirmation"
    assert result["protocol"]["maximum_requests"] == 8768
    assert result["usage"]["naive_luna"]["adapter_requests"] == 0
    assert result["usage"]["deepseek_naive_endpoint"]["adapter_requests"] == 0
    assert all(
        "naive_thinking" not in task["policies"] for task in result["tasks"]
    )
    assert confirmation.selected_task_ids(result["tasks"]) == [
        cig.cig_id for cig in confirmation.load_confirmation_cigs()
    ]
    raw_initial = json.loads(
        (output / "private" / "RAW_INITIAL.json").read_text()
    )
    assert raw_initial["seeds"] == [
        confirmation.INITIAL_SEED_START + index for index in range(64)
    ]

    replay = verify.verify_confirmation(output)
    assert replay["status"] == "verified"
    assert replay["checks"]["scientific_endpoint_recomputed"] is True
    assert replay["checks"]["exact_untouched_confirmation_cohort"] is True
    assert replay["model_calls"] == 0

    result_path = output / "RESULT.json"
    tampered = json.loads(result_path.read_text())
    tampered["tasks"][0]["policies"]["dynamic_depth2"]["brier"] += 0.01
    result_path.write_text(json.dumps(tampered))
    failed = verify.verify_confirmation(output)
    assert failed["status"] == "verification_failed"
    assert "$.tasks[0].policies.dynamic_depth2.brier" in failed["mismatches"]
