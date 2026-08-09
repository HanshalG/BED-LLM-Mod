from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import bongard_openworld_confirmation_final_handoff as handoff
from scripts import bongard_openworld_luna_confirmation64 as confirmation
from scripts import bongard_openworld_luna_paper_fragment as luna_fragment


def _write(path: Path, value: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _bindings() -> dict:
    return {"verified": True}


class Harness:
    def __init__(
        self, root: Path, *, claim_tier: str = "confirmation_null"
    ) -> None:
        self.output = root / "final"
        self.combined = root / "COMBINED_RESULT.json"
        self.blocks = [
            root / f"block-{block_id}.json"
            for block_id in confirmation.BLOCK_ORDER
        ]
        self.claim_tier = claim_tier
        self.calls: list[str] = []
        _write(
            self.combined,
            {
                "status": (
                    "confirmation_pass"
                    if claim_tier == "full_llm_native_confirmation"
                    else "confirmation_null"
                ),
                "claim_tier": claim_tier,
            },
        )
        for block_id, path in zip(
            confirmation.BLOCK_ORDER, self.blocks, strict=True
        ):
            _write(path, {"status": "block_mechanics_pass", "block_id": block_id})

    def combined_verifier(self, *, result_path: Path, block_results) -> dict:
        self.calls.append("verify")
        assert result_path == self.combined
        assert list(block_results) == self.blocks
        return {
            "verified": True,
            "status": (
                "confirmation_pass"
                if self.claim_tier == "full_llm_native_confirmation"
                else "confirmation_null"
            ),
            "claim_tier": self.claim_tier,
            "result_sha256": handoff._sha256(self.combined),
            "sealed_test_authorized": False,
        }

    def component_runner(self, name: str):
        statuses = {
            "classical": "classical_suite_complete",
            "mediation": "path_mediation_complete",
            "compute": "compute_matched_control_audit_complete",
            "opportunity": "classical_horizon_opportunity_stratum_complete",
            "random": "random_strategy_control_audit_complete",
        }

        def run(*, output_path: Path, stage: str, result_path: Path, block_results):
            self.calls.append(name)
            assert stage == "confirmation"
            assert result_path == self.combined
            assert list(block_results) == self.blocks
            result = {"status": statuses[name], "stage": stage, "name": name}
            _write(output_path, result)
            return result

        return run

    def paper_runner(self, *, output: Path, stage: str, **kwargs) -> dict:
        self.calls.append("paper")
        assert stage == "confirmation"
        assert kwargs["combined_result"] == self.combined
        assert kwargs["opportunity_path"] == (
            self.output / "CLASSICAL_HORIZON_OPPORTUNITY_RESULT.json"
        )
        _write(output, "DETERMINISTIC CONFIRMATION PAPER\n")
        metadata = {"stage": stage, "claim_tier": self.claim_tier}
        _write(output.with_suffix(".json"), metadata)
        headline = output.with_name(luna_fragment.HEADLINE_FILENAME)
        _write(headline, "DETERMINISTIC CONFIRMATION HEADLINE\n")
        return {
            "status": (
                "written_with_mandatory_classical_compute_opportunity_random_and_"
                "mediation_suites"
            ),
            "stage": stage,
            "claim_tier": self.claim_tier,
            "tex_path": str(output),
            "tex_sha256": handoff._sha256(output),
            "headline_path": str(headline),
            "headline_sha256": handoff._sha256(headline),
            "metadata_path": str(output.with_suffix(".json")),
            "metadata_sha256": handoff._sha256(output.with_suffix(".json")),
            "model_calls": 0,
            "cost_usd": 0.0,
        }

    def run(self, **overrides) -> dict:
        kwargs = {
            "output_dir": self.output,
            "combined_result": self.combined,
            "block_results": self.blocks,
            "combined_verifier": self.combined_verifier,
            "classical_runner": self.component_runner("classical"),
            "mediation_runner": self.component_runner("mediation"),
            "compute_runner": self.component_runner("compute"),
            "opportunity_runner": self.component_runner("opportunity"),
            "random_runner": self.component_runner("random"),
            "paper_runner": self.paper_runner,
            "binding_verifier": _bindings,
        }
        kwargs.update(overrides)
        return handoff.run_final_handoff(**kwargs)


def test_bound_terminal_components_are_exact() -> None:
    result = handoff.verify_bindings()
    assert result["protocol"]["sha256"] == handoff.PROTOCOL_SHA256
    assert result["implementations"] == {
        name: {"path": relative, "sha256": expected}
        for name, (relative, expected) in handoff.BOUND_IMPLEMENTATIONS.items()
    }
    assert result["confirmation_execution"]["verified"] is True


def test_preflight_waits_without_opening_combined_or_blocks(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.blocks[1].unlink()

    def bomb(**_) -> dict:
        raise AssertionError("combined verifier opened endpoints early")

    result = handoff.preflight_final_handoff(
        output_dir=harness.output,
        combined_result=harness.combined,
        block_results=harness.blocks,
        combined_verifier=bomb,
        binding_verifier=_bindings,
    )
    assert result["status"] == "waiting_for_block_b"
    assert result["completed_block_paths"] == [str(harness.blocks[0])]
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0
    assert not harness.output.exists()


def test_preflight_waits_for_combined_without_verification(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.combined.unlink()

    def bomb(**_) -> dict:
        raise AssertionError("combined verifier opened endpoints early")

    result = handoff.preflight_final_handoff(
        output_dir=harness.output,
        combined_result=harness.combined,
        block_results=harness.blocks,
        combined_verifier=bomb,
        binding_verifier=_bindings,
    )
    assert result["status"] == "waiting_for_combined_result"
    assert result["completed_block_paths"] == [str(path) for path in harness.blocks]


def test_preflight_ready_replays_combined_once(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    result = handoff.preflight_final_handoff(
        output_dir=harness.output,
        combined_result=harness.combined,
        block_results=harness.blocks,
        combined_verifier=harness.combined_verifier,
        binding_verifier=_bindings,
    )
    assert result["status"] == "ready_without_model_calls"
    assert result["combined_verification"]["verified"] is True
    assert harness.calls == ["verify"]


@pytest.mark.parametrize(
    "claim_tier", ["confirmation_null", "full_llm_native_confirmation"]
)
def test_both_dispositions_run_identical_terminal_sequence(
    tmp_path: Path, claim_tier: str
) -> None:
    harness = Harness(tmp_path, claim_tier=claim_tier)
    result = harness.run()
    assert harness.calls == [
        "verify",
        "classical",
        "mediation",
        "compute",
        "opportunity",
        "random",
        "paper",
    ]
    assert result["status"] == "confirmation_handoff_complete"
    assert result["claim_tier"] == claim_tier
    assert result["all_dispositions_rendered_identically"] is True
    assert result["model_calls"] == 0
    assert result["cost_usd"] == 0.0
    assert result["authorizes_paid_calls"] is False
    assert result["authorizes_rerun"] is False
    assert result["changes_claim_tier"] is False
    assert set(result["components"]) == {
        "block_a",
        "block_b",
        "block_c",
        "block_d",
        "combined_result",
        "classical_suite",
        "path_mediation",
        "compute_matched_control",
        "classical_horizon_opportunity",
        "random_strategy_control",
        "paper_tex",
        "paper_metadata",
        "paper_headline",
    }


@pytest.mark.parametrize(
    ("failed_name", "failed_stage", "prior_calls"),
    [
        ("classical", "classical_suite", []),
        ("mediation", "path_mediation", ["classical"]),
        ("compute", "compute_matched_control", ["classical", "mediation"]),
        (
            "opportunity",
            "classical_horizon_opportunity",
            ["classical", "mediation", "compute"],
        ),
        (
            "random",
            "random_strategy_control",
            ["classical", "mediation", "compute", "opportunity"],
        ),
        (
            "paper",
            "paper",
            ["classical", "mediation", "compute", "opportunity", "random"],
        ),
    ],
)
def test_downstream_failure_banks_exact_prefix_once(
    tmp_path: Path, failed_name: str, failed_stage: str, prior_calls: list[str]
) -> None:
    harness = Harness(tmp_path)

    def fail(**_) -> None:
        raise RuntimeError(f"{failed_name} failed")

    overrides = (
        {"paper_runner": fail}
        if failed_name == "paper"
        else {f"{failed_name}_runner": fail}
    )
    result = harness.run(**overrides)
    assert result["status"] == "failed_closed"
    assert result["failed_stage"] == failed_stage
    assert result["authorizes_paid_calls"] is False
    assert result["authorizes_rerun"] is False
    harness.calls.clear()
    replay = harness.run()
    assert replay == result
    assert harness.calls == ["verify", *prior_calls]


def test_complete_handoff_replays_without_new_artifact(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    first = harness.run()
    harness.calls.clear()
    replay = harness.run()
    assert replay == first
    assert harness.calls == [
        "verify",
        "classical",
        "mediation",
        "compute",
        "opportunity",
        "random",
        "paper",
    ]


def test_complete_handoff_rejects_nondeterministic_paper_metadata(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    harness.run()

    def path_dependent_paper(*, output: Path, stage: str, **kwargs) -> dict:
        result = harness.paper_runner(output=output, stage=stage, **kwargs)
        metadata_path = output.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["temporary_path"] = str(output.parent)
        _write(metadata_path, metadata)
        result["metadata_sha256"] = handoff._sha256(metadata_path)
        return result

    with pytest.raises(ValueError, match="byte-identically"):
        harness.run(paper_runner=path_dependent_paper)


def test_changed_input_component_hash_is_rejected(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.run()
    _write(harness.blocks[0], {"changed": True})
    with pytest.raises(ValueError, match="component changed"):
        harness.run()
