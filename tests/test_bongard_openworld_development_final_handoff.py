from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import bongard_openworld_development_final_handoff as handoff
from scripts import bongard_openworld_luna_paper_fragment as luna_fragment
from scripts import bongard_openworld_luna_vlm_development as development


def _write(path: Path, value: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _bindings() -> dict:
    return {"verified": True}


class Harness:
    def __init__(self, root: Path, *, claim_tier: str = "development_null") -> None:
        self.output = root / "final"
        self.combined = root / "COMBINED_RESULT.json"
        self.claim_report = root / "CLAIM_REPORT.json"
        self.blocks = [root / f"block-{block_id}.json" for block_id in development.BLOCK_ORDER]
        self.paired = {
            block_id: root / f"paired-{block_id}.json"
            for block_id in development.BLOCK_ORDER
        }
        self.claim_tier = claim_tier
        self.calls: list[str] = []
        _write(self.combined, {"status": "development_signal"})
        for block_id, block_path in zip(
            development.BLOCK_ORDER, self.blocks, strict=True
        ):
            _write(block_path, {"block_id": block_id})
            _write(
                self.paired[block_id],
                {
                    "status": "paired_daily_complete",
                    "block_id": block_id,
                },
            )

    def paired_validator(self, *, block_id: str) -> dict:
        self.calls.append(f"paired:{block_id}")
        return json.loads(self.paired[block_id].read_text(encoding="utf-8"))

    def claim_runner(self, **kwargs) -> dict:
        self.calls.append("claim")
        assert kwargs["combined_result"] == self.combined
        assert list(kwargs["block_results"]) == list(development.BLOCK_ORDER)
        _write(
            kwargs["claim_report_path"],
            {"status": "claim_scope_frozen", "claim_tier": self.claim_tier},
        )
        return {
            "status": (
                "confirmation_handoff_verified"
                if self.claim_tier
                == "full_path_dependent_llm_native_development_signal"
                else "development_claim_banked_confirmation_forbidden"
            ),
            "claim_tier": self.claim_tier,
            "confirmation_authorization": (
                {"verified": True}
                if self.claim_tier
                == "full_path_dependent_llm_native_development_signal"
                else None
            ),
        }

    def component_runner(self, name: str):
        statuses = {
            "classical": "classical_suite_complete",
            "mediation": "path_mediation_complete",
            "compute": "compute_matched_control_audit_complete",
            "random": "random_strategy_control_audit_complete",
        }

        def run(*, output_path: Path, stage: str, result_path: Path, block_results):
            self.calls.append(name)
            assert stage == "development"
            assert result_path == self.combined
            assert list(block_results) == self.blocks
            result = {"status": statuses[name], "stage": stage, "name": name}
            _write(output_path, result)
            return result

        return run

    def paper_runner(self, *, output: Path, stage: str, **kwargs) -> dict:
        self.calls.append("paper")
        assert stage == "development"
        assert kwargs["combined_result"] == self.combined
        _write(output, "DETERMINISTIC PAPER\n")
        metadata = {"stage": stage, "claim_tier": self.claim_tier}
        _write(output.with_suffix(".json"), metadata)
        headline = output.with_name(luna_fragment.HEADLINE_FILENAME)
        _write(headline, "DETERMINISTIC HEADLINE\n")
        return {
            "status": "written_with_mandatory_classical_compute_random_and_mediation_suites",
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
            "paired_paths": self.paired,
            "claim_report_path": self.claim_report,
            "paired_validator": self.paired_validator,
            "claim_runner": self.claim_runner,
            "classical_runner": self.component_runner("classical"),
            "mediation_runner": self.component_runner("mediation"),
            "compute_runner": self.component_runner("compute"),
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


def test_preflight_waits_for_first_missing_paired_block(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.paired["b"].unlink()
    result = handoff.preflight_final_handoff(
        output_dir=harness.output,
        paired_paths=harness.paired,
        combined_result=harness.combined,
        paired_validator=harness.paired_validator,
        binding_verifier=_bindings,
    )
    assert result["status"] == "waiting_for_block_b"
    assert set(result["completed_paired_blocks"]) == {"a"}
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0
    assert not harness.output.exists()


def test_preflight_ready_replays_all_paired_blocks(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    result = handoff.preflight_final_handoff(
        output_dir=harness.output,
        paired_paths=harness.paired,
        combined_result=harness.combined,
        paired_validator=harness.paired_validator,
        binding_verifier=_bindings,
    )
    assert result["status"] == "ready_without_model_calls"
    assert list(result["completed_paired_blocks"]) == list(development.BLOCK_ORDER)


@pytest.mark.parametrize(
    "claim_tier",
    [
        "development_null",
        "full_path_dependent_llm_native_development_signal",
    ],
)
def test_all_dispositions_run_complete_terminal_sequence(
    tmp_path: Path, claim_tier: str
) -> None:
    harness = Harness(tmp_path, claim_tier=claim_tier)
    result = harness.run()

    assert harness.calls == [
        "paired:a",
        "paired:b",
        "paired:c",
        "paired:d",
        "claim",
        "classical",
        "mediation",
        "compute",
        "random",
        "paper",
    ]
    assert result["status"] == "development_handoff_complete"
    assert result["claim_tier"] == claim_tier
    assert result["existing_claim_authorizes_confirmation"] is (
        claim_tier == "full_path_dependent_llm_native_development_signal"
    )
    assert result["all_dispositions_rendered_identically"] is True
    assert result["model_calls"] == 0
    assert result["cost_usd"] == 0.0
    assert result["authorizes_paid_calls"] is False
    assert result["authorizes_rerun"] is False
    assert result["this_record_authorizes_confirmation"] is False
    assert set(result["components"]) == {
        "paired_a",
        "paired_b",
        "paired_c",
        "paired_d",
        "claim_finalization",
        "claim_report",
        "classical_suite",
        "path_mediation",
        "compute_matched_control",
        "random_strategy_control",
        "paper_tex",
        "paper_metadata",
        "paper_headline",
    }


@pytest.mark.parametrize(
    ("failed_name", "failed_stage", "expected_prefix", "replay_calls"),
    [
        (
            "classical",
            "classical_suite",
            {"claim_finalization", "claim_report"},
            ["claim"],
        ),
        (
            "mediation",
            "path_mediation",
            {"claim_finalization", "claim_report", "classical_suite"},
            ["claim", "classical"],
        ),
        (
            "compute",
            "compute_matched_control",
            {
                "claim_finalization",
                "claim_report",
                "classical_suite",
                "path_mediation",
            },
            ["claim", "classical", "mediation"],
        ),
        (
            "random",
            "random_strategy_control",
            {
                "claim_finalization",
                "claim_report",
                "classical_suite",
                "path_mediation",
                "compute_matched_control",
            },
            ["claim", "classical", "mediation", "compute"],
        ),
        (
            "paper",
            "paper",
            {
                "claim_finalization",
                "claim_report",
                "classical_suite",
                "path_mediation",
                "compute_matched_control",
                "random_strategy_control",
            },
            ["claim", "classical", "mediation", "compute", "random"],
        ),
    ],
)
def test_downstream_failure_banks_exact_prefix_once(
    tmp_path: Path,
    failed_name: str,
    failed_stage: str,
    expected_prefix: set[str],
    replay_calls: list[str],
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
    assert expected_prefix.issubset(result["components"])
    assert result["authorizes_paid_calls"] is False
    assert result["authorizes_rerun"] is False
    assert result["this_record_authorizes_confirmation"] is False
    harness.calls.clear()
    replay = harness.run()
    assert replay == result
    assert harness.calls == [
        "paired:a",
        "paired:b",
        "paired:c",
        "paired:d",
        *replay_calls,
    ]


def test_complete_handoff_replays_every_component_without_new_artifact(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    first = harness.run()
    harness.calls.clear()
    replay = harness.run()

    assert replay == first
    assert harness.calls == [
        "paired:a",
        "paired:b",
        "paired:c",
        "paired:d",
        "claim",
        "classical",
        "mediation",
        "compute",
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


def test_changed_component_hash_is_rejected_before_replay(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.run()
    _write(harness.output / "CLASSICAL_SUITE_RESULT.json", {"changed": True})
    with pytest.raises(ValueError, match="component changed"):
        harness.run()
