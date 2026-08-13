from __future__ import annotations

from pathlib import Path

from scripts import autumnbench_llm_native_source_admission as source


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_public_manifest_fails_before_payloads() -> None:
    result = source.audit(
        manifest_path=Path(
            "/tmp/bed-source-audits/autumnbench-source-20260813/manifest_public.json"
        ),
        autumn_repo=Path("/tmp/bed-source-audits/autumn-cpp-20260813"),
        mara_repo=Path("/tmp/bed-source-audits/mara-protocol-20260813"),
        protocol_path=REPO_ROOT
        / "results/nonmyopic/AUTUMNBENCH_LLM_NATIVE_SOURCE_ADMISSION_PROTOCOL_20260813.md",
    )
    assert result["status"] == "source_failed_closed"
    assert result["failure_gate"] == "exact_129_tasks_43_worlds"
    assert result["manifest"]["observed_task_count"] == 60
    assert result["manifest"]["unique_base_world_count"] == 20
    assert result["manifest"]["all_observed_worlds_form_triplets"] is True
    assert result["payloads"] == {
        "nonfixture_programs_downloaded": 0,
        "nonfixture_prompts_downloaded": 0,
        "nonfixture_answers_downloaded": 0,
        "fixture_executions": 0,
        "model_calls": 0,
        "endpoint_outcomes_opened": 0,
    }
    assert result["gates"]["local_deterministic_handshake"] is None
    assert result["authorizes"] == "nothing"
