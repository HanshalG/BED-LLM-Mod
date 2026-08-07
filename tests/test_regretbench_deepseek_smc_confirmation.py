from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from scripts import regretbench_deepseek_smc_confirmation as confirmation
from scripts import regretbench_deepseek_smc_confirmation_report as report
from scripts import regretbench_deepseek_smc_confirmation_verify as verifier


def _experiment_test_helpers():
    path = Path(__file__).with_name(
        "test_regretbench_deepseek_smc_dynamic_depth2_experiment.py"
    )
    spec = importlib.util.spec_from_file_location("_smc_experiment_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ParentAdapter:
    def __init__(self) -> None:
        self.requests = 0
        self.seeds: list[int] = []

    def chat_complete_seeded_messages_batched_structured(
        self,
        messages,
        seeds,
        *,
        temperature,
        response_format,
        max_new_tokens,
    ):
        assert response_format["json_schema"]["name"] == "regretbench_support"
        self.requests += len(messages)
        self.seeds.extend(seeds)
        return [
            json.dumps(
                {
                    "hypotheses": [
                        {
                            "interpretation": f"interpretation {index}",
                            "final_answer": f"answer {index}",
                            "prior_weight": index + 1,
                        }
                        for index in range(8)
                    ],
                    "questions": [
                        "Which time is intended?",
                        "Which place is intended?",
                        "Which category is intended?",
                        "Which comparison is intended?",
                    ],
                }
            )
            for _ in messages
        ]

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
            "adapter_prompt_tokens": self.requests * 10,
            "adapter_completion_tokens": self.requests * 10,
        }


def test_exact_64_parent_bank_and_independent_replay(tmp_path) -> None:
    adapter = _ParentAdapter()
    output = tmp_path / "parents"

    result = confirmation.build_parent_bank(
        output_dir=output,
        adapter=adapter,
        daily_budget_status={"authorized": True},
    )

    assert result["status"] == "passed"
    assert result["mechanics_gates"]["all_pass"] is True
    assert adapter.requests == 64
    assert adapter.seeds == [confirmation.PARENT_SEED_START + i for i in range(64)]
    replay = verifier.verify_parent_bank(output)
    assert replay["status"] == "verified"
    assert replay["mismatches"] == []
    assert replay["model_calls"] == 0


def test_confirmation_seed_ranges_are_disjoint_from_development() -> None:
    confirmation_seeds = confirmation.expected_seed_sets()
    flattened = [seed for values in confirmation_seeds.values() for seed in values]
    assert len(flattened) == len(set(flattened))
    assert min(flattened) > 202608400063


def test_confirmation_scope_restores_module_state() -> None:
    original_loader = confirmation.primary.load_stage_cigs
    original_annotation = confirmation.experiment.DEVELOPMENT_ANNOTATION_SEED_START
    original_choose = confirmation.experiment.scorer.choose_roots

    with confirmation.confirmation_scope():
        assert confirmation.experiment.DEVELOPMENT_ANNOTATION_SEED_START == (
            confirmation.ANNOTATION_SEED_START
        )
        assert [cig.cig_id for cig in confirmation.primary.load_stage_cigs("development")] == [
            cig.cig_id for cig in confirmation.load_confirmation_cigs()
        ]

    assert confirmation.primary.load_stage_cigs is original_loader
    assert confirmation.experiment.DEVELOPMENT_ANNOTATION_SEED_START == original_annotation
    assert confirmation.experiment.scorer.choose_roots is original_choose


def test_development_authorization_requires_frozen_report_tier() -> None:
    valid = {
        "status": "authorized",
        "development_status": "passed",
        "independent_replay_status": "verified",
        "report_tier": "smc_provisional_development_signal_confirmation_required",
    }
    confirmation.validate_development_authorization(valid)

    invalid = dict(valid)
    invalid["report_tier"] = "smc_development_policy_null_confirmation_forbidden"
    try:
        confirmation.validate_development_authorization(invalid)
    except ValueError as exc:
        assert "literal verified development pass" in str(exc)
    else:
        raise AssertionError("null development tier opened confirmation")


def test_exact_scale_confirmation_replays_on_untouched_scope(
    tmp_path, monkeypatch
) -> None:
    helpers = _experiment_test_helpers()
    parent_dir = tmp_path / "parents"
    with confirmation.confirmation_scope():
        info = helpers._install_primary_stage(parent_dir, "development")
    for task_id, row in info.items():
        row["transition_question"] = f"Which new follow-up applies to {task_id}?"

    raw_path = parent_dir / "private/RAW_RESPONSES.json"
    raw = json.loads(raw_path.read_text())
    raw["seeds"] = [confirmation.PARENT_SEED_START + i for i in range(64)]
    raw_path.write_text(json.dumps(raw))
    controls_path = parent_dir / "private/CONTROLS.json"
    controls = json.loads(controls_path.read_text())
    controls["roots"] = [
        {
            "task_id": row["task_id"],
            "question": row["question"],
            "raw_parent_sha256": hashlib.sha256(value.encode()).hexdigest(),
        }
        for row, value in zip(controls["roots"], raw["root"], strict=True)
    ]
    controls_path.write_text(json.dumps(controls))
    parent_result = {
        "interface_version": confirmation.INTERFACE_VERSION,
        "kind": "smc_confirmation_parent_bank",
        "status": "passed",
        "authorizes": "smc_confirmation_policy_only",
        "protocol": {
            "protocol_sha256": confirmation.PROTOCOL_SHA256,
            "model": confirmation.primary.MODEL_ID,
            "reasoning": "disabled_excluded",
            "expected_requests": 64,
        },
        "usage": {
            "adapter_requests": 64,
            "http_attempts": 64,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "run_cost_usd": 0.0,
        },
        "mechanics_gates": {"all_pass": True},
        "task_ids": list(info),
    }
    (parent_dir / "RESULT.json").write_text(json.dumps(parent_result))

    def fake_truth(cig, seed):
        row = info[cig.cig_id]
        return row["truth_index"], row["truth"]

    def fake_map(cig, question, truth):
        row = info[cig.cig_id]
        if question == row["transition_question"]:
            return {
                "supported": True,
                "facet": "followup",
                "confidence": 1.0,
                "method": "test",
                "answer": row["second_reply"],
            }
        root = row["questions"].index(question)
        answers = [
            row["first_reply"],
            row["second_reply"],
            "category 0",
            "constant comparison",
        ]
        return {
            "supported": True,
            "facet": f"root-{root}",
            "confidence": 1.0,
            "method": "test",
            "answer": answers[root],
        }

    monkeypatch.setattr(confirmation.primary, "sample_truth", fake_truth)
    monkeypatch.setattr(confirmation.primary, "map_and_answer", fake_map)
    monkeypatch.setattr(verifier.independent, "_truth", fake_truth)
    monkeypatch.setattr(verifier.independent, "_map", fake_map)
    adapter = helpers._Adapter(info)
    output = tmp_path / "confirmation"
    authorization = {
        "status": "authorized",
        "development_status": "passed",
        "independent_replay_status": "verified",
        "report_tier": "smc_provisional_development_signal_confirmation_required",
    }

    result = confirmation.run_confirmation(
        output_dir=output,
        parent_dir=parent_dir,
        adapter=adapter,
        development_authorization=authorization,
        bootstrap_samples=100,
    )

    assert result["mechanics_gates"]["all_pass"] is True
    assert adapter.requests == result["usage"]["deepseek_primary"][
        "adapter_requests"
    ]
    assert adapter.seeds[:64] == [
        confirmation.ANNOTATION_SEED_START + i for i in range(64)
    ]
    replay = verifier.verify(output, parent_dir=parent_dir)
    assert replay["status"] == "verified"
    assert replay["mismatches"] == []
    assert replay["model_calls"] == 0
    (output / "VERIFICATION.json").write_text(json.dumps(replay))
    written = report.write_report(output, parent_dir=parent_dir)
    frozen = report.build_report(output, parent_dir=parent_dir)
    assert frozen["claim_tier"] == report.CLAIMS[result["status"]][0]
    assert frozen["development_and_confirmation_are_not_pooled"] is True
    assert written["model_calls"] == 0
