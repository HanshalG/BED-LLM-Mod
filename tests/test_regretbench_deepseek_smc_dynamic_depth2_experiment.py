from __future__ import annotations

import json
from pathlib import Path

from scripts import regretbench_deepseek_smc_dynamic_depth2_experiment as experiment
from scripts import regretbench_deepseek_smc_dynamic_depth2_policy as core


def _question(text: str) -> str:
    return text.strip().rstrip("?") + "?"


def _install_primary_stage(path: Path, stage: str) -> dict[str, dict]:
    cigs = experiment.primary.load_stage_cigs(stage)
    info = {}
    roots = []
    controls = []
    truth_seed = experiment.primary.STAGES[stage]["truth_seed_start"]
    for index, cig in enumerate(cigs):
        truth_index, truth = experiment.primary.sample_truth(cig, truth_seed + index)
        candidates = []
        for reference in cig.reference_questions:
            question = _question(reference.text)
            mapping = experiment.primary.map_and_answer(cig, question, truth)
            if mapping["supported"] and all(
                mapping["facet"] != prior[1]["facet"] for prior in candidates
            ):
                candidates.append((question, mapping))
        if not candidates:
            raise AssertionError(f"test task has no supported question: {cig.cig_id}")
        first_question, first_mapping = candidates[0]
        second_question, second_mapping = (
            candidates[1] if len(candidates) > 1 else candidates[0]
        )
        aliases = str((truth.slots or {})["answer_aliases"])
        truth_answer = aliases.split("|")[0].strip()
        questions = [
            first_question,
            second_question,
            "Which category is intended?",
            "Which comparison is intended?",
        ]
        # Preserve uniqueness when a source question has the same normalized text
        # as one of the generic mechanics questions.
        for position in range(1, 4):
            if experiment.primary.normalize_text(questions[position]) in {
                experiment.primary.normalize_text(value)
                for value in questions[:position]
            }:
                questions[position] = f"Which alternate dimension {position} is intended?"
        root = {
            "hypotheses": [
                {
                    "interpretation": f"interpretation {particle} for {cig.cig_id}",
                    "final_answer": truth_answer
                    if particle == 0
                    else f"distractor answer {particle}",
                    "prior_weight": 1,
                }
                for particle in range(8)
            ],
            "questions": questions,
        }
        roots.append(json.dumps(root))
        controls.append(
            {
                "task_id": cig.cig_id,
                "truth_index": truth_index,
                "question": first_question,
                "mapping": first_mapping,
                "aliases": aliases,
            }
        )
        info[cig.cig_id] = {
            "first_question": first_question,
            "first_reply": first_mapping["answer"],
            "second_question": second_question,
            "second_reply": second_mapping["answer"],
            "truth_answer": truth_answer,
        }
    private = path / "private"
    private.mkdir(parents=True)
    (private / "RAW_RESPONSES.json").write_text(
        json.dumps({"stage": stage, "root": roots, "branches": []})
    )
    (private / "CONTROLS.json").write_text(
        json.dumps({"stage": stage, "roots": controls, "privacy": []})
    )
    return info


def _install_smc_pass(path: Path) -> tuple[Path, Path, Path, Path]:
    result_path = path / "RESULT.json"
    verification_path = path / "VERIFICATION.json"
    daily_path = path / "DAILY_RESULT.json"
    ledger_path = path / "LEDGER.json"
    path.mkdir(parents=True, exist_ok=True)
    result = {
        "interface_version": "regretbench-deepseek-smc-support-recovery-daily-1",
        "status": "passed",
        "authorizes": "separately_preregistered_smc_policy_only",
        "protocol": {
            "protocol_sha256": core.smc_support.PROTOCOL_SHA256,
            "smc_policy_endpoint_opened": False,
            "primary_policy_endpoint_opened": False,
            "primary_confirmation_opened": False,
        },
        "mechanics_gates": {"all_pass": True},
        "science": {"gates": {"all_pass": True}},
    }
    result_path.write_text(json.dumps(result))
    verification = {
        "status": "verified",
        "result_status": "passed",
        "mismatches": [],
        "model_calls": 0,
        "cost_usd": 0.0,
        "artifact_sha256": {"RESULT.json": core.sha256_file(result_path)},
    }
    verification_path.write_text(json.dumps(verification))
    ledger = {
        "date": "2026-08-09",
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "account_wide_usage_counts_against_cap": True,
        "stage": {"status": "passed"},
    }
    ledger_path.write_text(json.dumps(ledger))
    daily = {
        "status": "complete_reconciled",
        "development_status": "passed",
        "independent_replay_passed": True,
        "smc_policy_endpoint_opened": False,
        "primary_policy_endpoint_opened": False,
        "primary_confirmation_opened": False,
        "authorizes": "separately_preregistered_smc_policy_only",
        "result_sha256": core.sha256_file(result_path),
        "verification_sha256": core.sha256_file(verification_path),
        "ledger_sha256": core.sha256_file(ledger_path),
    }
    daily_path.write_text(json.dumps(daily))
    return result_path, verification_path, daily_path, ledger_path


class _Adapter:
    def __init__(self, info: dict[str, dict]) -> None:
        self.info = info
        self.requests = 0
        self.seeds = []
        self.schema_names = []

    def _annotation(self, payload: dict) -> str:
        task = self.info[payload["task_id"]]
        rows = []
        for parent in payload["parent_particles"]:
            index = parent["parent_index"]
            rows.append(
                {
                    "parent_index": index,
                    "predicted_replies": [
                        task["first_reply"] if index == 0 else f"first {index}",
                        task["second_reply"] if index == 0 else f"second {index}",
                        f"category {index % 2}",
                        "constant comparison",
                    ],
                }
            )
        return json.dumps({"particles": rows})

    def _transition(self, payload: dict) -> str:
        task = self.info[payload["task_id"]]
        rows = []
        for parent in payload["parent_particles"]:
            index = parent["parent_index"]
            retained = index < 2
            rows.append(
                {
                    "parent_index": index,
                    "revision_type": "retained" if retained else "revised",
                    "interpretation": parent["interpretation"]
                    if retained
                    else parent["interpretation"] + " revised",
                    "final_answer": parent["final_answer"],
                    "prior_weight": index + 1,
                    "predicted_replies": [
                        task["second_reply"] if index == 0 else f"followup {index}",
                        f"period {index % 2}",
                        "constant category",
                        "constant comparison",
                    ],
                }
            )
        questions = [
            task["second_question"],
            "Which period is intended?",
            "Which category is intended?",
            "Which comparison is intended?",
        ]
        for position in range(1, 4):
            if experiment.primary.normalize_text(questions[position]) in {
                experiment.primary.normalize_text(value)
                for value in questions[:position]
            }:
                questions[position] = f"Which follow-up dimension {position} is intended?"
        return json.dumps({"hypotheses": rows, "questions": questions})

    def chat_complete_seeded_messages_batched_structured(
        self,
        messages,
        seeds,
        *,
        temperature,
        response_format,
        max_new_tokens,
    ):
        name = response_format["json_schema"]["name"]
        self.requests += len(messages)
        self.seeds.extend(seeds)
        self.schema_names.extend([name] * len(messages))
        responses = []
        for request in messages:
            payload = json.loads(request[1]["content"])
            responses.append(
                self._annotation(payload)
                if name == "regretbench_smc_parent_annotation"
                else self._transition(payload)
            )
        return responses

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


def test_exact_ten_smc_enriched_smoke(tmp_path) -> None:
    primary_dir = tmp_path / "primary-smoke"
    info = _install_primary_stage(primary_dir, "smoke")
    predecessor = _install_smc_pass(tmp_path / "smc-pass")
    adapter = _Adapter(info)

    result = experiment.run_smoke(
        output_dir=tmp_path / "smoke-output",
        adapter=adapter,
        primary_smoke_dir=primary_dir,
        smc_result_path=predecessor[0],
        smc_verification_path=predecessor[1],
        smc_daily_result_path=predecessor[2],
        smc_ledger_path=predecessor[3],
        daily_budget_status={"authorized": True},
    )

    assert result["status"] == "passed"
    assert result["gates"]["all_pass"] is True
    assert adapter.requests == 10
    assert adapter.schema_names.count("regretbench_smc_parent_annotation") == 4
    assert adapter.schema_names.count("regretbench_smc_enriched_transition") == 6


def test_full_8256_planning_schedule_freezes_before_truth(tmp_path) -> None:
    primary_dir = tmp_path / "primary-development"
    info = _install_primary_stage(primary_dir, "development")
    adapter = _Adapter(info)

    tree = experiment.build_development_planning_tree(
        output_dir=tmp_path / "planning-output",
        adapter=adapter,
        primary_development_dir=primary_dir,
    )

    assert tree["gates"]["all_pass"] is True
    assert adapter.requests == 8_256
    assert len(tree["initial_supports"]) == 64
    assert sum(len(rows) for rows in tree["branch_rows"]) == 4_096
    assert len(tree["planning"]) == 64
    assert all(len(row["selected"]) == 7 for row in tree["planning"])
    assert adapter.seeds[64:72] == [
        202608310000,
        202608310000,
        202608310001,
        202608310001,
        202608310002,
        202608310002,
        202608310003,
        202608310003,
    ]
    frozen = json.loads(
        (tmp_path / "planning-output/private/FROZEN_SELECTIONS.json").read_text()
    )
    assert frozen["hidden_truth_accessed"] is False
    assert frozen["planning_requests"] == 8_256
