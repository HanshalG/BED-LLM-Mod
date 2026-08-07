from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import regretbench_deepseek_smc_dynamic_depth2_experiment as experiment
from scripts import regretbench_deepseek_smc_frozen_report as smc_report
from scripts import regretbench_deepseek_smc_paper_fragment as smc_fragment
from scripts import regretbench_deepseek_smc_dynamic_depth2_policy as core
from scripts import regretbench_deepseek_smc_dynamic_depth2_verify as verifier


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
            "truth_index": truth_index,
            "truth": truth,
            "first_question": first_question,
            "first_reply": first_mapping["answer"],
            "second_question": second_question,
            "transition_question": second_question,
            "second_reply": second_mapping["answer"],
            "truth_answer": truth_answer,
            "questions": questions,
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
            task["transition_question"],
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


class _NaiveAdapter:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_seeded_messages_batched_structured(
        self,
        messages,
        seeds,
        *,
        temperature,
        response_format,
        max_new_tokens,
    ):
        self.requests += len(messages)
        responses = []
        for request in messages:
            payload = json.loads(request[1]["content"])
            label = "second" if payload["dialogue"] else "first"
            responses.append(
                json.dumps(
                    {"question": f"Naive {label} question for {payload['task_id']}?"}
                )
            )
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": self.requests * 5,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
            "adapter_prompt_tokens": self.requests * 10,
            "adapter_completion_tokens": self.requests * 10,
        }


def test_exact_ten_smc_enriched_smoke(tmp_path, monkeypatch) -> None:
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
    verification = verifier.verify_smoke(
        tmp_path / "smoke-output", primary_dir=primary_dir
    )
    assert verification["status"] == "verified"
    assert verification["mismatches"] == []
    assert verification["model_calls"] == 0
    smoke_result_path = tmp_path / "smoke-output/RESULT.json"
    stored_smoke_result = smoke_result_path.read_text()
    tampered_smoke_result = json.loads(stored_smoke_result)
    tampered_smoke_result["gates"]["exact_ten_requests"] = False
    smoke_result_path.write_text(json.dumps(tampered_smoke_result))
    tampered_smoke_verification = verifier.verify_smoke(
        tmp_path / "smoke-output", primary_dir=primary_dir
    )
    assert tampered_smoke_verification["status"] == "verification_failed"
    assert "$.gates.exact_ten_requests" in tampered_smoke_verification[
        "mismatches"
    ]
    smoke_result_path.write_text(stored_smoke_result)

    def fake_naive_map(cig, question, truth):
        label = "first" if "first" in question else "second"
        return {
            "supported": True,
            "facet": f"naive-{label}",
            "confidence": 1.0,
            "method": "test",
            "answer": info[cig.cig_id][f"{label}_reply"],
        }

    monkeypatch.setattr(experiment.primary, "map_and_answer", fake_naive_map)
    naive = experiment.run_naive_smoke(
        output_dir=tmp_path / "naive-smoke-output",
        adapter=_NaiveAdapter(),
        policy_smoke_result=tmp_path / "smoke-output/RESULT.json",
        daily_budget_status={"authorized": True},
    )
    assert naive["status"] == "passed"
    assert naive["gates"]["all_pass"] is True
    assert naive["usage"]["adapter_requests"] == 10


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


def test_realized_primary_execution_uses_frozen_roots_and_updated_parents(
    tmp_path, monkeypatch
) -> None:
    primary_dir = tmp_path / "primary-development"
    info = _install_primary_stage(primary_dir, "development")
    for task_id, row in info.items():
        row["transition_question"] = f"Which new follow-up applies to {task_id}?"
    adapter = _Adapter(info)
    output_dir = tmp_path / "planning-output"
    tree = experiment.build_development_planning_tree(
        output_dir=output_dir,
        adapter=adapter,
        primary_development_dir=primary_dir,
    )

    def fake_truth(cig, seed):
        row = info[cig.cig_id]
        return row["truth_index"], row["truth"]

    def fake_map(cig, question, truth):
        row = info[cig.cig_id]
        if question.startswith("Naive first"):
            return {
                "supported": True,
                "facet": "naive-first",
                "confidence": 1.0,
                "method": "test",
                "answer": row["first_reply"],
            }
        if question.startswith("Naive second"):
            return {
                "supported": True,
                "facet": "naive-second",
                "confidence": 1.0,
                "method": "test",
                "answer": row["second_reply"],
            }
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

    monkeypatch.setattr(experiment.primary, "sample_truth", fake_truth)
    monkeypatch.setattr(experiment.primary, "map_and_answer", fake_map)
    # The synthetic fixture intentionally reuses its precomputed truth rather
    # than the frozen SMC truth seed. Keep the independent verifier on that
    # same test oracle without coupling it to the producer implementation.
    monkeypatch.setattr(verifier.base, "_truth", fake_truth)
    monkeypatch.setattr(verifier.base, "_map", fake_map)

    realized = experiment.run_realized_primary(
        output_dir=output_dir,
        adapter=adapter,
        tree=tree,
        bootstrap_samples=100,
    )

    assert realized["mechanics_gates"]["all_pass"] is True
    assert realized["expected_primary_requests"] <= 8_768
    assert realized["usage"]["adapter_requests"] == realized[
        "expected_primary_requests"
    ]
    assert len(realized["tasks"]) == 64
    assert all(
        metrics["posterior_parent_update_applied"] is True
        for task in realized["tasks"]
        for metrics in task["policies"].values()
    )
    assert all(
        metrics["valid_two_action_trajectory"] is True
        and metrics["second_reply_likelihood_matched"] is True
        for task in realized["tasks"]
        for metrics in task["policies"].values()
    )
    raw = json.loads(
        (output_dir / "private/RAW_ACTUAL_PRIMARY.json").read_text()
    )
    assert len(raw["first_responses"]) == len(raw["final_responses"])
    assert all(
        row["seed"] == 202608340000 + row["task_index"]
        for row in raw["first_manifest"]
    )
    assert all(
        row["seed"] == 202608350000 + row["task_index"]
        for row in raw["final_manifest"]
    )

    naive = experiment.run_naive_baseline(
        output_dir=output_dir,
        contexts=tree["contexts"],
        initial_supports=tree["initial_supports"],
        naive_adapter=_NaiveAdapter(),
        endpoint_adapter=_Adapter(info),
    )
    assert naive["status"] == "available"
    assert len(naive["rows"]) == 64
    assert naive["luna_usage"]["adapter_requests"] == 128
    assert naive["endpoint_usage"]["adapter_requests"] == 128
    assert naive["diagnostics"]["all_descriptive_diagnostics_pass"] is True
    assert all(row["valid_two_action_trajectory"] for row in naive["rows"])

    final = experiment.finalize_development_result(
        output_dir=output_dir,
        primary_result=realized,
        policy_smoke={"sha256": "policy-smoke"},
        naive_smoke={"sha256": "naive-smoke"},
        naive_result=naive,
        primary_privacy=[*tree["privacy"], *realized["actual_privacy"]],
        naive_privacy=naive["naive_privacy"],
        endpoint_privacy=naive["endpoint_privacy"],
        daily_budget_status={"authorized": True},
        bootstrap_samples=100,
    )
    assert final["mechanics_gates"]["all_pass"] is True
    assert final["usage"]["combined_requests"] <= 9_024
    assert final["usage"]["deepseek_naive_endpoint"]["adapter_requests"] == 128
    assert final["usage"]["naive_luna"]["adapter_requests"] == 128
    assert all("naive_thinking" in task["policies"] for task in final["tasks"])
    assert final["science"]["comparisons"]["smc_myopic_refresh_brier"][
        "brier_dynamic_minus_baseline"
    ]["seed"] == 202608360200
    assert json.loads((output_dir / "RESULT.json").read_text())["status"] == final[
        "status"
    ]
    privacy = json.loads((output_dir / "private/PRIVACY.json").read_text())
    assert len(privacy["primary"]) == realized["expected_primary_requests"]
    assert len(privacy["naive"]) == 128
    assert len(privacy["naive_endpoint"]) == 128
    verification = verifier.verify(output_dir, primary_dir=primary_dir)
    assert verification["status"] == "verified"
    assert verification["mismatches"] == []
    assert verification["model_calls"] == 0
    (output_dir / "VERIFICATION.json").write_text(json.dumps(verification))
    report_written = smc_report.write_report(
        output_dir, primary_dir=primary_dir
    )
    frozen_report = smc_report.build_report(output_dir, primary_dir=primary_dir)
    assert frozen_report["claim_tier"] == smc_report.CLAIMS[final["status"]][0]
    assert frozen_report["pooled_or_secondary_evidence_can_change_tier"] is False
    assert report_written["model_calls"] == 0
    fragment_written = smc_fragment.write_fragment(
        output_dir,
        primary_dir=primary_dir,
        output=tmp_path / "smc-result.tex",
    )
    assert fragment_written["claim_tier"] == frozen_report["claim_tier"]
    assert fragment_written["model_calls"] == 0

    result_path = output_dir / "RESULT.json"
    stored_result = result_path.read_text()
    tampered_result = json.loads(stored_result)
    first_policy = next(iter(tampered_result["tasks"][0]["policies"]))
    tampered_result["tasks"][0]["policies"][first_policy]["brier"] += 0.1
    result_path.write_text(json.dumps(tampered_result))
    tampered_verification = verifier.verify(output_dir, primary_dir=primary_dir)
    assert tampered_verification["status"] == "verification_failed"
    assert any(
        path.startswith("$.tasks")
        for path in tampered_verification["mismatches"]
    )
    with pytest.raises(ValueError, match="independently verified"):
        smc_report.build_report(output_dir, primary_dir=primary_dir)
    result_path.write_text(stored_result)

    branches_path = output_dir / "private/RAW_BRANCHES.json"
    stored_branches = branches_path.read_text()
    tampered_branches = json.loads(stored_branches)
    first_branch = json.loads(tampered_branches["responses"][0])
    first_branch["retained_parent_indexes"] = [99]
    tampered_branches["responses"][0] = json.dumps(first_branch)
    branches_path.write_text(json.dumps(tampered_branches))
    with pytest.raises(ValueError):
        verifier.verify(output_dir, primary_dir=primary_dir)
    branches_path.write_text(stored_branches)

    without_naive = experiment.finalize_development_result(
        output_dir=tmp_path / "without-naive",
        primary_result=realized,
        policy_smoke={"sha256": "policy-smoke"},
        naive_smoke={"status": "failed"},
        naive_result=None,
        naive_error={"error": "baseline unavailable"},
        naive_usage_on_error={
            **experiment._empty_usage(),
            "adapter_requests": 3,
            "http_attempts": 3,
            "run_cost_usd": 0.01,
        },
        endpoint_usage_on_error={
            **experiment._empty_usage(),
            "adapter_requests": 2,
            "http_attempts": 2,
            "run_cost_usd": 0.01,
        },
        bootstrap_samples=100,
    )
    assert without_naive["mechanics_gates"]["all_pass"] is True
    assert without_naive["status"] == final["status"]
    assert without_naive["naive_baseline"]["status"] == "unavailable"
    assert without_naive["usage"]["combined_requests"] == (
        realized["usage"]["adapter_requests"] + 5
    )
    assert all(
        "naive_thinking" not in task["policies"]
        for task in without_naive["tasks"]
    )

    frozen_path = output_dir / "private/FROZEN_SELECTIONS.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["hidden_truth_accessed"] = True
    frozen_path.write_text(json.dumps(frozen))
    requests_before_refusal = adapter.requests
    with pytest.raises(ValueError, match="not frozen before truth"):
        experiment.run_realized_primary(
            output_dir=output_dir,
            adapter=adapter,
            tree=tree,
            bootstrap_samples=100,
        )
    assert adapter.requests == requests_before_refusal
