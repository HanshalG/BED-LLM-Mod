from __future__ import annotations

import json
import random

import pytest

from scripts import regretbench_deepseek_dynamic_depth2_policy as policy
from scripts import regretbench_deepseek_result_verify as verify
from scripts import regretbench_deepseek_support_recovery as recovery


def _support(*, truth_answer: str | None = None) -> dict:
    hypotheses = []
    for index in range(8):
        hypotheses.append(
            {
                "interpretation": f"interpretation {index}",
                "final_answer": truth_answer if index == 0 and truth_answer else f"answer {index}",
                "prior_weight": 1,
                "predicted_replies": [
                    f"binary {index % 2}",
                    f"unique {index}",
                    f"ternary {index % 3}",
                    f"binary other {index % 2}",
                ],
            }
        )
    return {
        "hypotheses": hypotheses,
        "questions": [
            "Which type do you mean?",
            "Which exact variant do you mean?",
            "Which context do you mean?",
            "Which period do you mean?",
        ],
    }


def test_parser_and_question_eig_are_exact() -> None:
    support = policy.parse_enriched_support(json.dumps(_support()))

    assert support["diagnostic"]["valid_unique_count"] == 8
    assert support["diagnostic"]["informative_question_count"] == 4
    assert policy.question_eig(support, 0) == pytest.approx(__import__("math").log(2))
    assert policy.question_eig(support, 1) == pytest.approx(__import__("math").log(8))
    assert policy.select_question(support) == 1
    assert sum(item["probability"] for item in support["hypotheses"]) == pytest.approx(1)


def test_protocol_binding_refuses_support_core_change(monkeypatch) -> None:
    monkeypatch.setattr(policy, "SUPPORT_RECOVERY_CORE_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="support-recovery core binding changed"):
        policy.validate_protocol_binding()


def test_protocol_binding_refuses_distinct_action_amendment_change(
    monkeypatch,
) -> None:
    monkeypatch.setattr(policy, "ACTION_NOVELTY_AMENDMENT_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="distinct-action amendment changed"):
        policy.validate_protocol_binding()


def test_distinct_actions_use_official_facet_identity() -> None:
    first = {"supported": True, "facet": "country"}

    assert policy.distinct_supported_actions(
        first, {"supported": True, "facet": "period"}
    )
    assert not policy.distinct_supported_actions(
        first, {"supported": True, "facet": "country"}
    )
    assert not policy.distinct_supported_actions(
        first, {"supported": False, "facet": None}
    )


def test_duplicate_enriched_particle_fails_instead_of_changing_width() -> None:
    payload = _support()
    payload["hypotheses"][1] = dict(payload["hypotheses"][0])

    with pytest.raises(ValueError, match="duplicate hypotheses"):
        policy.parse_enriched_support(json.dumps(payload))


def test_branch_truth_metric_rewards_recoverable_truth() -> None:
    recovered = policy.parse_enriched_support(
        json.dumps(_support(truth_answer="Los Lagos Region of Chile"))
    )
    missing = policy.parse_enriched_support(json.dumps(_support()))

    recovered_metric = policy.branch_truth_metrics(
        recovered, "Los Lagos Region of Chile"
    )
    missing_metric = policy.branch_truth_metrics(
        missing, "Los Lagos Region of Chile"
    )

    assert recovered_metric["truth_mass"] == pytest.approx(1 / 8)
    assert recovered_metric["expected_brier"] == 0.0
    assert missing_metric["truth_mass"] == 0.0
    assert missing_metric["expected_brier"] == 1.0


def test_realized_terminal_metric_conditions_generated_likelihoods() -> None:
    support = policy.parse_enriched_support(json.dumps(_support()))

    matched = policy.realized_terminal_metrics(
        support,
        question_index=0,
        observed_reply="binary 0",
        aliases="answer 0",
    )
    missing = policy.realized_terminal_metrics(
        support,
        question_index=0,
        observed_reply="not represented",
        aliases="answer 0",
    )

    assert matched["reply_matched"] is True
    assert matched["matched_hypothesis_count"] == 4
    assert matched["truth_mass"] == pytest.approx(0.25)
    assert matched["brier"] == pytest.approx(0.75**2)
    assert missing["reply_matched"] is False
    assert missing["truth_mass"] == 0.0
    assert missing["brier"] == 1.0


def test_root_level_crn_removes_seed_only_candidate_advantage() -> None:
    legacy_risks = []
    crn_risks = []
    for root in range(policy.QUESTIONS):
        legacy = [
            random.Random(
                policy.BRANCH_SEED_START
                + root * 16
                + hypothesis * 2
                + draw
            ).random()
            for hypothesis in range(policy.HYPOTHESES)
            for draw in range(policy.BRANCH_DRAWS)
        ]
        shared = [
            random.Random(
                policy.branch_seed(0, hypothesis, draw)
            ).random()
            for hypothesis in range(policy.HYPOTHESES)
            for draw in range(policy.BRANCH_DRAWS)
        ]
        legacy_risks.append(sum(legacy) / len(legacy))
        crn_risks.append(sum(shared) / len(shared))

    assert max(legacy_risks) - min(legacy_risks) > 0.10
    assert len(set(crn_risks)) == 1
    assert policy.actual_first_seed(7) == policy.ACTUAL_FIRST_SEED_START + 7
    assert policy.actual_final_seed(7) == policy.ACTUAL_FINAL_SEED_START + 7


def _science_task(index: int) -> dict:
    predicted_gain = 0.02 + index / 10_000
    realized_gain = 0.08 + index / 5_000
    dynamic_brier = 0.10
    roots = {
        "dynamic_depth2": 0,
        "myopic_width": 1,
        "history_blind_depth2": 2,
        "fixed_depth2": 3,
        "random": index % 4,
    }
    policies = {
        "dynamic_depth2": {"brier": dynamic_brier, "log_loss": 0.10},
        "myopic_width": {
            "brier": dynamic_brier + realized_gain,
            "log_loss": 0.30,
        },
        "history_blind_depth2": {"brier": 0.20, "log_loss": 0.25},
        "fixed_depth2": {"brier": 0.18, "log_loss": 0.22},
        "random": {"brier": 0.25, "log_loss": 0.30},
        "naive_thinking": {"brier": 0.16, "log_loss": 0.20},
    }
    for values in policies.values():
        values["fresh_brier"] = values["brier"]
        values["fresh_log_loss"] = values["log_loss"]
    return {
        "selected_roots": roots,
        "conditioned_root_risks": [
            {"brier": 0.10},
            {"brier": 0.10 + predicted_gain},
            {"brier": 0.20},
            {"brier": 0.22},
        ],
        "policies": policies,
    }


def test_scientific_summary_enforces_full_conjunctive_claim() -> None:
    summary = policy.scientific_summary(
        [_science_task(index) for index in range(64)], samples=300
    )

    assert summary["gates"]["all_pass"] is True
    assert summary["root_disagreements"]["myopic_width"] == 64
    assert summary["comparisons"]["myopic_width"]["wins_ties_losses"] == {
        "wins": 64,
        "ties": 0,
        "losses": 0,
    }
    assert "naive_thinking" not in summary["comparisons"]
    assert (
        "naive_thinking"
        in summary["fresh_regeneration_comparisons_descriptive"]
    )
    assert summary["predicted_to_realized_dynamic_myopic"]["spearman"] == pytest.approx(1)


def test_fresh_regeneration_endpoint_is_descriptive_only() -> None:
    tasks = [_science_task(index) for index in range(64)]
    for task in tasks:
        task["policies"]["dynamic_depth2"]["fresh_brier"] = 0.9
        task["policies"]["dynamic_depth2"]["fresh_log_loss"] = 2.0
        for baseline in (
            "myopic_width",
            "history_blind_depth2",
            "fixed_depth2",
            "random",
            "naive_thinking",
        ):
            task["policies"][baseline]["fresh_brier"] = 0.1
            task["policies"][baseline]["fresh_log_loss"] = 0.1

    summary = policy.scientific_summary(tasks, samples=300)

    assert summary["gates"]["all_pass"] is True
    assert summary["fresh_regeneration_comparisons_descriptive"][
        "myopic_width"
    ]["brier_dynamic_minus_baseline"]["mean"] == pytest.approx(0.8)


class _FixtureAdapter:
    WORDS = ("zero", "one", "two", "three")

    def __init__(self) -> None:
        self.requests = 0
        all_cigs = [
            *recovery.load_stage_cigs("smoke"),
            *recovery.load_stage_cigs("development"),
        ]
        self.facets = {
            cig.cig_id: [facet.replace("_", " ") for facet in cig.semantic_facets]
            for cig in all_cigs
        }
        self.truth_aliases = {}
        self.truth_replies = {}
        for index, cig in enumerate(recovery.load_stage_cigs("smoke")):
            _, truth = recovery.sample_truth(
                cig, recovery.STAGES["smoke"]["truth_seed_start"] + index
            )
            self.truth_replies[cig.cig_id] = {
                facet.replace("_", " "): str(
                    (truth.slots or {}).get(facet, "")
                )
                for facet in cig.semantic_facets
            }
        for index, cig in enumerate(recovery.load_stage_cigs("development")):
            _, truth = recovery.sample_truth(cig, policy.TRUTH_SEED_START + index)
            self.truth_aliases[cig.cig_id] = str(
                (truth.slots or {})["answer_aliases"]
            ).split("|")[0]
            self.truth_replies[cig.cig_id] = {
                facet.replace("_", " "): str(
                    (truth.slots or {}).get(facet, "")
                )
                for facet in cig.semantic_facets
            }

    def _root_from_question(self, text: str) -> int:
        for index, word in enumerate(self.WORDS):
            if f"option {word}" in text.casefold():
                return index
        return 0

    def _response(
        self, payload: dict, seed: int, *, branch_root: int | None = None
    ) -> str:
        task_id = payload["task_id"]
        dialogue = payload["dialogue"]
        root = self._root_from_question(dialogue[0]["content"]) if dialogue else None
        branch_formal = policy.BRANCH_SEED_START <= seed < policy.ACTUAL_FIRST_SEED_START
        answers = [f"fixture answer {index}" for index in range(8)]
        weights = [1] * 8
        if branch_formal:
            local = seed - policy.BRANCH_SEED_START
            hypothesis = (local % 16) // 2
            conditioned = bool(dialogue)
            if (conditioned and branch_root == 0) or (
                not conditioned and branch_root == 2
            ):
                answers[0] = f"candidate answer {hypothesis}"
                weights[0] = 20
        elif seed >= policy.ACTUAL_FIRST_SEED_START and dialogue:
            if root == 0:
                answers[0] = self.truth_aliases[task_id]
                weights[0] = 100
        elif dialogue and str(dialogue[-1]["content"]).startswith("sim-q0-h"):
            hypothesis = int(str(dialogue[-1]["content"]).rsplit("h", 1)[1])
            answers[0] = f"candidate answer {hypothesis}"
            weights[0] = 20
        elif not dialogue:
            answers = [f"candidate answer {index}" for index in range(8)]

        facets = self.facets[task_id]
        question_facets = [facets[index % len(facets)] for index in range(4)]
        if root is not None:
            first_facet = facets[root % len(facets)]
            next_facet = facets[(root + 1) % len(facets)]
            question_facets[1] = (
                next_facet if next_facet != first_facet else facets[-1]
            )
        questions = [
            f"Which {question_facets[index]} do you mean for option {word}?"
            for index, word in enumerate(self.WORDS)
        ]
        hypotheses = []
        for index in range(8):
            predicted_replies = [
                f"sim-q0-h{index % 2}",
                f"sim-q1-h{index}",
                f"sim-q2-h{index % 2}",
                f"sim-q3-h{index % 3}",
            ]
            if index == 0:
                predicted_replies = [
                    self.truth_replies[task_id][facet]
                    for facet in question_facets
                ]
            hypotheses.append(
                {
                    "interpretation": f"fixture interpretation {index}",
                    "final_answer": answers[index],
                    "prior_weight": weights[index],
                    "predicted_replies": predicted_replies,
                }
            )
        return json.dumps({"hypotheses": hypotheses, "questions": questions})

    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages, seeds, **kwargs
    ):
        assert kwargs["response_format"] == policy.enriched_response_format()
        responses = []
        branch_batch = len(batch_messages) == policy.PLANNING_REQUESTS - 64
        for batch_index, (messages, seed) in enumerate(
            zip(batch_messages, seeds, strict=True)
        ):
            payload = json.loads(messages[-1]["content"])
            branch_root = (
                ((batch_index // 2) % 64) // 16 if branch_batch else None
            )
            responses.append(
                self._response(payload, seed, branch_root=branch_root)
            )
            self.requests += 1
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
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
        }


class _FakeNaiveAdapter:
    def __init__(self) -> None:
        self.requests = 0
        self.facets = {
            cig.cig_id: [facet.replace("_", " ") for facet in cig.semantic_facets]
            for cig in [
                *recovery.load_stage_cigs("smoke"),
                *recovery.load_stage_cigs("development"),
            ]
        }

    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages, seeds, **kwargs
    ):
        assert kwargs["response_format"] == policy.naive_response_format()
        responses = []
        for messages, _seed in zip(batch_messages, seeds, strict=True):
            payload = json.loads(messages[-1]["content"])
            facet_index = 0 if not payload["dialogue"] else 1
            facets = self.facets[payload["task_id"]]
            facet = facets[min(facet_index, len(facets) - 1)]
            responses.append(json.dumps({"question": f"Which {facet} do you mean?"}))
            self.requests += 1
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": self.requests,
            "forced_exits": 0,
            "forced_final_requests": 0,
            "forced_final_successes": 0,
            "adapter_cost_usd": 0.0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
        }


class _RepeatingFacetAdapter(_FixtureAdapter):
    def _response(
        self, payload: dict, seed: int, *, branch_root: int | None = None
    ) -> str:
        value = json.loads(
            super()._response(payload, seed, branch_root=branch_root)
        )
        task_id = payload["task_id"]
        facet = self.facets[task_id][0]
        value["questions"] = [
            f"Which {facet} do you mean for repeated option {word}?"
            for word in self.WORDS
        ]
        truth_reply = self.truth_replies[task_id][facet]
        value["hypotheses"][0]["predicted_replies"] = [truth_reply] * 4
        return json.dumps(value)


def test_exact_ten_enriched_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        policy,
        "validate_support_predecessors",
        lambda **kwargs: {"support": "fixture"},
    )
    adapter = _FixtureAdapter()

    result = policy.run_smoke(
        output_dir=tmp_path / "smoke",
        run_id="fixture-policy-smoke",
        support_smoke_result=tmp_path / "support-smoke.json",
        support_development_result=tmp_path / "support-development.json",
        adapter=adapter,
    )

    assert result["status"] == "passed"
    assert all(result["gates"].values())
    assert adapter.requests == 10
    assert result["protocol"]["policy_endpoint_opened"] is False
    replay = verify.verify_policy_smoke(tmp_path / "smoke")
    assert replay["status"] == "verified"


def test_enriched_smoke_rejects_repeated_semantic_action(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        policy,
        "validate_support_predecessors",
        lambda **kwargs: {"support": "fixture"},
    )

    result = policy.run_smoke(
        output_dir=tmp_path / "repeated-smoke",
        run_id="fixture-repeated-policy-smoke",
        support_smoke_result=tmp_path / "support-smoke.json",
        support_development_result=tmp_path / "support-development.json",
        adapter=_RepeatingFacetAdapter(),
    )

    assert result["status"] == "mechanics_failed"
    assert result["gates"]["all_three_second_questions_supported"] is True
    assert result["gates"][
        "all_three_exact_second_replies_match_generated_likelihoods"
    ] is True
    assert result["gates"]["all_three_second_actions_are_novel"] is False
    replay = verify.verify_policy_smoke(tmp_path / "repeated-smoke")
    assert replay["status"] == "verified"

    result_path = tmp_path / "repeated-smoke" / "RESULT.json"
    tampered = json.loads(result_path.read_text())
    tampered["gates"]["all_three_second_actions_are_novel"] = True
    result_path.write_text(json.dumps(tampered))
    failed = verify.verify_policy_smoke(tmp_path / "repeated-smoke")
    assert failed["status"] == "verification_failed"
    assert "$.gates.all_three_second_actions_are_novel" in failed[
        "mismatches"
    ]


def test_exact_ten_naive_thinking_smoke(tmp_path, monkeypatch) -> None:
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
    adapter = _FakeNaiveAdapter()

    result = policy.run_naive_smoke(
        output_dir=tmp_path / "naive-smoke",
        run_id="fixture-naive-smoke",
        support_smoke_result=tmp_path / "support-smoke.json",
        support_development_result=tmp_path / "support-development.json",
        policy_smoke_result=tmp_path / "policy-smoke.json",
        adapter=adapter,
    )

    assert result["status"] == "passed"
    assert all(result["gates"].values())
    assert adapter.requests == 10
    assert result["usage"]["adapter_reasoning_tokens"] == 10
    assert result["gates"]["all_four_second_actions_are_novel"] is True


def test_full_8256_planning_response_path_and_actual_cache(
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
    monkeypatch.setattr(
        policy,
        "validate_naive_smoke",
        lambda path: {"path": str(path), "sha256": "fixture"},
    )
    adapter = _FixtureAdapter()
    naive_endpoint_adapter = _FixtureAdapter()
    naive_adapter = _FakeNaiveAdapter()

    result = policy.run_development(
        output_dir=tmp_path / "development",
        run_id="fixture-policy-development",
        support_smoke_result=tmp_path / "support-smoke.json",
        support_development_result=tmp_path / "support-development.json",
        policy_smoke_result=tmp_path / "policy-smoke.json",
        naive_smoke_result=tmp_path / "naive-smoke.json",
        adapter=adapter,
        naive_adapter=naive_adapter,
        naive_endpoint_adapter=naive_endpoint_adapter,
        bootstrap_samples=50,
    )

    assert result["protocol"]["planning_requests"] == 8_256
    assert result["protocol"]["expected_primary_deepseek_requests"] == adapter.requests
    assert 8_256 < adapter.requests <= 8_768
    assert naive_endpoint_adapter.requests == 128
    assert result["protocol"]["actual_naive_requests"] == naive_adapter.requests == 128
    assert result["protocol"]["actual_combined_requests"] == (
        adapter.requests + naive_endpoint_adapter.requests + naive_adapter.requests
    )
    assert all(result["mechanics_gates"].values())
    assert result["mechanics_gates"][
        "every_policy_has_40_novel_second_actions"
    ] is True
    assert result["naive_baseline"]["status"] == "available"
    assert result["naive_baseline"]["all_transport_and_schema_gates_pass"] is True
    assert all(
        task["policies"]["dynamic_depth2"]["endpoint_mode"]
        == "aligned_generated_likelihood"
        for task in result["tasks"]
    )
    assert all(
        "fresh_brier" in task["policies"]["dynamic_depth2"]
        for task in result["tasks"]
    )
    assert result["protocol"]["fresh_regeneration_endpoint"] == (
        "secondary_descriptive"
    )
    assert result["status"] in {"passed", "gated_null"}
    assert len(result["tasks"]) == 64
    assert all("naive_thinking" in task["policies"] for task in result["tasks"])
    assert result["protocol"]["selection_frozen_before_truth_access"] is True
    public = (tmp_path / "development" / "RESULT.json").read_text()
    assert "aliases" not in public
    assert "selected_questions" not in public
    assert (tmp_path / "development" / "private" / "FROZEN_SELECTIONS.json").exists()
    replay = verify.verify_policy(tmp_path / "development")
    assert replay["status"] == "verified"
    assert replay["model_calls"] == 0

    result_path = tmp_path / "development" / "RESULT.json"
    tampered = json.loads(result_path.read_text())
    tampered["tasks"][0]["policies"]["dynamic_depth2"]["brier"] += 0.01
    result_path.write_text(json.dumps(tampered))
    failed = verify.verify_policy(tmp_path / "development")
    assert failed["status"] == "verification_failed"
    assert "$.tasks[0].policies.dynamic_depth2.brier" in failed["mismatches"]


def test_formal_naive_failure_cannot_veto_primary_result(
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
    monkeypatch.setattr(
        policy,
        "validate_naive_smoke",
        lambda path: {"path": str(path), "sha256": "fixture"},
    )
    primary_adapter = _FixtureAdapter()
    endpoint_adapter = _FixtureAdapter()
    naive_adapter = _FakeNaiveAdapter()

    def fail_naive(*args, **kwargs):
        raise ValueError("deliberate descriptive baseline failure")

    monkeypatch.setattr(policy, "_call_naive", fail_naive)
    result = policy.run_development(
        output_dir=tmp_path / "development-naive-failure",
        run_id="fixture-policy-naive-failure",
        support_smoke_result=tmp_path / "support-smoke.json",
        support_development_result=tmp_path / "support-development.json",
        policy_smoke_result=tmp_path / "policy-smoke.json",
        naive_smoke_result=tmp_path / "naive-smoke.json",
        adapter=primary_adapter,
        naive_adapter=naive_adapter,
        naive_endpoint_adapter=endpoint_adapter,
        bootstrap_samples=50,
    )

    assert result["status"] != "mechanics_failed"
    assert result["science"] is not None
    assert all(result["mechanics_gates"].values())
    assert result["naive_baseline"]["status"] == "failed_closed"
    assert result["naive_baseline"]["can_affect_primary_status"] is False
    assert all("naive_thinking" not in task["policies"] for task in result["tasks"])
    assert endpoint_adapter.requests == 0


def test_disabled_naive_baseline_keeps_primary_exact_path(
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
    primary_adapter = _FixtureAdapter()

    result = policy.run_development(
        output_dir=tmp_path / "development-naive-disabled",
        run_id="fixture-policy-naive-disabled",
        support_smoke_result=tmp_path / "support-smoke.json",
        support_development_result=tmp_path / "support-development.json",
        policy_smoke_result=tmp_path / "policy-smoke.json",
        naive_smoke_result=tmp_path / "naive-smoke-failure.json",
        adapter=primary_adapter,
        naive_baseline_enabled=False,
        bootstrap_samples=50,
    )

    assert result["status"] != "mechanics_failed"
    assert result["science"] is not None
    assert all(result["mechanics_gates"].values())
    assert result["naive_baseline"]["status"] == "disabled_by_smoke"
    assert result["usage"]["naive_luna"]["adapter_requests"] == 0
    assert result["usage"]["deepseek_naive_endpoint"]["adapter_requests"] == 0
    assert result["protocol"]["naive_can_gate_or_abort_primary"] is False
