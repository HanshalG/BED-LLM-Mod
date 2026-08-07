from __future__ import annotations

import json

import pytest

from scripts import regretbench_deepseek_dynamic_depth2_policy as policy
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
    return {
        "selected_roots": roots,
        "conditioned_root_risks": [
            {"brier": 0.10},
            {"brier": 0.10 + predicted_gain},
            {"brier": 0.20},
            {"brier": 0.22},
        ],
        "policies": {
            "dynamic_depth2": {"brier": dynamic_brier, "log_loss": 0.10},
            "myopic_width": {
                "brier": dynamic_brier + realized_gain,
                "log_loss": 0.30,
            },
            "history_blind_depth2": {"brier": 0.20, "log_loss": 0.25},
            "fixed_depth2": {"brier": 0.18, "log_loss": 0.22},
            "random": {"brier": 0.25, "log_loss": 0.30},
        },
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
    assert summary["predicted_to_realized_dynamic_myopic"]["spearman"] == pytest.approx(1)


class _FixtureAdapter:
    WORDS = ("zero", "one", "two", "three")

    def __init__(self) -> None:
        self.requests = 0
        all_cigs = [
            *recovery.load_stage_cigs("smoke"),
            *recovery.load_stage_cigs("development"),
        ]
        self.facets = {
            cig.cig_id: cig.semantic_facets[0].replace("_", " ") for cig in all_cigs
        }
        self.truth_aliases = {}
        for index, cig in enumerate(recovery.load_stage_cigs("development")):
            _, truth = recovery.sample_truth(cig, policy.TRUTH_SEED_START + index)
            self.truth_aliases[cig.cig_id] = str(
                (truth.slots or {})["answer_aliases"]
            ).split("|")[0]

    def _root_from_question(self, text: str) -> int:
        for index, word in enumerate(self.WORDS):
            if f"option {word}" in text.casefold():
                return index
        return 0

    def _response(self, payload: dict, seed: int) -> str:
        task_id = payload["task_id"]
        dialogue = payload["dialogue"]
        root = self._root_from_question(dialogue[0]["content"]) if dialogue else None
        branch_formal = policy.BRANCH_SEED_START <= seed < policy.ACTUAL_FIRST_SEED_START
        answers = [f"fixture answer {index}" for index in range(8)]
        weights = [1] * 8
        if branch_formal:
            local = seed - policy.BRANCH_SEED_START
            seed_root = (local % 64) // 16
            hypothesis = (local % 16) // 2
            conditioned = bool(dialogue)
            if (conditioned and seed_root == 0) or (
                not conditioned and seed_root == 2
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

        facet = self.facets[task_id]
        questions = [
            f"Which {facet} do you mean for option {word}?" for word in self.WORDS
        ]
        hypotheses = []
        for index in range(8):
            hypotheses.append(
                {
                    "interpretation": f"fixture interpretation {index}",
                    "final_answer": answers[index],
                    "prior_weight": weights[index],
                    "predicted_replies": [
                        f"sim-q0-h{index % 2}",
                        f"sim-q1-h{index}",
                        f"sim-q2-h{index % 2}",
                        f"sim-q3-h{index % 3}",
                    ],
                }
            )
        return json.dumps({"hypotheses": hypotheses, "questions": questions})

    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages, seeds, **kwargs
    ):
        assert kwargs["response_format"] == policy.enriched_response_format()
        responses = []
        for messages, seed in zip(batch_messages, seeds, strict=True):
            payload = json.loads(messages[-1]["content"])
            responses.append(self._response(payload, seed))
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
    adapter = _FixtureAdapter()

    result = policy.run_development(
        output_dir=tmp_path / "development",
        run_id="fixture-policy-development",
        support_smoke_result=tmp_path / "support-smoke.json",
        support_development_result=tmp_path / "support-development.json",
        policy_smoke_result=tmp_path / "policy-smoke.json",
        adapter=adapter,
        bootstrap_samples=50,
    )

    assert result["protocol"]["planning_requests"] == 8_256
    assert result["protocol"]["expected_requests"] == adapter.requests
    assert 8_256 < adapter.requests <= 8_768
    assert all(result["mechanics_gates"].values())
    assert result["status"] in {"passed", "gated_null"}
    assert len(result["tasks"]) == 64
    assert result["protocol"]["selection_frozen_before_truth_access"] is True
    public = (tmp_path / "development" / "RESULT.json").read_text()
    assert "aliases" not in public
    assert "selected_questions" not in public
    assert (tmp_path / "development" / "private" / "FROZEN_SELECTIONS.json").exists()
