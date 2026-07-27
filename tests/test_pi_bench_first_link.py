from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import Config, ModelSpec
from scripts.pi_bench_first_link import (
    BED_POLICIES,
    BranchRefresh,
    BranchSemanticMap,
    BeliefAndQuestions,
    PrivateTask,
    PiBenchGPT54Adapter,
    PublicTask,
    RequirementWorld,
    SemanticMap,
    TurnCase,
    assert_policy_messages_target_blind,
    build_rollout_branches,
    build_public_result,
    evaluate_truth_recall,
    exact_one_sided_permutation_p,
    immediate_scores,
    invalid_question_reason,
    load_tasks,
    parse_belief,
    run_actual_trajectories,
    run_initial_planning,
    run_official_turns,
    terminal_scores,
    validate_source,
)


def _belief() -> BeliefAndQuestions:
    return BeliefAndQuestions(
        worlds=tuple(
            RequirementWorld(
                requirements=(
                    f"world {world_index} requirement a",
                    f"world {world_index} requirement b",
                    f"world {world_index} requirement c",
                )
            )
            for world_index in range(8)
        ),
        questions=tuple(
            f"Could you clarify topic {index}?" for index in range(6)
        ),
    )


def _initial_map() -> SemanticMap:
    matches = []
    for question_index in range(6):
        for _world_index in range(8):
            if question_index == 1:
                matches.append((0, 1, 2))
            elif question_index == 0:
                matches.append((0,))
            else:
                matches.append(())
    return SemanticMap(
        matches=tuple(matches),
        question_count=6,
        world_count=8,
    )


def test_rollout_scoring_uses_original_particle_and_handles_complete_root() -> None:
    belief = _belief()
    semantic_map = _initial_map()
    selected_worlds = (0, 1, 2, 3)
    branches = build_rollout_branches("manual-task", belief, semantic_map)

    assert len(branches) == 20
    assert all(branch.root_question_index != 1 for branch in branches)
    refresh = BranchRefresh(
        worlds=tuple(
            RequirementWorld(requirements=("remaining x", "remaining y"))
            for _ in range(4)
        ),
        questions=tuple(
            f"What about followup {index}?" for index in range(4)
        ),
    )
    support_map = SemanticMap(
        matches=tuple((0,) for _ in range(4 * 4)),
        question_count=4,
        world_count=4,
    )
    branch_map = BranchSemanticMap(
        support_map=support_map,
        particle_matches=((), (), (), ()),
    )
    terminal = terminal_scores(
        belief,
        semantic_map,
        selected_worlds,
        branches,
        tuple(refresh for _ in branches),
        tuple(branch_map for _ in branches),
    )

    assert immediate_scores(belief, semantic_map)[1] == 1.0
    assert terminal[1] == 1.0
    assert terminal[0] == pytest.approx(2 / 3)
    assert terminal[2] == pytest.approx(2 / 3)


def test_parse_belief_rejects_generic_or_duplicate_questions() -> None:
    payload = {
        "worlds": [
            {
                "requirements": [
                    f"requirement {world_index} a",
                    f"requirement {world_index} b",
                    f"requirement {world_index} c",
                ]
            }
            for world_index in range(8)
        ],
        "questions": [
            "Could you clarify the budget?",
            "Could you clarify the timeline?",
            "Could you clarify the audience?",
            "Could you clarify the format?",
            "Could you clarify the constraints?",
            "Is there anything else?",
        ],
    }
    with pytest.raises(ValueError, match="invalid questions"):
        parse_belief(
            json.dumps(payload),
            world_count=8,
            question_count=6,
            min_requirements=3,
        )
    assert invalid_question_reason("Is there anything else?") == (
        "generic_or_omnibus"
    )
    assert (
        invalid_question_reason(
            "Does the draft exist before doing anything else?"
        )
        is None
    )


def test_gpt54_payload_uses_only_routable_structured_parameters(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = Config(
        task="mediq",
        run_id="payload-test",
        mediq_seed=24422,
        openrouter_budget_usd=140.0,
        openrouter_spend_path=str(tmp_path / "spend.json"),
    )
    adapter = PiBenchGPT54Adapter(
        ModelSpec(
            model="openai/gpt-5.4",
            backend="openrouter",
            reasoning_effort="none",
        ),
        config,
    )
    payload = adapter._payload(
        [{"role": "user", "content": "test"}],
        0.0,
        1,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    )

    assert not {"temperature", "top_p", "top_k", "n"} & set(payload)
    assert payload["reasoning"] == {"enabled": False, "exclude": True}
    assert payload["seed"] == 24422


def test_policy_leakage_guard_allows_visible_reply_but_rejects_unrevealed() -> None:
    private = PrivateTask(
        task_id="t",
        hidden_intents=(
            "The output must include a sensitivity analysis.",
            "The output must use a six month forecast horizon.",
        ),
        initial_statuses=("not_provided", "not_provided"),
    )
    visible_reply = private.hidden_intents[0]
    messages = [{"role": "user", "content": visible_reply}]
    assert_policy_messages_target_blind(
        messages,
        private,
        allowed_visible=(visible_reply,),
    )
    with pytest.raises(ValueError, match="indexes=\\[2\\]"):
        assert_policy_messages_target_blind(
            [{"role": "user", "content": private.hidden_intents[1]}],
            private,
            allowed_visible=(visible_reply,),
        )


class _DecisionModel:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched_structured(
        self,
        batch_messages,
        *,
        temperature,
        block_size,
        response_format,
        max_new_tokens=None,
    ):
        del temperature, block_size, max_new_tokens
        name = response_format["json_schema"]["name"]
        outputs = []
        for messages in batch_messages:
            payload = json.loads(messages[-1]["content"])
            count = len(payload["candidates"])
            if "satisfaction" in name:
                decisions = [False] * count
            else:
                decisions = [False] * count
                decisions[min(1, count - 1)] = True
            outputs.append(json.dumps({"decisions": decisions}))
        self.requests += len(batch_messages)
        return outputs

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


class _PipelineModel:
    def __init__(self, *, reasoning_tokens: int = 0) -> None:
        self.requests = 0
        self.reasoning_tokens = reasoning_tokens

    @staticmethod
    def _belief(schema):
        properties = schema["json_schema"]["schema"]["properties"]
        world_count = properties["worlds"]["minItems"]
        question_count = properties["questions"]["minItems"]
        return {
            "worlds": [
                {
                    "requirements": [
                        f"generated world {world} requirement {index}"
                        for index in range(4)
                    ]
                }
                for world in range(world_count)
            ],
            "questions": [
                f"Could you clarify generated topic {index}?"
                for index in range(question_count)
            ],
        }

    @staticmethod
    def _semantic_map(payload):
        vectors = []
        for pair in payload["ordered_pairs"]:
            count = len(pair["requirements"])
            question_index = int(pair["question_index"])
            indexes = ["R0"]
            if question_index == 0 and count >= 2:
                indexes.append("R1")
            vectors.append(indexes)
        return {"matches": vectors}

    @staticmethod
    def _refresh(payload):
        return {
            "branches": [
                {
                    "worlds": [
                        {
                            "requirements": [
                                f"branch {branch['branch_index']} world {world} "
                                f"remaining {index}"
                                for index in range(2)
                            ]
                        }
                        for world in range(4)
                    ],
                    "questions": [
                        f"Could you clarify branch {branch['branch_index']} "
                        f"followup {index}?"
                        for index in range(4)
                    ],
                }
                for branch in payload["branches"]
            ]
        }

    @staticmethod
    def _branch_map(payload):
        branches = []
        for branch in payload["branches"]:
            support_matches = []
            for pair in branch["support_pairs"]:
                support_matches.append(
                    ["R0"] if int(pair["question_index"]) == 0 else []
                )
            particle_matches = []
            root_index = int(branch["branch_index"]) // 4
            for pair in branch["particle_pairs"]:
                if int(pair["question_index"]) == 0:
                    if root_index == 1:
                        indexes = [
                            f"R{index}"
                            for index in range(len(pair["requirements"]))
                        ]
                    else:
                        indexes = ["R0"]
                else:
                    indexes = []
                particle_matches.append(indexes)
            branches.append(
                {
                    "support_matches": support_matches,
                    "particle_matches": particle_matches,
                }
            )
        return {"branches": branches}

    def chat_complete_messages_batched_structured(
        self,
        batch_messages,
        *,
        temperature,
        block_size,
        response_format,
        max_new_tokens=None,
    ):
        del temperature, block_size, max_new_tokens
        name = response_format["json_schema"]["name"]
        outputs = []
        for messages in batch_messages:
            payload = json.loads(messages[-1]["content"])
            if "initial_support" in name or "realized_refresh" in name:
                result = self._belief(response_format)
            elif "initial_semantic_map" in name or "realized_semantic_map" in name:
                result = self._semantic_map(payload)
            elif "rollout_refresh" in name:
                result = self._refresh(payload)
            elif "rollout_followup_map" in name:
                result = self._branch_map(payload)
            elif "naive_question" in name:
                result = {"question": "Could you clarify the desired outcome?"}
            elif "satisfaction" in name:
                result = {
                    "decisions": [False] * len(payload["candidates"])
                }
            elif "targeted" in name:
                result = {
                    "decisions": [
                        index == 0
                        for index in range(len(payload["candidates"]))
                    ]
                }
            elif "truth_recall" in name:
                result = {
                    "represented": [True]
                    * len(payload["actual_requirements"])
                }
            else:
                raise AssertionError(f"unexpected schema {name}")
            outputs.append(json.dumps(result))
        self.requests += len(batch_messages)
        return outputs

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": self.reasoning_tokens,
            "forced_exits": 0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def test_official_turn_deduplicates_paired_identical_cases() -> None:
    model = _DecisionModel()
    cases = [
        TurnCase(
            case_id=f"t:{policy}:turn1",
            task_id="t",
            hidden_intents=("first intent", "second intent", "third intent"),
            statuses=("not_provided", "not_provided", "not_provided"),
            question="Could you clarify the second intent?",
        )
        for policy in BED_POLICIES[:2]
    ]
    raw_calls = []
    results = run_official_turns(
        cases,
        model=model,
        block_size=16,
        raw_calls=raw_calls,
    )

    assert model.requests == 2
    assert results[cases[0].case_id] == results[cases[1].case_id]
    assert results[cases[0].case_id].provided_indexes == (2,)
    assert results[cases[0].case_id].statuses == (
        "not_provided",
        "provided",
        "not_provided",
    )


def test_exact_one_sided_permutation_p() -> None:
    assert exact_one_sided_permutation_p([1.0, 1.0, 1.0]) == 0.125
    assert exact_one_sided_permutation_p([0.0, 0.0]) == 1.0


def test_deterministic_pipeline_runs_end_to_end_without_private_leakage() -> None:
    tasks = [
        PublicTask(
            task_id=f"task_{index}",
            persona=f"persona_{index}",
            persona_context=f"public persona context {index}",
            initial_input=f"Please help with public task {index}.",
        )
        for index in range(5)
    ]
    private = {
        task.task_id: PrivateTask(
            task_id=task.task_id,
            hidden_intents=tuple(
                f"private intent {task.task_id} number {intent}"
                for intent in range(4)
            ),
            initial_statuses=("not_provided",) * 4,
        )
        for task in tasks
    }
    bed_model = _PipelineModel()
    naive_model = _PipelineModel(reasoning_tokens=10)
    raw_calls = []
    planning = run_initial_planning(
        tasks,
        private,
        model=bed_model,
        block_size=64,
        raw_calls=raw_calls,
    )
    trajectories = run_actual_trajectories(
        tasks,
        private,
        planning,
        bed_model=bed_model,
        naive_model=naive_model,
        block_size=64,
        raw_calls=raw_calls,
    )
    initial_recall, _ = evaluate_truth_recall(
        tasks,
        private,
        planning,
        trajectories,
        model=bed_model,
        block_size=64,
        raw_calls=raw_calls,
    )
    result = build_public_result(
        stage="serving_smoke",
        tasks=tasks,
        private_tasks=private,
        cohort={"partition": "mechanics"},
        planning=planning,
        trajectories=trajectories,
        initial_recall=initial_recall,
        bed_usage={
            "reasoning_tokens": 0,
            "forced_exits": 0,
            "cost_usd": 0.0,
            "physical_requests": bed_model.requests,
        },
        naive_usage={
            "reasoning_tokens": 10,
            "forced_exits": 0,
            "cost_usd": 0.0,
            "physical_requests": naive_model.requests,
        },
        elapsed_seconds=1.0,
        private_raw_sha256="0" * 64,
    )

    assert result["summary"]["myopic_depth2_root_changed_count"] == 5
    assert result["summary"]["gates"]["all_pass"]
    assert result["usage"]["total_cost_usd"] == 0.0


def test_pinned_serving_cohort_loads_without_model_calls() -> None:
    repo = Path("/tmp/pi-bench-audit.4P0JJr")
    if not repo.exists():
        pytest.skip("pinned Pi-Bench checkout is not available")
    manifest_path = Path(
        "results/nonmyopic/pi_bench_release_source_manifest.json"
    )
    manifest = validate_source(repo, manifest_path)
    public, private, cohort = load_tasks(
        repo, manifest, stage="serving_smoke"
    )

    assert len(public) == len(private) == 5
    assert cohort["partition"] == "mechanics"
    assert [task.task_id for task in public] == [
        "Financier_task_001",
        "law_trainee_task_001",
        "marketer_task_001",
        "pharmacist_task_001",
        "researcher_task_001",
    ]
