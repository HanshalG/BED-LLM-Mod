from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import inspect
import json
import math
from pathlib import Path

import pytest
from PIL import Image

from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_vlm_mechanics_tree as tree
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_vlm_bed as bed


def test_mechanics_adapter_reserves_luna_attempt_cost(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    adapter = tree._adapter(output_dir=tmp_path, run_id="precharge-test")
    assert adapter.max_request_cost_usd == serving.MAX_REQUEST_COST_USD


def _image_bytes(index: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (36, 28), (50 + index, 100, 150)).save(
        output, format="JPEG"
    )
    return output.getvalue()


def _task(index: int) -> bed.VisualTask:
    image_ids = tuple(f"image-{image_index:02d}" for image_index in range(14))
    labels = {
        image_id: image_index % 2 == 0
        for image_index, image_id in enumerate(image_ids)
    }
    return bed.VisualTask(
        task_id=f"task-{index:012d}",
        image_ids=image_ids,
        initial_history=tuple(
            sorted((image_id, labels[image_id]) for image_id in image_ids[:4])
        ),
        candidate_ids=image_ids[4:12],
        endpoint_ids=image_ids[12:14],
        image_bytes={
            image_id: _image_bytes(image_index)
            for image_index, image_id in enumerate(image_ids)
        },
        actual_labels=labels,
        hidden_values=(
            f"secret concept {index}",
            f"images/{index:04d}/pos__0__source.jpg",
        ),
    )


def _request(messages):
    return json.loads(
        next(
            item["text"]
            for item in messages[0]["content"]
            if item.get("type") == "text"
        )
    )


def _response(messages, *, seed: int | None = None) -> str:
    request = _request(messages)
    image_ids = request["image_order"]
    observed = {
        item["image_id"]: item["label"] == "positive"
        for item in request["observed_labels"]
    }
    initial_ids = set(image_ids[:4])
    extras = sorted(set(observed) - initial_ids)
    first_extra = extras[0] if extras else None
    rows = []
    for hypothesis_index, hypothesis_id in enumerate(bed.HYPOTHESIS_IDS):
        probabilities = []
        centered = (hypothesis_index - 4.5) / 4.5
        for image_index, image_id in enumerate(image_ids):
            if image_id in observed:
                value = 90 - hypothesis_index if observed[image_id] else 10 + hypothesis_index
            else:
                seed_scale = 0 if seed is None else seed % 7
                amplitude = 8 + seed_scale * 3 + (image_index % 4) * 2
                if first_extra == "image-05":
                    amplitude = 35 if image_id != first_extra else 8
                elif first_extra is not None:
                    amplitude = 8
                base = 65 if image_index % 2 == 0 else 35
                value = round(base + amplitude * centered)
                value = min(95, max(5, value))
            probabilities.append(value)
        branch_tag = first_extra or "root"
        label_tag = (
            "none" if first_extra is None else str(int(observed[first_extra]))
        )
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "rule": (
                    f"{request['task_id']} {branch_tag} {label_tag} "
                    f"semantic visual rule {hypothesis_index + 1}"
                ),
                "history_weight": 20 - hypothesis_index,
                "positive_probabilities": probabilities,
            }
        )
    return json.dumps({"hypotheses": rows})


class FixtureAdapter:
    def __init__(self):
        self.requests = 0
        self.seeded_batch_sizes = []

    def chat_complete_messages_batched_structured(self, batch_messages, **kwargs):
        del kwargs
        self.requests += len(batch_messages)
        return [_response(messages) for messages in batch_messages]

    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages, seeds, **kwargs
    ):
        del kwargs
        assert len(batch_messages) == len(seeds)
        self.seeded_batch_sizes.append(len(batch_messages))
        self.requests += len(batch_messages)
        return [
            _response(messages, seed=seed)
            for messages, seed in zip(batch_messages, seeds, strict=True)
        ]

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": self.requests * 100,
            "adapter_completion_tokens": self.requests * 200,
            "adapter_cost_usd": self.requests * 0.001,
        }


def _serving_result(tmp_path: Path, tasks) -> Path:
    cases = serving.build_smoke_cases(tasks)

    class ServingAdapter:
        def chat_complete_messages_batched_structured(self, batch_messages, **kwargs):
            del kwargs
            return [_response(messages) for messages in batch_messages]

        def usage_snapshot(self):
            return {
                "adapter_requests": 10,
                "http_attempts": 10,
                "retry_count": 0,
                "provider_error_retries": 0,
                "adapter_reasoning_tokens": 0,
                "forced_exits": 0,
                "adapter_prompt_tokens": 1000,
                "adapter_completion_tokens": 2000,
                "adapter_cost_usd": 0.01,
            }

    output = tmp_path / "serving"
    result = serving.run_smoke(
        output_dir=output,
        run_id="serving-fixture",
        tasks=tasks,
        adapter=ServingAdapter(),
    )
    assert result["status"] == "passed"
    assert len(cases) == 10
    return output / "RESULT.json"


def test_full_fixture_tree_is_shared_executable_and_endpoint_scored(
    tmp_path: Path,
) -> None:
    tasks = [_task(index) for index in range(4)]
    serving_result = _serving_result(tmp_path, tasks)
    adapter = FixtureAdapter()
    result = tree.run_mechanics(
        output_dir=tmp_path / "tree",
        run_id="tree-fixture",
        serving_result=serving_result,
        tasks=tasks,
        adapter=adapter,
    )
    assert result["status"] == "mechanics_pass"
    assert result["gates"]["all_pass"]
    assert result["gates"][
        "simulated_branch_labels_beat_constant_half_brier_in_both_classes"
    ]
    assert result["branch_label_obedience"]["negative_mean_brier"] < 0.25
    assert result["branch_label_obedience"]["positive_mean_brier"] < 0.25
    assert result["terminal_label_obedience"]["negative_mean_brier"] < 0.25
    assert result["terminal_label_obedience"]["positive_mean_brier"] < 0.25
    assert result["gates"][
        "terminal_beliefs_retain_both_queried_labels_better_than_constant_half"
    ]
    assert result["protocol"]["first_stage_requests"] == 132
    assert result["protocol"]["conditioned_branch_requests"] == 64
    assert result["protocol"]["history_blind_branch_requests"] == 64
    assert result["usage"]["adapter_requests"] == (
        132 + result["protocol"]["distinct_final_history_requests"]
    )
    assert result["protocol"]["development_accessed"] is False
    assert set(result["pooled_policy_metrics"]) == set(tree.POLICIES)
    assert all(
        set(task_result["policies"]) == set(tree.POLICIES)
        for task_result in result["trees"]
    )
    assert all(
        len(task_result["all_first_action_paths"]) == 8
        for task_result in result["trees"]
    )
    assert set(result["mean_ranking_fidelity"]) == set(tree.SCORE_POLICIES)
    assert result["gates"][
        "fixed_score_dynamic_update_exactly_matches_fixed_first_and_dynamic_second"
    ]
    assert all(
        task_result["policies"]["fixed_score_dynamic_update"][
            "first_image_id"
        ]
        == task_result["policies"]["fixed_depth2"]["first_image_id"]
        and task_result["policies"]["fixed_score_dynamic_update"][
            "final_history_key"
        ]
        == task_result["all_first_action_paths"][
            task_result["policies"]["fixed_depth2"]["first_image_id"]
        ]["final_history_key"]
        for task_result in result["trees"]
    )
    changed = [
        task_result
        for task_result in result["trees"]
        if task_result["policies"]["dynamic_depth2"]["first_image_id"]
        != task_result["policies"]["myopic_width"]["first_image_id"]
    ]
    assert changed
    assert all(
        task_result["policies"]["dynamic_depth2"]["first_score_margin"]
        >= tree.MIN_ACTION_MARGIN_NATS
        for task_result in changed
    )
    raw = json.loads(
        (tmp_path / "tree/private/RAW_RESPONSES.json").read_text()
    )
    assert raw["endpoint_labels_accessed_after_all_query_selection"] is True
    assert len(raw["first_stage_request_seeds"]) == 132
    assert len(raw["final_request_seeds"]) == result["protocol"][
        "distinct_final_history_requests"
    ]
    assert raw["development_accessed"] is False
    final_batches = raw["final_request_pairing"]["dispatch_batches"]
    assert adapter.seeded_batch_sizes[1:] == [
        batch["request_count"] for batch in final_batches
    ]
    assert all(
        size <= tree.CONCURRENCY for size in adapter.seeded_batch_sizes[1:]
    )
    serving_verification = aug10.validate_serving_artifact(
        serving_result, tasks=tasks
    )
    mechanics_verification = aug10.validate_mechanics_artifact(
        tmp_path / "tree/RESULT.json",
        serving_result=serving_result,
        tasks=tasks,
    )
    assert serving_verification["verified"]
    assert mechanics_verification["verified"]

    raw["final_request_pairing"]["dispatch_batches"][0]["request_count"] -= 1
    raw_path = tmp_path / "tree/private/RAW_RESPONSES.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    result_path = tmp_path / "tree/RESULT.json"
    tampered_result = json.loads(result_path.read_text())
    tampered_result["raw_responses_sha256"] = tree.sha256_file(raw_path)
    result_path.write_text(json.dumps(tampered_result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="terminal batch size changed"):
        aug10.validate_mechanics_artifact(
            result_path,
            serving_result=serving_result,
            tasks=tasks,
        )


def test_shuffled_control_permutes_complete_continuation_values() -> None:
    task = _task(0)
    myopic = {
        candidate: 0.1 + index * 0.01
        for index, candidate in enumerate(sorted(task.candidate_ids))
    }
    dynamic = {
        candidate: myopic[candidate] + 0.2 + index * 0.03
        for index, candidate in enumerate(sorted(task.candidate_ids))
    }

    shuffled, mapping, values = tree.shuffled_continuation_control(
        task=task,
        myopic_scores=myopic,
        dynamic_scores=dynamic,
    )

    assert all(source != target for source, target in mapping.items())
    assert len(set(mapping.values())) == len(mapping)
    assert sorted(values["dynamic_expected_continuation_utility"].values()) == sorted(
        values["shuffled_expected_continuation_utility"].values()
    )
    for target, source in mapping.items():
        assert values["shuffled_expected_continuation_utility"][target] == (
            values["dynamic_expected_continuation_utility"][source]
        )
        assert shuffled[target] == (
            myopic[target]
            + values["dynamic_expected_continuation_utility"][source]
        )


def test_shuffled_control_accepts_signed_support_regeneration_values() -> None:
    task = _task(0)
    myopic = {candidate: 0.2 for candidate in task.candidate_ids}
    dynamic = {
        candidate: 0.1 - index * 0.01
        for index, candidate in enumerate(sorted(task.candidate_ids))
    }
    shuffled, _, values = tree.shuffled_continuation_control(
        task=task, myopic_scores=myopic, dynamic_scores=dynamic
    )
    assert all(math.isfinite(value) for value in shuffled.values())
    assert all(
        value < 0.0
        for value in values["dynamic_expected_continuation_utility"].values()
    )


def test_shuffled_control_never_reuses_a_mismatched_branch_support() -> None:
    task = _task(0)
    mapping = tree.rotated_candidate_mapping(task)
    assert set(mapping) == set(task.candidate_ids)
    assert set(mapping.values()) == set(task.candidate_ids)
    assert "branches" not in inspect.signature(
        tree.shuffled_continuation_control
    ).parameters


def test_history_blind_pairs_share_seed_and_hide_the_simulated_answer() -> None:
    tasks = [_task(index) for index in range(4)]
    cases = tree.first_stage_cases(tasks)
    messages = [
        bed.build_belief_messages(case.task, case.history) for case in cases
    ]
    seeds = tree.request_seeds_for_cases(cases, base_seed=123_000)
    diagnostics = tree.paired_request_diagnostics(
        cases=cases, messages=messages, seeds=seeds
    )
    assert diagnostics["pair_count"] == 64
    assert diagnostics["unique_pair_seed_count"] == 64
    assert diagnostics["gates"]["all_pass"]
    assert all(
        row["same_requested_seed"]
        and row["adjacent_dynamic_then_blind"]
        and row["same_dispatch_batch"]
        and row["blind_history_is_initial"]
        and row["blind_prompt_matches_root"]
        and row["prompts_differ_only_by_simulated_answer"]
        for row in diagnostics["pairs"]
    )
    assert diagnostics["gates"][
        "each_pair_is_adjacent_dynamic_then_blind"
    ]
    assert diagnostics["gates"]["each_pair_shares_one_dispatch_batch"]

    tampered = list(seeds)
    blind_index = next(
        index for index, case in enumerate(cases) if case.kind == "history_blind"
    )
    tampered[blind_index] += 99_000
    assert not tree.paired_request_diagnostics(
        cases=cases, messages=messages, seeds=tampered
    )["gates"]["all_pass"]

    reordered_cases = list(cases)
    reordered_messages = list(messages)
    reordered_seeds = list(seeds)
    first_dynamic = next(
        index for index, case in enumerate(cases) if case.kind == "branch"
    )
    for values in (reordered_cases, reordered_messages, reordered_seeds):
        values[first_dynamic + 1], values[first_dynamic + 2] = (
            values[first_dynamic + 2],
            values[first_dynamic + 1],
        )
    reordered = tree.paired_request_diagnostics(
        cases=reordered_cases,
        messages=reordered_messages,
        seeds=reordered_seeds,
    )
    assert not reordered["gates"][
        "each_pair_is_adjacent_dynamic_then_blind"
    ]
    assert not reordered["gates"]["all_pass"]


def test_first_action_plans_are_invariant_to_unreleased_candidate_labels() -> None:
    task = _task(0)
    root = bed.parse_belief_response(
        _response(bed.build_belief_messages(task, task.initial_history)),
        image_ids=task.image_ids,
        history=task.initial_history,
    )
    branches = {}
    history_blind = {}
    for candidate in task.candidate_ids:
        for label in (False, True):
            history = tuple(sorted((*task.initial_history, (candidate, label))))
            branches[(candidate, label)] = bed.parse_belief_response(
                _response(bed.build_belief_messages(task, history)),
                image_ids=task.image_ids,
                history=history,
            )
            history_blind[(candidate, label)] = root
    flipped = replace(
        task,
        actual_labels={
            image_id: (
                not label if image_id in task.candidate_ids else label
            )
            for image_id, label in task.actual_labels.items()
        },
    )

    original_plan = tree.plan_task_policies(
        task=task,
        root=root,
        branches=branches,
        history_blind_branches=history_blind,
    )
    flipped_plan = tree.plan_task_policies(
        task=flipped,
        root=root,
        branches=branches,
        history_blind_branches=history_blind,
    )
    endpoint_flipped = replace(
        task,
        actual_labels={
            image_id: (
                not label if image_id in task.endpoint_ids else label
            )
            for image_id, label in task.actual_labels.items()
        },
    )
    endpoint_flipped_plan = tree.plan_task_policies(
        task=endpoint_flipped,
        root=root,
        branches=branches,
        history_blind_branches=history_blind,
    )

    assert original_plan["root_scores"] == flipped_plan["root_scores"]
    assert original_plan["root_scores"] == endpoint_flipped_plan["root_scores"]
    assert original_plan["score_objective"] == tree.SCORE_OBJECTIVE
    assert original_plan["continuation_values"] == flipped_plan[
        "continuation_values"
    ]
    assert {
        policy: row["first_image_id"]
        for policy, row in original_plan["policies"].items()
    } == {
        policy: row["first_image_id"]
        for policy, row in flipped_plan["policies"].items()
    }
    blind_row = original_plan["policies"]["history_blind_depth2"]
    blind_first = blind_row["first_image_id"]
    realized_label = bool(task.actual_labels[blind_first])
    remaining = tuple(
        candidate for candidate in task.candidate_ids if candidate != blind_first
    )
    assert blind_row["second_scores"] == bed.candidate_endpoint_eigs(
        branches[(blind_first, realized_label)], remaining, task.endpoint_ids
    )
    dynamic_row = original_plan["policies"]["dynamic_depth2"]
    dynamic_first = dynamic_row["first_image_id"]
    dynamic_label = bool(task.actual_labels[dynamic_first])
    assert dynamic_row["second_scores"] == bed.candidate_endpoint_eigs(
        branches[(dynamic_first, dynamic_label)],
        tuple(
            candidate
            for candidate in task.candidate_ids
            if candidate != dynamic_first
        ),
        task.endpoint_ids,
    )
    fixed = original_plan["policies"]["fixed_depth2"]
    matched = original_plan["policies"]["fixed_score_dynamic_update"]
    assert matched["first_image_id"] == fixed["first_image_id"]
    assert matched["first_score"] == fixed["first_score"]
    assert matched["second_scores"] == bed.candidate_endpoint_eigs(
        branches[(matched["first_image_id"], bool(task.actual_labels[matched["first_image_id"]]))],
        tuple(
            candidate
            for candidate in task.candidate_ids
            if candidate != matched["first_image_id"]
        ),
        task.endpoint_ids,
    )


def test_final_cases_deduplicate_shared_policy_histories() -> None:
    task = _task(0)
    history = tuple(
        sorted((*task.initial_history, ("image-04", True), ("image-05", False)))
    )
    plans = {
        task.task_id: {
            "policies": {
                policy: {
                    "final_history": history,
                    "final_history_key": tree.history_key(history),
                }
                for policy in tree.POLICIES
            }
        }
    }
    cases = tree.final_cases([task], plans)
    assert len(cases) == 1


def test_terminal_requests_use_common_seed_and_primary_pair_order() -> None:
    tasks = [_task(0), _task(1)]
    plans = {}
    cases = []
    for task_index, task in enumerate(tasks):
        dynamic = tuple(
            sorted(
                (*task.initial_history, (task.candidate_ids[0], True), (task.candidate_ids[1], False))
            )
        )
        blind = tuple(
            sorted(
                (*task.initial_history, (task.candidate_ids[2], False), (task.candidate_ids[3], True))
            )
        )
        plans[task.task_id] = {
            "policies": {
                policy: {
                    "final_history": dynamic if policy == "dynamic_depth2" else blind,
                    "final_history_key": tree.history_key(
                        dynamic if policy == "dynamic_depth2" else blind
                    ),
                }
                for policy in tree.POLICIES
            }
        }
        cases.extend(tree.final_cases([task], plans))

    seeds = tree.request_seeds_for_cases(
        cases, base_seed=900_000, final_stage=True
    )
    diagnostics = tree.final_request_diagnostics(
        cases=cases, seeds=seeds, plans=plans
    )

    assert diagnostics["gates"]["all_pass"]
    assert len(set(seeds[:2])) == 1
    assert len(set(seeds[2:])) == 1
    assert seeds[0] != seeds[2]
    assert all(
        row["dynamic_then_history_blind_are_adjacent_when_distinct"]
        and row["dynamic_and_history_blind_share_dispatch_batch"]
        and row["task_cases_share_one_dispatch_batch"]
        for row in diagnostics["tasks"]
    )

    tampered = list(seeds)
    tampered[1] += 1
    assert not tree.final_request_diagnostics(
        cases=cases, seeds=tampered, plans=plans
    )["gates"]["all_pass"]


def test_terminal_dispatch_keeps_task_atomic_at_naive_batch_boundary() -> None:
    counts = [10, 9, 4, 4]
    cases = []
    for task_index, count in enumerate(counts):
        task = _task(task_index)
        for history_index in range(count):
            cases.append(
                tree.BeliefCase(
                    case_id=f"{task.task_id}-synthetic-{history_index}",
                    task=task,
                    history=((f"synthetic-{history_index}", True),),
                    kind="final",
                )
            )

    batches = tree.task_preserving_dispatch_batches(cases)

    assert [(batch["start"], batch["stop"]) for batch in batches] == [
        (0, 23),
        (23, 27),
    ]
    assert cases[23].task.task_id == cases[24].task.task_id
    for batch in batches:
        assert batch["request_count"] <= tree.CONCURRENCY
        for task_id in batch["task_ids"]:
            task_indices = [
                index
                for index, case in enumerate(cases)
                if case.task.task_id == task_id
            ]
            assert min(task_indices) >= batch["start"]
            assert max(task_indices) < batch["stop"]


def test_terminal_common_seed_cancels_adversarial_seed_only_policy_effect() -> None:
    task = _task(0)
    dynamic = tuple(
        sorted(
            (*task.initial_history, (task.candidate_ids[0], True), (task.candidate_ids[1], False))
        )
    )
    blind = tuple(
        sorted(
            (*task.initial_history, (task.candidate_ids[2], False), (task.candidate_ids[3], True))
        )
    )
    plans = {
        task.task_id: {
            "policies": {
                policy: {
                    "final_history": dynamic if policy == "dynamic_depth2" else blind,
                    "final_history_key": tree.history_key(
                        dynamic if policy == "dynamic_depth2" else blind
                    ),
                }
                for policy in tree.POLICIES
            }
        }
    }
    cases = tree.final_cases([task], plans)
    unique_history_seeds = [101, 202]
    common_task_seeds = tree.request_seeds_for_cases(
        cases, base_seed=101, final_stage=True
    )

    def seed_only_brier(seed: int) -> float:
        probability = 0.9 if seed % 2 else 0.1
        return (probability - 1.0) ** 2

    assert seed_only_brier(unique_history_seeds[0]) != seed_only_brier(
        unique_history_seeds[1]
    )
    assert seed_only_brier(common_task_seeds[0]) == seed_only_brier(
        common_task_seeds[1]
    )


def test_reconcile_full_tree_preserves_prior_smoke_spend() -> None:
    ledger = {
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": 0.03,
        "bongard_luna_vlm_serving_smoke": {
            "status": "passed",
            "actual_cost_usd": 0.03,
        },
    }
    result = tree.reconcile_ledger(
        ledger=ledger,
        measured_cost_usd=0.22,
        live_after={
            "total_credits_usd": 130.0,
            "total_usage_usd": 100.2,
            "balance_usd": 29.8,
        },
        status="mechanics_pass",
    )
    assert math.isclose(result["recorded_actual_spend_usd"], 0.25)
    assert math.isclose(
        result["bongard_luna_vlm_mechanics_tree"]["actual_cost_usd"],
        0.22,
    )
    assert result["bongard_luna_vlm_serving_smoke"]["status"] == "passed"
