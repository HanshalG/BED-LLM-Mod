#!/usr/bin/env python3
"""Run semantic-partition EIG on frozen InfoQuest common histories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts import infoquest_partition_eig_causal_gate as partition
from scripts import infoquest_refresh_continuation_gate as refresh
from scripts import infoquest_support_causal_link_gate as base


INTERFACE_VERSION = "infoquest-cached-partition-eig-1"
INTERFACE_VERSION_V2 = "infoquest-cached-partition-eig-2"
CACHED_PUBLIC_SHA256 = (
    "140f77447da9bd3d83e8a7be7fd45dc8fc792422d4e1cef398f6d8460d13ccc3"
)
CACHED_RAW_SHA256 = (
    "c4dec013386083afb4279754f4ffb1b0a2f6559bb2a2cf1863d80a849fe6e93e"
)
EXPECTED_SERVING_REQUESTS = 5
EXPECTED_MECHANICS_REQUESTS = 126
SERVING_MAX_COST_USD = 0.12
MECHANICS_MAX_COST_USD = 0.85

ModelBundle = refresh.ModelBundle
GateExecutionError = refresh.GateExecutionError


class CachedDeterministicGenerator(partition.DeterministicGenerator):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        stages = [
            messages[0]["content"].split("STAGE=", 1)[1].split(".", 1)[0]
            for messages in batch_messages
        ]
        if all(
            stage in {"PARTITION_DYNAMIC", "PARTITION_FIXED"}
            for stage in stages
        ):
            responses = []
            for stage, messages in zip(stages, batch_messages):
                request = json.loads(messages[-1]["content"])
                root = request["clarification_question"]
                if stage == "PARTITION_FIXED":
                    hypotheses = request["initial_hypotheses"]
                    selected = 3
                else:
                    seed = request["ambiguous_seed_message"]
                    hypotheses = [
                        (
                            f"{seed} Refreshed context {index} after {root} "
                            f"has goal {index} and constraint {index}."
                        )
                        for index in range(1, base.SUPPORT_SIZE + 1)
                    ]
                    selected = hashlib.sha256(root.encode()).digest()[0] % 4
                value: dict[str, Any] = {
                    **{
                        f"h{index}": hypotheses[index - 1]
                        for index in range(1, base.SUPPORT_SIZE + 1)
                    },
                    **{
                        f"w{index}": 10
                        for index in range(1, base.SUPPORT_SIZE + 1)
                    },
                }
                for action_index, action in enumerate(partition.ACTION_KEYS):
                    if action_index == selected:
                        labels = (0, 0, 1, 1, 2, 2, 3, 3)
                    elif action_index == (selected + 1) % 4:
                        labels = (0, 0, 0, 0, 1, 1, 1, 1)
                    else:
                        labels = (0,) * base.SUPPORT_SIZE
                    for index, label in enumerate(labels, start=1):
                        value[f"{action}{index}"] = label
                responses.append(json.dumps(value, separators=(",", ":")))
            self.requests += len(responses)
            return responses
        return super().chat_complete_messages_batched(
            batch_messages,
            temperature,
            block_size,
            max_new_tokens,
        )


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cached_histories(
    fixtures: Sequence[base.WorldFixture],
    *,
    raw_path: Path,
    public_path: Path,
    expected_raw_sha256: str = CACHED_RAW_SHA256,
    expected_public_sha256: str = CACHED_PUBLIC_SHA256,
) -> tuple[
    dict[int, base.InitialPolicy],
    list[list[str]],
]:
    if _sha256_path(raw_path) != expected_raw_sha256:
        raise ValueError("cached private raw SHA-256 mismatch")
    if _sha256_path(public_path) != expected_public_sha256:
        raise ValueError("cached public mechanics SHA-256 mismatch")
    raw = json.loads(raw_path.read_text())
    public = json.loads(public_path.read_text())
    if raw.get("interface_version") != discrete_interface_version():
        raise ValueError("cached raw interface version mismatch")
    if public.get("protocol", {}).get("interface_version") != (
        discrete_interface_version()
    ):
        raise ValueError("cached public interface version mismatch")
    if public.get("protocol", {}).get("private_raw_sha256") != (
        expected_raw_sha256
    ):
        raise ValueError("cached public artifact does not bind private raw")
    cached_fixture_ids = [
        value.get("fixture_id") for value in raw.get("private_fixtures", [])
    ]
    if cached_fixture_ids != [fixture.fixture_id for fixture in fixtures]:
        raise ValueError("cached fixture order mismatch")
    initial_raw = raw.get("initial")
    root_raw = raw.get("root_answers")
    if not isinstance(initial_raw, list) or len(initial_raw) != 3:
        raise ValueError("cached histories do not contain three initials")
    if not isinstance(root_raw, list) or len(root_raw) != 30:
        raise ValueError("cached histories do not contain thirty root answers")
    initials = {
        record_id: base.parse_initial(response)
        for record_id, response in zip(base.MECHANICS_IDS, initial_raw)
    }
    root_flat = [
        base._clean_text(value, maximum=1_200) for value in root_raw
    ]
    root_answers = [
        root_flat[index : index + base.ROOT_COUNT]
        for index in range(0, len(root_flat), base.ROOT_COUNT)
    ]
    return initials, root_answers


def discrete_interface_version() -> str:
    return "infoquest-discrete-action-causal-1"


def interface_version(cluster_label_max: int) -> str:
    if cluster_label_max == 3:
        return INTERFACE_VERSION
    if cluster_label_max == 7:
        return INTERFACE_VERSION_V2
    raise ValueError("cached partition cluster label maximum must be 3 or 7")


def _synthetic_initial() -> base.InitialPolicy:
    return base.InitialPolicy(
        hypotheses=tuple(
            (
                f"Synthetic hidden context {index} has goal {index}, "
                f"obstacle {index}, and constraint {index}."
            )
            for index in range(1, base.SUPPORT_SIZE + 1)
        ),
        roots=tuple(
            f"What is synthetic detail {index}?"
            for index in range(1, base.ROOT_COUNT + 1)
        ),
    )


def run_serving_gate(
    models: ModelBundle,
    *,
    raw_path: Path,
    cluster_label_max: int = 3,
) -> dict[str, Any]:
    version = interface_version(cluster_label_max)
    raw: dict[str, Any] = {"interface_version": version}
    try:
        seed_message = "Synthetic ambiguous request."
        initial = _synthetic_initial()
        root_answer = "The relevant synthetic detail is value one."
        dynamic_raw = refresh._complete(
            models.generator,
            [
                partition.dynamic_partition_messages(
                    seed_message,
                    initial,
                    0,
                    root_answer,
                    max_cluster_label=cluster_label_max,
                )
            ],
            max_new_tokens=1_300,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "dynamic_partition",
            dynamic_raw,
        )
        dynamic = partition.parse_partition_belief(
            dynamic_raw[0],
            initial,
            0,
            require_fixed_support=False,
            max_cluster_label=cluster_label_max,
        )

        fixed_raw = refresh._complete(
            models.generator,
            [
                partition.fixed_partition_messages(
                    seed_message,
                    initial,
                    0,
                    root_answer,
                    max_cluster_label=cluster_label_max,
                )
            ],
            max_new_tokens=1_300,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "fixed_partition",
            fixed_raw,
        )
        fixed = partition.parse_partition_belief(
            fixed_raw[0],
            initial,
            0,
            require_fixed_support=True,
            max_cluster_label=cluster_label_max,
        )

        simulator_system = (
            "You are a hidden synthetic user. Answer the latest question in "
            "one concise sentence."
        )
        followup_raw = refresh._complete(
            models.simulator,
            [
                base.simulator_followup_messages(
                    simulator_system,
                    initial.roots[0],
                    root_answer,
                    dynamic.selected_question,
                ),
                base.simulator_followup_messages(
                    simulator_system,
                    initial.roots[0],
                    root_answer,
                    fixed.selected_question,
                ),
            ],
            max_new_tokens=160,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "followup_answers",
            followup_raw,
        )
        followup_answers = [
            base._clean_text(value, maximum=1_000) for value in followup_raw
        ]

        fixture = base.WorldFixture(
            fixture_id="SYNTHETIC",
            record_id=-1,
            world=1,
            seed_message=seed_message,
            simulator_system=simulator_system,
            truth_packet={},
            checklist=tuple(f"Checklist item {index}" for index in range(5)),
        )
        checklist_raw = refresh._complete(
            models.checklist_judge,
            [
                base.checklist_judge_messages(
                    fixture,
                    initial,
                    [root_answer] * base.ROOT_COUNT,
                    [dynamic.as_policy()] * base.ROOT_COUNT,
                    [followup_answers[0]] * base.ROOT_COUNT,
                    [fixed.selected_question] * base.ROOT_COUNT,
                    [followup_answers[1]] * base.ROOT_COUNT,
                )
            ],
            max_new_tokens=320,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "checklist_judgment",
            checklist_raw,
        )
        base.parse_checklist_judgment(checklist_raw[0])
        usage = refresh.aggregate_usage(models)
    except Exception as exc:
        base._checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            refresh.aggregate_usage(models),
        ) from exc

    gates = {
        "exact_5_physical_requests": (
            usage["physical_requests"] == EXPECTED_SERVING_REQUESTS
        ),
        "exact_5_http_attempts": (
            usage["http_attempts"] == EXPECTED_SERVING_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_12": (
            usage["adapter_cost_usd"] <= SERVING_MAX_COST_USD
        ),
        "all_stage_parsers_pass": True,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": version,
            "stage": "serving",
            "expected_requests": EXPECTED_SERVING_REQUESTS,
            "synthetic_initial_and_root_answer": True,
            "exact_eig_scorer": True,
            "response_cluster_label_max": cluster_label_max,
            "scientific_endpoint_evaluated": False,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "all_batches_checkpointed_before_parse": True,
            "models": {
                "semantic_support_likelihood": refresh.GENERATOR_MODEL_ID,
                "simulator": refresh.SIMULATOR_MODEL_ID,
                "checklist_judge": refresh.CHECKLIST_JUDGE_MODEL_ID,
            },
        },
        "gates": gates,
        "usage": usage,
    }


def run_mechanics_gate(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    root_answers: Sequence[Sequence[str]],
    models: ModelBundle,
    *,
    raw_path: Path,
    cluster_label_max: int = 3,
) -> dict[str, Any]:
    if len(fixtures) != 6 or len(root_answers) != 6:
        raise ValueError("cached mechanics requires six fixtures")
    version = interface_version(cluster_label_max)
    raw: dict[str, Any] = {
        "interface_version": version,
        "cached_public_sha256": CACHED_PUBLIC_SHA256,
        "cached_raw_sha256": CACHED_RAW_SHA256,
        "private_fixtures": [
            {
                "fixture_id": fixture.fixture_id,
                "seed_message": fixture.seed_message,
                "simulator_system": fixture.simulator_system,
                "checklist": fixture.checklist,
            }
            for fixture in fixtures
        ],
    }
    try:
        dynamic_requests = []
        fixed_requests = []
        for fixture_index, fixture in enumerate(fixtures):
            initial = initials[fixture.record_id]
            if len(root_answers[fixture_index]) != base.ROOT_COUNT:
                raise ValueError("cached root-answer row has wrong length")
            for root_index in range(base.ROOT_COUNT):
                answer = root_answers[fixture_index][root_index]
                dynamic_requests.append(
                    partition.dynamic_partition_messages(
                        fixture.seed_message,
                        initial,
                        root_index,
                        answer,
                        max_cluster_label=cluster_label_max,
                    )
                )
                fixed_requests.append(
                    partition.fixed_partition_messages(
                        fixture.seed_message,
                        initial,
                        root_index,
                        answer,
                        max_cluster_label=cluster_label_max,
                    )
                )

        dynamic_raw = refresh._complete(
            models.generator,
            dynamic_requests,
            max_new_tokens=1_400,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "dynamic_partitions",
            dynamic_raw,
        )
        dynamic_flat = []
        for response_index, response in enumerate(dynamic_raw):
            fixture_index = response_index // base.ROOT_COUNT
            root_index = response_index % base.ROOT_COUNT
            initial = initials[fixtures[fixture_index].record_id]
            dynamic_flat.append(
                partition.parse_partition_belief(
                    response,
                    initial,
                    root_index,
                    require_fixed_support=False,
                    max_cluster_label=cluster_label_max,
                )
            )
        dynamic_beliefs = [
            dynamic_flat[index : index + base.ROOT_COUNT]
            for index in range(0, len(dynamic_flat), base.ROOT_COUNT)
        ]

        fixed_raw = refresh._complete(
            models.generator,
            fixed_requests,
            max_new_tokens=1_400,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "fixed_partitions",
            fixed_raw,
        )
        fixed_flat = []
        for response_index, response in enumerate(fixed_raw):
            fixture_index = response_index // base.ROOT_COUNT
            root_index = response_index % base.ROOT_COUNT
            initial = initials[fixtures[fixture_index].record_id]
            fixed_flat.append(
                partition.parse_partition_belief(
                    response,
                    initial,
                    root_index,
                    require_fixed_support=True,
                    max_cluster_label=cluster_label_max,
                )
            )
        fixed_beliefs = [
            fixed_flat[index : index + base.ROOT_COUNT]
            for index in range(0, len(fixed_flat), base.ROOT_COUNT)
        ]

        followup_requests = []
        for fixture_index, fixture in enumerate(fixtures):
            initial = initials[fixture.record_id]
            for root_index, root in enumerate(initial.roots):
                common = (
                    fixture.simulator_system,
                    root,
                    root_answers[fixture_index][root_index],
                )
                followup_requests.append(
                    base.simulator_followup_messages(
                        *common,
                        dynamic_beliefs[fixture_index][
                            root_index
                        ].selected_question,
                    )
                )
                followup_requests.append(
                    base.simulator_followup_messages(
                        *common,
                        fixed_beliefs[fixture_index][
                            root_index
                        ].selected_question,
                    )
                )
        followup_raw = refresh._complete(
            models.simulator,
            followup_requests,
            max_new_tokens=220,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "followup_answers",
            followup_raw,
        )
        followup_clean = [
            base._clean_text(value, maximum=1_200) for value in followup_raw
        ]
        dynamic_answers: list[list[str]] = []
        fixed_answers: list[list[str]] = []
        cursor = 0
        for _fixture in fixtures:
            dynamic_row = []
            fixed_row = []
            for _root in range(base.ROOT_COUNT):
                dynamic_row.append(followup_clean[cursor])
                fixed_row.append(followup_clean[cursor + 1])
                cursor += 2
            dynamic_answers.append(dynamic_row)
            fixed_answers.append(fixed_row)

        checklist_raw = refresh._complete(
            models.checklist_judge,
            [
                base.checklist_judge_messages(
                    fixture,
                    initials[fixture.record_id],
                    root_answers[fixture_index],
                    [
                        belief.as_policy()
                        for belief in dynamic_beliefs[fixture_index]
                    ],
                    dynamic_answers[fixture_index],
                    [
                        belief.selected_question
                        for belief in fixed_beliefs[fixture_index]
                    ],
                    fixed_answers[fixture_index],
                )
                for fixture_index, fixture in enumerate(fixtures)
            ],
            max_new_tokens=320,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "checklist_judgments",
            checklist_raw,
        )
        judgments = [
            base.parse_checklist_judgment(value) for value in checklist_raw
        ]
        usage = refresh.aggregate_usage(models)
        metrics, fixture_metrics, scientific_gates = partition._public_metrics(
            fixtures,
            initials,
            root_answers,
            dynamic_beliefs,
            fixed_beliefs,
            dynamic_answers,
            fixed_answers,
            judgments,
        )
    except Exception as exc:
        base._checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            refresh.aggregate_usage(models),
        ) from exc

    mechanics_gates = {
        "exact_126_physical_requests": (
            usage["physical_requests"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "exact_126_http_attempts": (
            usage["http_attempts"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_85": (
            usage["adapter_cost_usd"] <= MECHANICS_MAX_COST_USD
        ),
        "exact_6_fixtures_30_root_cells": (
            metrics["fixtures"] == 6
            and metrics["root_world_cells"] == 30
        ),
    }
    gates = {**mechanics_gates, **scientific_gates}
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": version,
            "stage": "mechanics",
            "mechanics_ids": list(base.MECHANICS_IDS),
            "cached_public_sha256": CACHED_PUBLIC_SHA256,
            "cached_raw_sha256": CACHED_RAW_SHA256,
            "cached_common_history": True,
            "expected_requests": EXPECTED_MECHANICS_REQUESTS,
            "exact_eig_scorer": True,
            "shared_discrete_action_bank": True,
            "response_clusters_per_action": cluster_label_max + 1,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "all_batches_checkpointed_before_parse": True,
            "opportunity_or_later_split_read": False,
            "development_only_if_passed": True,
            "models": {
                "semantic_support_likelihood": refresh.GENERATOR_MODEL_ID,
                "simulator": refresh.SIMULATOR_MODEL_ID,
                "checklist_judge": refresh.CHECKLIST_JUDGE_MODEL_ID,
            },
        },
        "metrics": metrics,
        "fixture_metrics": fixture_metrics,
        "gates": gates,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("serving", "mechanics"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cached-raw", type=Path)
    parser.add_argument("--cached-public", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cluster-label-max",
        type=int,
        choices=(3, 7),
        default=3,
    )
    args = parser.parse_args()

    fixtures, public_fixture = base.build_fixtures(
        settings_path=args.settings,
        baseline_path=args.baseline,
    )
    cached: tuple[dict[int, base.InitialPolicy], list[list[str]]] | None = None
    if args.stage == "mechanics":
        if args.cached_raw is None or args.cached_public is None:
            parser.error("--cached-raw and --cached-public require mechanics")
        cached = load_cached_histories(
            fixtures,
            raw_path=args.cached_raw,
            public_path=args.cached_public,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = (
        0.07 if args.stage == "serving" else 0.63
    )
    config.openrouter_run_budget_usd = (
        SERVING_MAX_COST_USD
        if args.stage == "serving"
        else MECHANICS_MAX_COST_USD
    )
    config.openrouter_concurrency = (
        EXPECTED_SERVING_REQUESTS
        if args.stage == "serving"
        else 30
    )
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = 1_500
    config.log_path = args.output_dir / "run.log"
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    models = (
        ModelBundle(
            generator=CachedDeterministicGenerator("generator"),
            simulator=base.DeterministicFixtureModel("simulator"),
            checklist_judge=base.DeterministicFixtureModel(
                "checklist_judge"
            ),
        )
        if args.dry_run
        else refresh._build_models(config)
    )

    try:
        if args.stage == "serving":
            result = run_serving_gate(
                models,
                raw_path=raw_path,
                cluster_label_max=args.cluster_label_max,
            )
        else:
            assert cached is not None
            result = run_mechanics_gate(
                fixtures,
                cached[0],
                cached[1],
                models,
                raw_path=raw_path,
                cluster_label_max=args.cluster_label_max,
            )
        result["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        result["protocol"]["fixture_sha256"] = public_fixture[
            "private_fixture_sha256"
        ]
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": interface_version(args.cluster_label_max),
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        base._checkpoint(args.output_dir / "GATE_FAILURE.json", failure)
        raise

    output_name = (
        "SERVING.json" if args.stage == "serving" else "MECHANICS.json"
    )
    base._checkpoint(args.output_dir / output_name, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gates"]["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
