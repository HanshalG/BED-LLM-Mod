#!/usr/bin/env python3
"""Run the DiscoverLLM coarse-tier non-myopic mechanics smoke."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts import discoverllm_priority_world_mechanics as cardinal
from scripts import discoverllm_priority_world_ordinal_serving as ordinal
from scripts import discoverllm_priority_world_tier_serving as tier


INTERFACE_VERSION = "discoverllm-priority-world-tier-mechanics-1"
MODEL_ID = "openai/gpt-5.4"
TASK_IDS = tier.RESERVED_MECHANICS_TASK_IDS
EXPECTED_TASKS = 3
EXPECTED_REQUESTS = 15
MAX_TRANSPORT_RETRIES = 4
PROJECTED_COST_USD = 0.25
MAX_COST_USD = 0.50
MIN_DYNAMIC_RANGE_NATS = 0.01
MIN_ROOT_SACRIFICE_NATS = 0.005
MIN_TRUTH_TOP_TIER_RATE = 0.60


class MechanicsExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _load_tasks(
    paths: dict[str, Path],
    manifest_path: Path,
) -> list[cardinal.MechanicsTask]:
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    development_ids = tuple(frozen["splits"]["development"]["artifact_ids"])
    expected_prefix = (
        ordinal.SERVING_TASK_ID,
        tier.SERVING_TASK_ID,
        *TASK_IDS,
    )
    if development_ids[: len(expected_prefix)] != expected_prefix:
        raise ValueError("DiscoverLLM tier mechanics reservation changed")
    tasks = [
        ordinal._load_task_by_id(paths, manifest_path, task_id)
        for task_id in TASK_IDS
    ]
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError("DiscoverLLM tier mechanics task count changed")
    return tasks


def _posterior(
    prior: Sequence[float],
    assignments: dict[str, str],
    weights: dict[str, float],
) -> list[float]:
    return cardinal._normalize(
        [
            prior[index] * weights[assignments[world]]
            for index, world in enumerate(cardinal.WORLD_LABELS)
        ]
    )


def _analyze_task(
    mapping: dict[str, tuple[int, ...]],
    root_tiers: dict[str, dict[str, str]],
    followup_tiers: dict[str, dict[str, str]],
    weights: dict[str, float],
) -> dict[str, Any]:
    prior = [0.25] * 4
    start_entropy = cardinal._entropy(prior)
    actions: dict[str, Any] = {}
    for action in cardinal.ACTION_LABELS:
        root_posteriors = []
        terminal_posteriors = []
        truth_indices = []
        for observation_index, observation in enumerate(
            cardinal.OBSERVATION_LABELS
        ):
            key = f"{action}_{observation}"
            root_posterior = _posterior(
                prior,
                root_tiers[key],
                weights,
            )
            terminal_posterior = _posterior(
                root_posterior,
                followup_tiers[key],
                weights,
            )
            root_posteriors.append(root_posterior)
            terminal_posteriors.append(terminal_posterior)
            truth_indices.append(mapping[action][observation_index])
        actions[action] = {
            "root_eig": start_entropy
            - statistics.mean(map(cardinal._entropy, root_posteriors)),
            "depth_two_eig": start_entropy
            - statistics.mean(map(cardinal._entropy, terminal_posteriors)),
            "root_truth_log_posterior": statistics.mean(
                math.log(max(posterior[truth], 1e-300))
                for posterior, truth in zip(root_posteriors, truth_indices)
            ),
            "terminal_truth_log_posterior": statistics.mean(
                math.log(max(posterior[truth], 1e-300))
                for posterior, truth in zip(
                    terminal_posteriors,
                    truth_indices,
                )
            ),
            "terminal_map_accuracy": statistics.mean(
                posterior[truth] == max(posterior)
                for posterior, truth in zip(
                    terminal_posteriors,
                    truth_indices,
                )
            ),
        }
    myopic = max(
        cardinal.ACTION_LABELS,
        key=lambda action: actions[action]["root_eig"],
    )
    nonmyopic = max(
        cardinal.ACTION_LABELS,
        key=lambda action: actions[action]["depth_two_eig"],
    )
    return {
        "actions": actions,
        "myopic_action": myopic,
        "nonmyopic_action": nonmyopic,
        "root_changed": myopic != nonmyopic,
        "delayed_reversal": (
            myopic != nonmyopic
            and actions[nonmyopic]["root_eig"]
            <= (
                actions[myopic]["root_eig"]
                - MIN_ROOT_SACRIFICE_NATS
            )
            and actions[nonmyopic]["terminal_truth_log_posterior"]
            > actions[myopic]["terminal_truth_log_posterior"]
        ),
    }


def _tier_calibration(
    mappings: Sequence[dict[str, tuple[int, ...]]],
    tier_outputs: Sequence[dict[str, dict[str, str]]],
    weights: dict[str, float],
) -> dict[str, float]:
    top_tier = []
    strict_top = []
    nonuniform = []
    truth_advantages = []
    for mapping, task_tiers in zip(mappings, tier_outputs):
        for action in cardinal.ACTION_LABELS:
            for observation_index, observation in enumerate(
                cardinal.OBSERVATION_LABELS
            ):
                key = f"{action}_{observation}"
                values = [
                    weights[task_tiers[key][world]]
                    for world in cardinal.WORLD_LABELS
                ]
                truth_index = mapping[action][observation_index]
                truth_value = values[truth_index]
                distractors = [
                    value
                    for index, value in enumerate(values)
                    if index != truth_index
                ]
                top_tier.append(truth_value == max(values))
                strict_top.append(truth_value > max(distractors))
                nonuniform.append(len(set(values)) > 1)
                truth_advantages.append(
                    truth_value - statistics.mean(distractors)
                )
    return {
        "truth_top_tier_rate": statistics.mean(top_tier),
        "truth_strict_top_rate": statistics.mean(strict_top),
        "nonuniform_rate": statistics.mean(nonuniform),
        "mean_truth_weight_advantage": statistics.mean(
            truth_advantages
        ),
    }


def _aggregate(
    analyses: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    dynamic_tasks = sum(
        abs(
            analysis["actions"]["A"]["root_eig"]
            - analysis["actions"]["B"]["root_eig"]
        )
        >= MIN_DYNAMIC_RANGE_NATS
        and abs(
            analysis["actions"]["A"]["depth_two_eig"]
            - analysis["actions"]["B"]["depth_two_eig"]
        )
        >= MIN_DYNAMIC_RANGE_NATS
        for analysis in analyses
    )
    nonmyopic_terminal = [
        analysis["actions"][analysis["nonmyopic_action"]][
            "terminal_truth_log_posterior"
        ]
        for analysis in analyses
    ]
    myopic_terminal = [
        analysis["actions"][analysis["myopic_action"]][
            "terminal_truth_log_posterior"
        ]
        for analysis in analyses
    ]
    random_terminal = [
        statistics.mean(
            analysis["actions"][action][
                "terminal_truth_log_posterior"
            ]
            for action in cardinal.ACTION_LABELS
        )
        for analysis in analyses
    ]
    nonmyopic_map = [
        analysis["actions"][analysis["nonmyopic_action"]][
            "terminal_map_accuracy"
        ]
        for analysis in analyses
    ]
    myopic_map = [
        analysis["actions"][analysis["myopic_action"]][
            "terminal_map_accuracy"
        ]
        for analysis in analyses
    ]
    root_values = [
        analysis["actions"][action]["root_eig"]
        for analysis in analyses
        for action in cardinal.ACTION_LABELS
    ]
    depth_values = [
        analysis["actions"][action]["depth_two_eig"]
        for analysis in analyses
        for action in cardinal.ACTION_LABELS
    ]
    terminal_values = [
        analysis["actions"][action]["terminal_truth_log_posterior"]
        for analysis in analyses
        for action in cardinal.ACTION_LABELS
    ]
    root_rho = cardinal._spearman(root_values, terminal_values)
    depth_rho = cardinal._spearman(depth_values, terminal_values)
    return {
        "dynamic_tasks": dynamic_tasks,
        "root_changes": sum(
            analysis["root_changed"] for analysis in analyses
        ),
        "strict_delayed_reversals": sum(
            analysis["delayed_reversal"] for analysis in analyses
        ),
        "mean_nonmyopic_minus_myopic_terminal_truth_log_posterior": (
            statistics.mean(
                nonmyopic - myopic
                for nonmyopic, myopic in zip(
                    nonmyopic_terminal,
                    myopic_terminal,
                )
            )
        ),
        "mean_nonmyopic_minus_random_terminal_truth_log_posterior": (
            statistics.mean(
                nonmyopic - random
                for nonmyopic, random in zip(
                    nonmyopic_terminal,
                    random_terminal,
                )
            )
        ),
        "mean_nonmyopic_minus_myopic_terminal_map_accuracy": (
            statistics.mean(
                nonmyopic - myopic
                for nonmyopic, myopic in zip(
                    nonmyopic_map,
                    myopic_map,
                )
            )
        ),
        "root_eig_terminal_truth_log_spearman": root_rho,
        "depth_two_eig_terminal_truth_log_spearman": depth_rho,
    }


def _robust_reversal_count(
    analyses_by_weights: Sequence[Sequence[dict[str, Any]]],
) -> int:
    return sum(
        all(
            analyses[task_index]["delayed_reversal"]
            and analyses[task_index]["myopic_action"]
            == analyses_by_weights[0][task_index]["myopic_action"]
            and analyses[task_index]["nonmyopic_action"]
            == analyses_by_weights[0][task_index]["nonmyopic_action"]
            for analyses in analyses_by_weights
        )
        for task_index in range(EXPECTED_TASKS)
    )


def _mechanics_gates(
    usage: dict[str, Any],
    summary: dict[str, Any],
    root_calibration: dict[str, float],
    followup_calibration: dict[str, float],
    robust_reversals: int,
) -> dict[str, bool]:
    requests = int(usage.get("adapter_requests", -1))
    attempts = int(usage.get("http_attempts", -1))
    retries = int(usage.get("retry_count", -1))
    gates = {
        "exact_15_logical_requests": requests == EXPECTED_REQUESTS,
        "bounded_transport_retries": 0 <= retries <= MAX_TRANSPORT_RETRIES,
        "http_attempts_match_requests_plus_retries": (
            attempts == requests + retries
        ),
        "zero_reasoning_tokens": int(
            usage.get("adapter_reasoning_tokens", 0)
        )
        == 0,
        "zero_forced_exits": int(usage.get("forced_exits", 0)) == 0,
        "cost_at_most_0_50": float(
            usage.get("adapter_cost_usd", float("inf"))
        )
        <= MAX_COST_USD,
        "root_truth_top_tier_rate_at_least_0_60": (
            root_calibration["truth_top_tier_rate"]
            >= MIN_TRUTH_TOP_TIER_RATE
        ),
        "followup_truth_top_tier_rate_at_least_0_60": (
            followup_calibration["truth_top_tier_rate"]
            >= MIN_TRUTH_TOP_TIER_RATE
        ),
        "root_truth_weight_advantage_positive": (
            root_calibration["mean_truth_weight_advantage"] > 0
        ),
        "followup_truth_weight_advantage_positive": (
            followup_calibration["mean_truth_weight_advantage"] > 0
        ),
        "at_least_two_dynamic_tasks": summary["dynamic_tasks"] >= 2,
        "at_least_one_root_change": summary["root_changes"] >= 1,
        "at_least_one_strict_delayed_reversal": (
            summary["strict_delayed_reversals"] >= 1
        ),
        "nonmyopic_mean_terminal_truth_log_above_myopic": (
            summary[
                "mean_nonmyopic_minus_myopic_terminal_truth_log_posterior"
            ]
            > 0
        ),
        "nonmyopic_mean_terminal_truth_log_above_random": (
            summary[
                "mean_nonmyopic_minus_random_terminal_truth_log_posterior"
            ]
            > 0
        ),
        "nonmyopic_mean_terminal_map_no_lower": (
            summary[
                "mean_nonmyopic_minus_myopic_terminal_map_accuracy"
            ]
            >= 0
        ),
        "depth_score_terminal_rho_at_least_0_20": (
            summary["depth_two_eig_terminal_truth_log_spearman"] >= 0.20
        ),
        "depth_rho_advantage_at_least_0_10": (
            summary["depth_two_eig_terminal_truth_log_spearman"]
            >= summary["root_eig_terminal_truth_log_spearman"] + 0.10
        ),
        "at_least_one_weight_robust_delayed_reversal": (
            robust_reversals >= 1
        ),
    }
    gates["all_pass"] = all(gates.values())
    return gates


def run_mechanics(
    config: Config,
    *,
    paths: dict[str, Path],
    manifest_path: Path,
    raw_path: Path,
    model: ordinal.ChatModel,
) -> dict[str, Any]:
    tasks = _load_tasks(paths, manifest_path)
    raw: dict[str, Any] = {"task_ids": [task.key for task in tasks]}
    try:
        root_raw = model.chat_complete_messages_batched(
            [cardinal._root_observation_messages(task) for task in tasks],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=2_200,
        )
        raw["root_observations"] = root_raw
        ordinal._checkpoint(raw_path, raw)
        root_outputs = [
            cardinal._parse_text_object(
                response,
                cardinal._root_observation_keys(),
                maximum_length=800,
            )
            for response in root_raw
        ]
        shuffled = [
            cardinal._shuffle_root_observations(task, output)
            for task, output in zip(tasks, root_outputs)
        ]
        observations = [item[0] for item in shuffled]
        mappings = [item[1] for item in shuffled]

        root_tier_raw = model.chat_complete_messages_batched(
            [
                tier._root_tier_messages(task, task_observations)
                for task, task_observations in zip(tasks, observations)
            ],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=320,
        )
        raw["root_tiers"] = root_tier_raw
        ordinal._checkpoint(raw_path, raw)
        root_tiers = [
            tier._parse_tiers(response, cardinal._branch_keys())
            for response in root_tier_raw
        ]

        followup_raw = model.chat_complete_messages_batched(
            [
                cardinal._followup_messages(task, task_observations)
                for task, task_observations in zip(tasks, observations)
            ],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=2_200,
        )
        raw["followups"] = followup_raw
        ordinal._checkpoint(raw_path, raw)
        followups = [
            cardinal._parse_text_object(
                response,
                cardinal._branch_keys(),
                maximum_length=1_000,
            )
            for response in followup_raw
        ]

        second_raw = model.chat_complete_messages_batched(
            [
                cardinal._followup_observation_messages(
                    task,
                    task_observations,
                    mapping,
                    task_followups,
                )
                for task, task_observations, mapping, task_followups in zip(
                    tasks,
                    observations,
                    mappings,
                    followups,
                )
            ],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=2_200,
        )
        raw["followup_observations"] = second_raw
        ordinal._checkpoint(raw_path, raw)
        second_observations = [
            cardinal._parse_text_object(
                response,
                cardinal._branch_keys(),
                maximum_length=800,
            )
            for response in second_raw
        ]

        followup_tier_raw = model.chat_complete_messages_batched(
            [
                tier._followup_tier_messages(
                    task,
                    task_observations,
                    task_followups,
                    task_second_observations,
                )
                for (
                    task,
                    task_observations,
                    task_followups,
                    task_second_observations,
                ) in zip(
                    tasks,
                    observations,
                    followups,
                    second_observations,
                )
            ],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=320,
        )
        raw["followup_tiers"] = followup_tier_raw
        ordinal._checkpoint(raw_path, raw)
        followup_tiers = [
            tier._parse_tiers(response, cardinal._branch_keys())
            for response in followup_tier_raw
        ]
        usage = model.usage_snapshot()
    except Exception as exc:
        ordinal._checkpoint(raw_path, raw)
        raise MechanicsExecutionError(
            f"{type(exc).__name__}: {exc}",
            model.usage_snapshot(),
        ) from exc

    weight_maps = (
        tier.TIER_WEIGHTS,
        *tier.SENSITIVITY_TIER_WEIGHTS,
    )
    analyses_by_weights = [
        [
            _analyze_task(
                mapping,
                task_root_tiers,
                task_followup_tiers,
                weights,
            )
            for mapping, task_root_tiers, task_followup_tiers in zip(
                mappings,
                root_tiers,
                followup_tiers,
            )
        ]
        for weights in weight_maps
    ]
    primary_analyses = analyses_by_weights[0]
    summary = _aggregate(primary_analyses)
    root_calibration = _tier_calibration(
        mappings,
        root_tiers,
        tier.TIER_WEIGHTS,
    )
    followup_calibration = _tier_calibration(
        mappings,
        followup_tiers,
        tier.TIER_WEIGHTS,
    )
    robust_reversals = _robust_reversal_count(analyses_by_weights)
    gates = _mechanics_gates(
        usage,
        summary,
        root_calibration,
        followup_calibration,
        robust_reversals,
    )
    public_tasks = [
        {
            "task_id": task.key,
            "myopic_action": analysis["myopic_action"],
            "nonmyopic_action": analysis["nonmyopic_action"],
            "root_changed": analysis["root_changed"],
            "delayed_reversal": analysis["delayed_reversal"],
            "actions": analysis["actions"],
            "sensitivity_choices": [
                {
                    "weights": weights,
                    "myopic_action": weight_analyses[index][
                        "myopic_action"
                    ],
                    "nonmyopic_action": weight_analyses[index][
                        "nonmyopic_action"
                    ],
                    "delayed_reversal": weight_analyses[index][
                        "delayed_reversal"
                    ],
                }
                for weights, weight_analyses in zip(
                    weight_maps,
                    analyses_by_weights,
                )
            ],
        }
        for index, (task, analysis) in enumerate(
            zip(tasks, primary_analyses)
        )
    ]
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "reasoning_requested": False,
            "temperature": 0.0,
            "task_ids": list(TASK_IDS),
            "manifest_sha256": cardinal.MANIFEST_SHA256,
            "observation_shuffle_seed": cardinal.OBSERVATION_SEED,
            "world_presentation_seed": tier.WORLD_PRESENTATION_SEED,
            "tier_weights": tier.TIER_WEIGHTS,
            "sensitivity_tier_weights": list(
                tier.SENSITIVITY_TIER_WEIGHTS
            ),
            "expected_logical_requests": EXPECTED_REQUESTS,
            "maximum_transport_retries": MAX_TRANSPORT_RETRIES,
            "semantic_reissues_allowed": False,
            "released_scores_read": False,
            "released_winner_labels_read": False,
            "truth_map_exposed_to_likelihood_scorer": False,
            "truth_map_exposed_to_policy": False,
            "semantic_content_emitted": False,
        },
        "tasks": public_tasks,
        "calibration": {
            "root": root_calibration,
            "followup": followup_calibration,
        },
        "summary": {
            **summary,
            "weight_robust_delayed_reversals": robust_reversals,
        },
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel(ordinal.DeterministicFixtureModel):
    @staticmethod
    def _true_world(text: str) -> str:
        return next(
            world for world in cardinal.WORLD_LABELS if world in text
        )

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        stage = self._stage(batch_messages[0])
        if stage not in {"ROOT_TIERS", "FOLLOWUP_TIERS"}:
            return super().chat_complete_messages_batched(
                batch_messages,
                temperature,
                block_size,
                max_new_tokens,
            )
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            payload = json.loads(messages[-1]["content"])
            lines = []
            for key in payload["output_keys"]:
                action, observation = key.split("_")
                if stage == "ROOT_TIERS":
                    text = payload["observed_user_feedback"][action][
                        observation
                    ]
                else:
                    text = payload["branches"][key]["new_user_feedback"]
                truth = self._true_world(text)
                truth_index = cardinal.WORLD_LABELS.index(truth)
                distractor = cardinal.WORLD_LABELS[(truth_index + 1) % 4]
                assignments = {}
                for world in cardinal.WORLD_LABELS:
                    if stage == "FOLLOWUP_TIERS" and action == "A":
                        assignments[world] = "M"
                    elif world == truth:
                        assignments[world] = "H"
                    elif (
                        stage == "ROOT_TIERS"
                        and action == "B"
                        and world == distractor
                    ):
                        assignments[world] = "H"
                    else:
                        assignments[world] = "L"
                lines.append(
                    key
                    + "|"
                    + ",".join(
                        f"{world}:{assignments[world]}"
                        for world in cardinal.WORLD_LABELS
                    )
                )
            responses.append("\n".join(lines))
        self.requests += len(responses)
        return responses


def _build_model(config: Config) -> ordinal.ChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("DiscoverLLM tier mechanics config selects wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--creative-writing", type=Path, required=True)
    parser.add_argument("--technical-writing", type=Path, required=True)
    parser.add_argument("--svg-drawing", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = EXPECTED_TASKS
    config.openrouter_max_retries = MAX_TRANSPORT_RETRIES
    config.openrouter_max_output_tokens = 2_200
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ordinal.ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        result = run_mechanics(
            config,
            paths={
                "creative_writing": args.creative_writing,
                "technical_writing": args.technical_writing,
                "svg_drawing": args.svg_drawing,
            },
            manifest_path=args.manifest,
            raw_path=raw_path,
            model=model,
        )
        result["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, MechanicsExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        ordinal._checkpoint(args.output_dir / "MECHANICS_FAILURE.json", failure)
        raise
    output_path = args.output_dir / "MECHANICS.json"
    ordinal._checkpoint(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output_path),
                "calibration": result["calibration"],
                "summary": result["summary"],
                "gates": result["gates"],
                "usage": result["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
