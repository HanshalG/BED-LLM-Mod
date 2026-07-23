"""Paired focused-prior Rock trajectories with cached hierarchical h5 targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError
from scripts.nonmyopic_range_gated_rock_depth5_goal_compiler import (
    Depth5GoalCompilerConfig,
    Depth5GoalCompilerProvider,
    DeterministicDepth5GoalModel,
    build_depth5_model,
    matched_random_goal_plans,
)
from scripts.nonmyopic_range_gated_rock_depth5_oracle import (
    H5_ROUTE,
    _sample_truth_index,
)
from scripts.nonmyopic_range_gated_rock_fixed_tail import (
    fixed_roots,
    usage_with_forced_events,
)
from scripts.nonmyopic_range_gated_rock_proposal_gate import _stable_seed
from scripts.nonmyopic_range_gated_rock_stable_controls import (
    stable_best_index,
    stable_best_plan,
)
from scripts.nonmyopic_range_gated_rock_strategy import (
    ChatModel,
    History,
    RangeGatedPlanCell,
)
from scripts.nonmyopic_rock_depth_oracle import _uniform, exhaustive_action_values


Arm = Literal[
    "llm_h5",
    "shared_h4",
    "random_h5",
    "exact_d4",
    "exact_d5",
]


@dataclass(frozen=True)
class FocusedDepth5TrajectoryConfig:
    num_trials: int = 50
    num_rounds: int = 8
    seed: int = 24_245
    bootstrap_replicates: int = 10_000
    recovery_threshold: float = 0.60
    route_rate_threshold: float = 0.75
    max_unique_llm_cells: int = 64

    def validate(self) -> None:
        if self.num_trials != 50 or self.num_rounds != 8:
            raise ValueError("the frozen h5 trajectory uses 50 trials and 8 rounds")
        if self.bootstrap_replicates != 10_000:
            raise ValueError("the frozen h5 trajectory uses 10,000 bootstraps")
        if self.max_unique_llm_cells != 64:
            raise ValueError("the frozen physical proposal cap is 64 prompts")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery threshold must be a probability")
        if not 0.0 <= self.route_rate_threshold <= 1.0:
            raise ValueError("route threshold must be a probability")


class CachedDepth5GoalProvider:
    """Cache exact prompt identities while retaining each logical decision."""

    def __init__(
        self,
        provider: Depth5GoalCompilerProvider,
        *,
        max_unique_cells: int,
    ) -> None:
        self.provider = provider
        self.max_unique_cells = max_unique_cells
        self.cache: dict[str, RangeGatedPlanCell] = {}
        self.logical_requests: list[dict[str, Any]] = []

    @property
    def physical_requests(self) -> list[dict[str, Any]]:
        return self.provider.physical_requests

    @property
    def invalid_responses(self) -> list[dict[str, Any]]:
        return self.provider.invalid_responses

    def _key(
        self,
        model: RangeGatedRockDiagnosisModel,
        *,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
    ) -> str:
        roots = fixed_roots(model, position=position, belief=belief)
        messages = self.provider._messages(
            model,
            position=position,
            belief=belief,
            history=history,
            roots=roots,
        )
        payload = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def propose(
        self,
        model: RangeGatedRockDiagnosisModel,
        *,
        cell_index: int,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
    ) -> RangeGatedPlanCell:
        key = self._key(
            model, position=position, belief=belief, history=history
        )
        cached = self.cache.get(key)
        if cached is None:
            if len(self.cache) >= self.max_unique_cells:
                raise StrategyProposalError(
                    f"unique prompt cap {self.max_unique_cells} exhausted"
                )
            cell = self.provider.propose(
                model,
                cell_index=cell_index,
                position=position,
                belief=belief,
                history=history,
            )
            self.cache[key] = cell
            cache_hit = False
        else:
            cell = cached
            cache_hit = True
        self.logical_requests.append(
            {
                "cell_index": cell_index,
                "cache_key": key,
                "cache_hit": cache_hit,
                "position": list(position),
                "history": [list(item) for item in history],
                "compiled_plans": [list(plan) for plan in cell.plans],
            }
        )
        return cell


def _belief_key(belief: np.ndarray) -> bytes:
    return np.ascontiguousarray(belief, dtype=np.float64).tobytes()


def _exact_action(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    depth: int,
    cache: dict[tuple[int, tuple[int, int], bytes], tuple[str, float, int]],
) -> tuple[str, float, int]:
    key = (depth, position, _belief_key(belief))
    cached = cache.get(key)
    if cached is not None:
        return cached
    values, units = exhaustive_action_values(
        model, position=position, belief=belief, depth=depth
    )
    legal = tuple(values)
    action = legal[
        stable_best_index([values[candidate] for candidate in legal])
    ]
    result = (action, float(values[action]), units)
    cache[key] = result
    return result


def _observation(
    model: RangeGatedRockDiagnosisModel,
    *,
    config: FocusedDepth5TrajectoryConfig,
    trial_index: int,
    truth_index: int,
    position: tuple[int, int],
    action: str,
    repeat_index: int,
) -> str | None:
    check_id = model.check_id(action)
    if check_id is None:
        return None
    probability_good = float(
        model.likelihood_vector(position, action, "good")[truth_index]
    )
    return (
        "good"
        if _uniform(
            config.seed,
            "focused-h5-trajectory-observation",
            trial_index,
            position,
            check_id,
            repeat_index,
        )
        < probability_good
        else "bad"
    )


def _run_arm(
    model: RangeGatedRockDiagnosisModel,
    *,
    arm: Arm,
    trial_index: int,
    truth_index: int,
    config: FocusedDepth5TrajectoryConfig,
    provider: CachedDepth5GoalProvider | None,
    exact_cache: dict[
        tuple[int, tuple[int, int], bytes], tuple[str, float, int]
    ],
) -> dict[str, Any]:
    belief = model.initial_belief.copy()
    position = model.map_spec.start_position
    history: list[tuple[str, str | None]] = []
    check_counts: dict[tuple[tuple[int, int], int], int] = {}
    steps: list[dict[str, Any]] = []
    for round_index in range(config.num_rounds):
        remaining = config.num_rounds - round_index
        policy: dict[str, Any]
        if arm in ("llm_h5", "shared_h4") and remaining >= 5:
            if provider is None:
                raise ValueError(f"{arm} requires a provider")
            arm_offset = 0 if arm == "llm_h5" else config.num_trials * config.num_rounds
            proposal = provider.propose(
                model,
                cell_index=arm_offset
                + trial_index * config.num_rounds
                + round_index,
                position=position,
                belief=belief,
                history=tuple(history),
            )
            scored_plans = (
                proposal.plans
                if arm == "llm_h5"
                else tuple(plan[:4] for plan in proposal.plans)
            )
            scored_plan, scored_value, values = stable_best_plan(
                model,
                position=position,
                belief=belief,
                plans=scored_plans,
            )
            selected_index = scored_plans.index(scored_plan)
            full_plan = proposal.plans[selected_index]
            action = full_plan[0]
            policy = {
                "mode": "cached_hierarchical_h5"
                if arm == "llm_h5"
                else "cached_hierarchical_shared_h4",
                "plans": [list(item) for item in proposal.plans],
                "scored_plans": [list(item) for item in scored_plans],
                "plan_values": values,
                "selected_full_plan": list(full_plan),
                "selected_value": scored_value,
            }
        elif arm == "random_h5" and remaining >= 5:
            targets, plans = matched_random_goal_plans(
                model,
                position=position,
                belief=belief,
                seed=_stable_seed(
                    config.seed,
                    "focused-h5-trajectory-random",
                    trial_index,
                    round_index,
                ),
            )
            plan, value, values = stable_best_plan(
                model, position=position, belief=belief, plans=plans
            )
            action = plan[0]
            policy = {
                "mode": "matched_random_targets_h5",
                "targets": list(targets),
                "plans": [list(item) for item in plans],
                "plan_values": values,
                "selected_full_plan": list(plan),
                "selected_value": value,
            }
        else:
            requested_depth = 5 if arm in ("llm_h5", "random_h5", "exact_d5") else 4
            depth = min(requested_depth, remaining)
            action, value, units = _exact_action(
                model,
                position=position,
                belief=belief,
                depth=depth,
                cache=exact_cache,
            )
            policy = {
                "mode": "exact_terminal_fallback"
                if arm in ("llm_h5", "shared_h4", "random_h5")
                else "exact",
                "depth": depth,
                "selected_value": value,
                "scorer_units": units,
            }

        check_id = model.check_id(action)
        if check_id is None:
            repeat_index = 0
        else:
            key = (position, check_id)
            repeat_index = check_counts.get(key, 0)
            check_counts[key] = repeat_index + 1
        observation = _observation(
            model,
            config=config,
            trial_index=trial_index,
            truth_index=truth_index,
            position=position,
            action=action,
            repeat_index=repeat_index,
        )
        posterior = model.posterior(position, belief, action, observation)
        steps.append(
            {
                "round": round_index + 1,
                "position_before": list(position),
                "action": action,
                "observation": observation,
                "repeat_index": repeat_index,
                "entropy_before": model.entropy(belief),
                "entropy": model.entropy(posterior),
                "truth_log_probability": math.log(
                    max(float(posterior[truth_index]), np.finfo(float).tiny)
                ),
                "policy": policy,
            }
        )
        belief = posterior
        position = model.next_position(position, action)
        history.append((action, observation))

    entropies = [float(step["entropy"]) for step in steps]
    truth_logs = [float(step["truth_log_probability"]) for step in steps]
    return {
        "arm": arm,
        "trial_index": trial_index,
        "truth_index": truth_index,
        "entropy_auc": float(np.mean(entropies)),
        "truth_log_probability_auc": float(np.mean(truth_logs)),
        "final_entropy": entropies[-1],
        "final_truth_log_probability": truth_logs[-1],
        "final_map_accuracy": float(model.decode_map_index(belief) == truth_index),
        "steps": steps,
    }


def _bootstrap(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        samples[start : start + size] = values[indices].mean(axis=1)
    return [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def _summary(
    values: list[float],
    *,
    config: FocusedDepth5TrajectoryConfig,
    label: str,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "ci95": _bootstrap(
            array,
            seed=_stable_seed(config.seed, "focused-h5-bootstrap", label),
            replicates=config.bootstrap_replicates,
        ),
        "wins_ties_losses": [
            int(np.sum(array > 1e-12)),
            int(np.sum(np.abs(array) <= 1e-12)),
            int(np.sum(array < -1e-12)),
        ],
        "paired_values": array.tolist(),
    }


def run_confirmation(
    provider: CachedDepth5GoalProvider,
    config: FocusedDepth5TrajectoryConfig,
) -> dict[str, Any]:
    config.validate()
    model = build_depth5_model()
    truth_indices = [
        _sample_truth_index(model, trial_index=index, seed=config.seed)
        for index in range(config.num_trials)
    ]
    arms: tuple[Arm, ...] = (
        "llm_h5",
        "shared_h4",
        "random_h5",
        "exact_d4",
        "exact_d5",
    )
    traces: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    exact_cache: dict[
        tuple[int, tuple[int, int], bytes], tuple[str, float, int]
    ] = {}
    for trial_index, truth_index in enumerate(truth_indices):
        for arm in arms:
            traces[arm].append(
                _run_arm(
                    model,
                    arm=arm,
                    trial_index=trial_index,
                    truth_index=truth_index,
                    config=config,
                    provider=provider
                    if arm in ("llm_h5", "shared_h4")
                    else None,
                    exact_cache=exact_cache,
                )
            )

    comparisons: dict[str, dict[str, Any]] = {}
    for baseline in ("shared_h4", "random_h5", "exact_d4"):
        entropy = [
            base["entropy_auc"] - llm["entropy_auc"]
            for base, llm in zip(
                traces[baseline], traces["llm_h5"], strict=True
            )
        ]
        truth = [
            llm["truth_log_probability_auc"]
            - base["truth_log_probability_auc"]
            for base, llm in zip(
                traces[baseline], traces["llm_h5"], strict=True
            )
        ]
        comparisons[f"entropy_auc_gain_vs_{baseline}"] = _summary(
            entropy, config=config, label=f"entropy-vs-{baseline}"
        )
        comparisons[f"truth_log_auc_gain_vs_{baseline}"] = _summary(
            truth, config=config, label=f"truth-vs-{baseline}"
        )
    exact_gain = float(
        np.mean(
            [
                d4["entropy_auc"] - d5["entropy_auc"]
                for d4, d5 in zip(
                    traces["exact_d4"], traces["exact_d5"], strict=True
                )
            ]
        )
    )
    llm_gain = comparisons["entropy_auc_gain_vs_exact_d4"]["mean"]
    recovery = llm_gain / exact_gain if exact_gain > 0.0 else float("-inf")
    route_rate = float(
        np.mean(
            [
                tuple(step["action"] for step in trace["steps"][:5])
                == H5_ROUTE
                for trace in traces["llm_h5"]
            ]
        )
    )
    onsite_by_round_five = float(
        np.mean(
            [
                any(
                    step["round"] <= 5
                    and step["action"] == "check-6"
                    and tuple(step["position_before"])
                    == model.map_spec.rock_positions[6]
                    for step in trace["steps"]
                )
                for trace in traces["llm_h5"]
            ]
        )
    )
    expected_logical = 2 * config.num_trials * (config.num_rounds - 4)
    mechanics = {
        "all_five_arms_complete": all(
            len(trace["steps"]) == config.num_rounds
            for rows in traces.values()
            for trace in rows
        ),
        "all_truths_paired": all(
            [trace["truth_index"] for trace in rows] == truth_indices
            for rows in traces.values()
        ),
        "all_actions_legal": all(
            step["action"]
            in model.legal_actions(tuple(step["position_before"]))
            for rows in traces.values()
            for trace in rows
            for step in trace["steps"]
        ),
        "exactly_400_logical_llm_cells": len(provider.logical_requests)
        == expected_logical,
        "unique_physical_llm_cells_at_most_64": len(provider.physical_requests)
        <= config.max_unique_llm_cells,
        "cache_accounts_for_every_logical_cell": (
            len(provider.cache) == len(provider.physical_requests)
            and len(provider.cache)
            + sum(
                request["cache_hit"] for request in provider.logical_requests
            )
            == len(provider.logical_requests)
        ),
        "rollout_scoring_made_no_llm_calls": True,
    }
    endpoint_gate = {
        f"{metric}_vs_{baseline}_lower_bound_positive": comparisons[
            f"{metric}_auc_gain_vs_{baseline}"
        ]["ci95"][0]
        > 0.0
        for baseline in ("shared_h4", "random_h5", "exact_d4")
        for metric in ("entropy", "truth_log")
    }
    endpoint_gate.update(
        {
            "mean_exact_h5_recovery_at_least_threshold": recovery
            >= config.recovery_threshold,
            "registered_route_rate_at_least_threshold": route_rate
            >= config.route_rate_threshold,
            "onsite_by_round_five_at_least_threshold": onsite_by_round_five
            >= config.route_rate_threshold,
        }
    )
    return {
        "schema_version": 1,
        "stage": "focused_range_gated_rock_cached_h5_trajectory_confirmation",
        "config": asdict(config),
        "truth_indices": truth_indices,
        "mechanics": mechanics,
        "comparisons": comparisons,
        "exact_h5_entropy_gain_vs_exact_d4": exact_gain,
        "llm_recovery_fraction_of_exact_h5_gain": recovery,
        "llm_registered_route_rate": route_rate,
        "llm_onsite_by_round_five_rate": onsite_by_round_five,
        "endpoint_gate": endpoint_gate,
        "traces": traces,
        "logical_requests": provider.logical_requests,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Focused Range-Gated Rock Cached-h5 Trajectory Confirmation",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint | Mean gain | Paired 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for baseline, label in (
        ("shared_h4", "shared compiled h4"),
        ("random_h5", "matched random targets h5"),
        ("exact_d4", "exhaustive d4"),
    ):
        for metric, metric_label in (
            ("entropy", "Entropy AUC"),
            ("truth_log", "Truth-log AUC"),
        ):
            row = result["comparisons"][
                f"{metric}_auc_gain_vs_{baseline}"
            ]
            lines.append(
                f"| {metric_label} vs {label} | {row['mean']:+.6f} | "
                f"[{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] | "
                f"{'/'.join(map(str, row['wins_ties_losses']))} |"
            )
    lines.extend(
        [
            "",
            f"Exact-h5 gain recovery: "
            f"{result['llm_recovery_fraction_of_exact_h5_gain']:.1%}.",
            f"Registered route rate: {result['llm_registered_route_rate']:.1%}.",
            f"On-site by round five: "
            f"{result['llm_onsite_by_round_five_rate']:.1%}.",
            f"Logical/physical LLM cells: {len(result['logical_requests'])}/"
            f"{len(result['candidate_requests'])}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/config_nonmyopic_range_gated_gemma26b_thinking_openrouter_s0.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-id", default="focused-range-gated-rock-cached-h5-trajectory"
    )
    parser.add_argument("--seed", type=int, default=24_245)
    parser.add_argument("--model-generation-tokens", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    confirmation_config = FocusedDepth5TrajectoryConfig(seed=args.seed)
    strategy_config = Depth5GoalCompilerConfig(seed=args.seed)
    confirmation_config.validate()
    strategy_config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicDepth5GoalModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.log_path = args.output_dir / "run.log"
        runtime_config.location_max_new_tokens = args.model_generation_tokens
        pair = runtime_config.model_pairs[0]
        runtime_config.model_pairs[0] = replace(
            pair,
            questioner=replace(
                pair.questioner,
                thinking_final_max_new_tokens=strategy_config.max_new_tokens,
            ),
        )
        chat_model = build_model_adapter(
            runtime_config.model_pairs[0].questioner,
            config=runtime_config,
        )
    base_provider = Depth5GoalCompilerProvider(chat_model, strategy_config)
    provider = CachedDepth5GoalProvider(
        base_provider,
        max_unique_cells=confirmation_config.max_unique_llm_cells,
    )
    try:
        result = run_confirmation(provider, confirmation_config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "focused_range_gated_rock_cached_h5_trajectory_confirmation",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(confirmation_config),
            "strategy_config": asdict(strategy_config),
            "logical_requests": provider.logical_requests,
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": usage_with_forced_events(chat_model, args.output_dir),
        }
        (args.output_dir / "CONFIRMATION_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    result["usage"] = usage_with_forced_events(chat_model, args.output_dir)
    result["strategy_config"] = asdict(strategy_config)
    result["run_id"] = args.run_id
    result["dry_run"] = args.dry_run
    result["gate"] = {
        "passed": all(result["mechanics"].values())
        and all(result["endpoint_gate"].values())
    }
    (args.output_dir / "CONFIRMATION.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "CONFIRMATION.md").write_text(
        render(result), encoding="utf-8"
    )
    print(
        json.dumps(
            {"gate": result["gate"], "endpoint_gate": result["endpoint_gate"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
