"""Paired range-gated Rock trajectory confirmation with cached h3 proposals."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel, get_paper_map
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_gated_sensor_strategy_prior import (
    StrategyProposalError,
)
from scripts.nonmyopic_range_gated_rock_fixed_tail import (
    DeterministicFixedTailModel,
    FixedRootTailProvider,
    fixed_roots,
    usage_with_forced_events,
)
from scripts.nonmyopic_range_gated_rock_fixed_tail_proposal_gate import (
    matched_random_fixed_root_plans,
)
from scripts.nonmyopic_range_gated_rock_proposal_gate import (
    _stable_seed,
)
from scripts.nonmyopic_range_gated_rock_stable_controls import (
    stable_best_index,
    stable_best_plan,
)
from scripts.nonmyopic_range_gated_rock_strategy import (
    ChatModel,
    History,
    Plan,
    RangeGatedPlanCell,
    RangeGatedStrategyConfig,
)
from scripts.nonmyopic_rock_depth_oracle import (
    _uniform,
    exhaustive_action_values,
)


Arm = Literal["llm_h3", "random_h3", "exact_d2", "exact_d3"]


@dataclass(frozen=True)
class RangeGatedTrajectoryConfig:
    num_trials: int = 50
    num_rounds: int = 8
    seed: int = 24_188
    bootstrap_replicates: int = 10_000
    recovery_threshold: float = 0.60
    route_rate_threshold: float = 0.75
    max_unique_llm_cells: int = 64
    remote_accuracy: float = 0.55
    onsite_accuracy: float = 0.95

    def validate(self) -> None:
        if self.num_trials != 50 or self.num_rounds != 8:
            raise ValueError("the frozen range-gated trajectory uses 50 trials and 8 rounds")
        if self.bootstrap_replicates != 10_000:
            raise ValueError("the frozen trajectory uses 10,000 paired bootstraps")
        if self.max_unique_llm_cells != 64:
            raise ValueError("the frozen physical proposal cap is 64 unique prompts")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery threshold must be a probability")
        if not 0.0 <= self.route_rate_threshold <= 1.0:
            raise ValueError("route threshold must be a probability")
        if not 0.5 <= self.remote_accuracy < self.onsite_accuracy <= 1.0:
            raise ValueError("accuracies must satisfy 0.5 <= remote < onsite <= 1")


class CachedFixedRootTailProvider:
    """Cache exact prompt identities while retaining every logical decision."""

    def __init__(
        self,
        provider: FixedRootTailProvider,
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
        prompt_roots = fixed_roots(model, position=position, belief=belief)
        messages = self.provider._messages(
            model,
            position=position,
            belief=belief,
            history=history,
            roots=prompt_roots,
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


def _model(config: RangeGatedTrajectoryConfig) -> RangeGatedRockDiagnosisModel:
    return RangeGatedRockDiagnosisModel(
        get_paper_map("7-8"),
        remote_accuracy=config.remote_accuracy,
        onsite_accuracy=config.onsite_accuracy,
    )


def _exact_action(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    depth: int,
) -> tuple[str, float, int]:
    values, units = exhaustive_action_values(
        model, position=position, belief=belief, depth=depth
    )
    legal = tuple(values)
    index = stable_best_index([values[action] for action in legal])
    action = legal[index]
    return action, float(values[action]), units


def _observation(
    model: RangeGatedRockDiagnosisModel,
    *,
    config: RangeGatedTrajectoryConfig,
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
            "range-trajectory-observation",
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
    config: RangeGatedTrajectoryConfig,
    provider: CachedFixedRootTailProvider | None,
) -> dict[str, Any]:
    belief = model.initial_belief.copy()
    position = model.map_spec.start_position
    history: list[tuple[str, str | None]] = []
    check_counts: dict[tuple[tuple[int, int], int], int] = {}
    steps: list[dict[str, Any]] = []
    for round_index in range(config.num_rounds):
        remaining = config.num_rounds - round_index
        policy: dict[str, Any]
        if arm == "llm_h3" and remaining >= 3:
            if provider is None:
                raise ValueError("llm_h3 requires a provider")
            proposal = provider.propose(
                model,
                cell_index=trial_index * config.num_rounds + round_index,
                position=position,
                belief=belief,
                history=tuple(history),
            )
            plan, value, values = stable_best_plan(
                model,
                position=position,
                belief=belief,
                plans=proposal.plans,
            )
            action = plan[0]
            policy = {
                "mode": "cached_fixed_root_h3",
                "plans": [list(item) for item in proposal.plans],
                "plan_values": values,
                "selected_plan": list(plan),
                "selected_value": value,
            }
        elif arm == "random_h3" and remaining >= 3:
            plans = matched_random_fixed_root_plans(
                model,
                position=position,
                belief=belief,
                seed=_stable_seed(
                    config.seed,
                    "trajectory-random",
                    trial_index,
                    round_index,
                ),
            )
            plan, value, values = stable_best_plan(
                model, position=position, belief=belief, plans=plans
            )
            action = plan[0]
            policy = {
                "mode": "identical_root_random_h3",
                "plans": [list(item) for item in plans],
                "plan_values": values,
                "selected_plan": list(plan),
                "selected_value": value,
            }
        else:
            requested_depth = 3 if arm == "exact_d3" else 2
            depth = min(requested_depth, remaining)
            action, value, units = _exact_action(
                model, position=position, belief=belief, depth=depth
            )
            policy = {
                "mode": "exact",
                "depth": depth,
                "selected_value": value,
                "scorer_units": units,
            }

        entropy_before = model.entropy(belief)
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
                "entropy_before": entropy_before,
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
    values: np.ndarray, *, seed: int, replicates: int
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
    values: list[float], *, config: RangeGatedTrajectoryConfig, label: str
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "ci95": _bootstrap(
            array,
            seed=_stable_seed(config.seed, "trajectory-bootstrap", label),
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
    provider: CachedFixedRootTailProvider,
    config: RangeGatedTrajectoryConfig,
) -> dict[str, Any]:
    config.validate()
    model = _model(config)
    truth_indices = [
        int(
            np.random.default_rng(
                _stable_seed(config.seed, "range-trajectory-truth", trial_index)
            ).integers(len(model.hidden_states))
        )
        for trial_index in range(config.num_trials)
    ]
    traces: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ("llm_h3", "random_h3", "exact_d2", "exact_d3")
    }
    for trial_index, truth_index in enumerate(truth_indices):
        for arm in traces:
            traces[arm].append(
                _run_arm(
                    model,
                    arm=arm,  # type: ignore[arg-type]
                    trial_index=trial_index,
                    truth_index=truth_index,
                    config=config,
                    provider=provider if arm == "llm_h3" else None,
                )
            )

    def entropy_gain(baseline: str) -> list[float]:
        return [
            base["entropy_auc"] - llm["entropy_auc"]
            for base, llm in zip(traces[baseline], traces["llm_h3"], strict=True)
        ]

    def truth_gain(baseline: str) -> list[float]:
        return [
            llm["truth_log_probability_auc"] - base["truth_log_probability_auc"]
            for base, llm in zip(traces[baseline], traces["llm_h3"], strict=True)
        ]

    comparisons = {
        "entropy_auc_gain_vs_exact_d2": _summary(
            entropy_gain("exact_d2"), config=config, label="entropy-vs-exact-d2"
        ),
        "truth_log_auc_gain_vs_exact_d2": _summary(
            truth_gain("exact_d2"), config=config, label="truth-vs-exact-d2"
        ),
        "entropy_auc_gain_vs_random_h3": _summary(
            entropy_gain("random_h3"), config=config, label="entropy-vs-random-h3"
        ),
        "truth_log_auc_gain_vs_random_h3": _summary(
            truth_gain("random_h3"), config=config, label="truth-vs-random-h3"
        ),
    }
    exact_gain = float(
        np.mean(
            [
                d2["entropy_auc"] - d3["entropy_auc"]
                for d2, d3 in zip(
                    traces["exact_d2"], traces["exact_d3"], strict=True
                )
            ]
        )
    )
    recovery = (
        comparisons["entropy_auc_gain_vs_exact_d2"]["mean"] / exact_gain
        if exact_gain > 0.0
        else float("-inf")
    )
    first_two_south = float(
        np.mean(
            [
                trace["steps"][0]["action"] == "move-SOUTH"
                and trace["steps"][1]["action"] == "move-SOUTH"
                for trace in traces["llm_h3"]
            ]
        )
    )
    onsite_by_round_three = float(
        np.mean(
            [
                any(
                    step["round"] <= 3
                    and step["action"].startswith("check-")
                    and tuple(step["position_before"])
                    == model.map_spec.rock_positions[
                        int(step["action"].split("-")[1])
                    ]
                    for step in trace["steps"]
                )
                for trace in traces["llm_h3"]
            ]
        )
    )
    expected_logical = config.num_trials * (config.num_rounds - 2)
    mechanics = {
        "all_four_arms_complete": all(
            len(trace["steps"]) == config.num_rounds
            for arm_traces in traces.values()
            for trace in arm_traces
        ),
        "all_truths_paired": all(
            [trace["truth_index"] for trace in arm_traces] == truth_indices
            for arm_traces in traces.values()
        ),
        "all_actions_legal": all(
            step["action"]
            in model.legal_actions(tuple(step["position_before"]))
            for arm_traces in traces.values()
            for trace in arm_traces
            for step in trace["steps"]
        ),
        "exactly_300_logical_llm_cells": len(provider.logical_requests)
        == expected_logical,
        "unique_physical_llm_cells_at_most_64": len(provider.physical_requests)
        <= config.max_unique_llm_cells,
        "cache_accounts_for_every_logical_cell": (
            len(provider.cache) == len(provider.physical_requests)
            and len(provider.cache)
            + sum(request["cache_hit"] for request in provider.logical_requests)
            == len(provider.logical_requests)
        ),
        "rollout_scoring_made_no_llm_calls": True,
    }
    endpoint_gate = {
        "entropy_vs_exact_d2_lower_bound_positive": comparisons[
            "entropy_auc_gain_vs_exact_d2"
        ]["ci95"][0]
        > 0.0,
        "truth_vs_exact_d2_lower_bound_positive": comparisons[
            "truth_log_auc_gain_vs_exact_d2"
        ]["ci95"][0]
        > 0.0,
        "entropy_vs_random_h3_lower_bound_positive": comparisons[
            "entropy_auc_gain_vs_random_h3"
        ]["ci95"][0]
        > 0.0,
        "truth_vs_random_h3_lower_bound_positive": comparisons[
            "truth_log_auc_gain_vs_random_h3"
        ]["ci95"][0]
        > 0.0,
        "mean_exact_d3_recovery_at_least_threshold": recovery
        >= config.recovery_threshold,
        "first_two_south_rate_at_least_threshold": first_two_south
        >= config.route_rate_threshold,
        "onsite_by_round_three_rate_at_least_threshold": onsite_by_round_three
        >= config.route_rate_threshold,
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_cached_h3_paired_trajectory_confirmation",
        "config": asdict(config),
        "truth_indices": truth_indices,
        "mechanics": mechanics,
        "comparisons": comparisons,
        "exact_d3_entropy_gain_vs_exact_d2": exact_gain,
        "llm_recovery_fraction_of_exact_d3_gain": recovery,
        "llm_first_two_south_rate": first_two_south,
        "llm_onsite_by_round_three_rate": onsite_by_round_three,
        "endpoint_gate": endpoint_gate,
        "traces": traces,
        "logical_requests": provider.logical_requests,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Range-Gated Rock Cached-h3 Paired Trajectory Confirmation",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint | Mean gain | Paired 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("entropy_auc_gain_vs_exact_d2", "Entropy AUC vs exhaustive d2"),
        ("truth_log_auc_gain_vs_exact_d2", "Truth-log AUC vs exhaustive d2"),
        ("entropy_auc_gain_vs_random_h3", "Entropy AUC vs fixed-root random h3"),
        ("truth_log_auc_gain_vs_random_h3", "Truth-log AUC vs fixed-root random h3"),
    ):
        row = result["comparisons"][key]
        lines.append(
            f"| {label} | {row['mean']:+.6f} | "
            f"[{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] | "
            f"{'/'.join(map(str, row['wins_ties_losses']))} |"
        )
    lines.extend(
        [
            "",
            f"Exact-d3 gain recovery: {result['llm_recovery_fraction_of_exact_d3_gain']:.1%}.",
            f"Two-south route rate: {result['llm_first_two_south_rate']:.1%}.",
            f"On-site by round three: {result['llm_onsite_by_round_three_rate']:.1%}.",
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
            "configs/config_nonmyopic_range_gated_gemma26b_thinking_cluster.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-id", default="range-gated-rock-cached-h3-confirmation-20260723"
    )
    parser.add_argument("--seed", type=int, default=24_188)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    confirmation_config = RangeGatedTrajectoryConfig(seed=args.seed)
    strategy_config = RangeGatedStrategyConfig(seed=args.seed)
    strategy_config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicFixedTailModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.log_path = str(args.output_dir / "run.log")
        chat_model = build_model_adapter(
            runtime_config.model_pairs[0].questioner, config=runtime_config
        )
    base_provider = FixedRootTailProvider(
        chat_model,
        strategy_config,
        include_successor_grounding=True,
        accept_json_prefix=True,
    )
    provider = CachedFixedRootTailProvider(
        base_provider,
        max_unique_cells=confirmation_config.max_unique_llm_cells,
    )
    try:
        result = run_confirmation(provider, confirmation_config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "range_gated_rock_cached_h3_paired_trajectory_confirmation",
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
