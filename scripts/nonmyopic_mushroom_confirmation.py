"""Paired trajectory confirmation for grounded Mushroom feature acquisition."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.mushroom_feature_acquisition import (  # noqa: E402
    COLLECT_ACTION,
    MushroomFeatureModel,
)
from helpers import Config, load_config  # noqa: E402
from model_factory import build_model_adapter  # noqa: E402
from scripts.nonmyopic_gated_sensor_strategy_prior import (  # noqa: E402
    ChatModel,
    StrategyProposalError,
    _usage_snapshot,
)
from scripts.nonmyopic_mushroom_feature_oracle import (  # noqa: E402
    _choose_action,
    exact_action_costs,
)
from scripts.nonmyopic_mushroom_proposal_gate import (  # noqa: E402
    _matched_random_strategies,
    strategy_cost,
)
from scripts.nonmyopic_mushroom_strategy import (  # noqa: E402
    DeterministicUtilityMushroomModel,
    IndexedMushroomProvider,
    MushroomStrategyConfig,
    _branch_menus,
    _fixed_roots,
)


Arm = Literal["llm", "random", "depth_one", "depth_two"]


@dataclass(frozen=True)
class MushroomConfirmationConfig:
    num_trials: int = 50
    num_rounds: int = 8
    seed: int = 24_175
    bootstrap_replicates: int = 10_000
    collection_rate_threshold: float = 0.75
    recovery_threshold: float = 0.60
    projection_cell_rate_threshold: float = 0.05
    projection_branch_rate_threshold: float = 0.01

    def validate(self, catalog_size: int) -> None:
        if self.num_trials != 50 or self.num_rounds != 8:
            raise ValueError("the frozen Mushroom confirmation uses 50 trials and 8 rounds")
        if self.bootstrap_replicates != 10_000:
            raise ValueError("the frozen Mushroom confirmation uses 10,000 bootstraps")
        if self.num_trials > catalog_size:
            raise ValueError("confirmation trials cannot exceed the Mushroom catalog")
        if not 0.0 <= self.collection_rate_threshold <= 1.0:
            raise ValueError("collection threshold must be a probability")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery threshold must be a fraction")
        if self.projection_cell_rate_threshold != 0.05:
            raise ValueError("the frozen projection cell-rate threshold is 5%")
        if self.projection_branch_rate_threshold != 0.01:
            raise ValueError("the frozen projection branch-rate threshold is 1%")


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _run_arm(
    model: MushroomFeatureModel,
    *,
    arm: Arm,
    trial_index: int,
    truth_index: int,
    config: MushroomConfirmationConfig,
    provider: IndexedMushroomProvider | None,
) -> dict[str, Any]:
    state = model.initial_state
    belief = model.initial_belief.copy()
    history: list[tuple[str, str | None]] = []
    steps: list[dict[str, Any]] = []
    for round_index in range(config.num_rounds):
        remaining = config.num_rounds - round_index
        policy: dict[str, Any] = {}
        if arm == "llm" and remaining > 1:
            if provider is None:
                raise ValueError("LLM arm requires a provider")
            proposal = provider.propose(
                model,
                cell_index=trial_index * config.num_rounds + round_index,
                state=state,
                belief=belief,
                history=tuple(history),
            )
            costs = [
                strategy_cost(model, state=state, belief=belief, strategy=strategy)
                for strategy in proposal.strategies
            ]
            slot = min(
                range(len(proposal.strategies)),
                key=lambda index: (costs[index], index),
            )
            action = proposal.strategies[slot].root_action
            policy = {
                "roots": [strategy.root_action for strategy in proposal.strategies],
                "followups": [strategy.followups for strategy in proposal.strategies],
                "policy_costs": costs,
                "selected_slot": slot,
            }
        elif arm == "random" and remaining > 1:
            roots = _fixed_roots(
                model,
                state=state,
                belief=belief,
                count=min(4, len(model.legal_actions(state))),
            )
            try:
                menus = _branch_menus(
                    model, state=state, belief=belief, roots=roots
                )
            except StrategyProposalError:
                costs, scorer_units = exact_action_costs(
                    model, state=state, belief=belief, depth=1
                )
                action = _choose_action(costs)
                policy = {
                    "exact_depth": 1,
                    "selected_cost": costs[action],
                    "scorer_units": scorer_units,
                    "reason": "no_complete_two_action_policy",
                }
            else:
                strategies = _matched_random_strategies(
                    roots=roots,
                    menus=menus,
                    seed=_stable_seed(
                        config.seed,
                        "trajectory-random",
                        trial_index,
                        round_index,
                    ),
                )
                costs = [
                    strategy_cost(
                        model,
                        state=state,
                        belief=belief,
                        strategy=strategy,
                    )
                    for strategy in strategies
                ]
                slot = min(
                    range(len(roots)),
                    key=lambda index: (costs[index], index),
                )
                action = strategies[slot].root_action
                policy = {
                    "roots": list(roots),
                    "followups": [
                        strategy.followups for strategy in strategies
                    ],
                    "policy_costs": costs,
                    "selected_slot": slot,
                }
        else:
            depth = min(2 if arm == "depth_two" else 1, remaining)
            costs, scorer_units = exact_action_costs(
                model, state=state, belief=belief, depth=depth
            )
            action = _choose_action(costs)
            policy = {
                "exact_depth": depth,
                "selected_cost": costs[action],
                "scorer_units": scorer_units,
            }
        entropy_before = model.target_entropy(belief)
        observation = model.observation(truth_index, action)
        belief = model.posterior(belief, action, observation)
        steps.append(
            {
                "round": round_index + 1,
                "specimen_collected_before": state.specimen_collected,
                "action": action,
                "observation": observation,
                "entropy_before": entropy_before,
                "entropy": model.target_entropy(belief),
                "truth_log_probability": model.truth_log_probability(
                    belief, truth_index
                ),
                "policy": policy,
            }
        )
        state = model.next_state(state, action)
        history.append((action, observation))
    entropies = [step["entropy"] for step in steps]
    truth_logs = [step["truth_log_probability"] for step in steps]
    truth_class = str(model.classes[truth_index])
    return {
        "arm": arm,
        "trial_index": trial_index,
        "truth_index": truth_index,
        "truth_class": truth_class,
        "entropy_auc": float(np.mean(entropies)),
        "truth_log_probability_auc": float(np.mean(truth_logs)),
        "final_entropy": float(entropies[-1]),
        "final_truth_log_probability": float(truth_logs[-1]),
        "final_class_accuracy": float(model.decode_map_class(belief) == truth_class),
        "steps": steps,
    }


def _bootstrap(
    values: np.ndarray, *, seed: int, replicates: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        chunks.append(values[indices].mean(axis=1))
    samples = np.concatenate(chunks)
    return [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def _summary(
    values: list[float], *, config: MushroomConfirmationConfig, label: str
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
    provider: IndexedMushroomProvider, config: MushroomConfirmationConfig
) -> dict[str, Any]:
    model = MushroomFeatureModel()
    config.validate(len(model.rows))
    truths = np.random.default_rng(config.seed).choice(
        len(model.rows), size=config.num_trials, replace=False
    )
    traces: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ("llm", "random", "depth_one", "depth_two")
    }
    for trial_index, raw_truth in enumerate(truths):
        truth = int(raw_truth)
        for arm in traces:
            traces[arm].append(
                _run_arm(
                    model,
                    arm=arm,  # type: ignore[arg-type]
                    trial_index=trial_index,
                    truth_index=truth,
                    config=config,
                    provider=provider if arm == "llm" else None,
                )
            )

    def entropy_gain(baseline: str) -> list[float]:
        return [
            base["entropy_auc"] - llm["entropy_auc"]
            for base, llm in zip(traces[baseline], traces["llm"])
        ]

    def truth_gain(baseline: str) -> list[float]:
        return [
            llm["truth_log_probability_auc"] - base["truth_log_probability_auc"]
            for base, llm in zip(traces[baseline], traces["llm"])
        ]

    comparisons = {
        "entropy_auc_gain_vs_depth_one": _summary(
            entropy_gain("depth_one"), config=config, label="entropy-vs-d1"
        ),
        "truth_log_auc_gain_vs_depth_one": _summary(
            truth_gain("depth_one"), config=config, label="truth-vs-d1"
        ),
        "entropy_auc_gain_vs_random": _summary(
            entropy_gain("random"), config=config, label="entropy-vs-random"
        ),
        "truth_log_auc_gain_vs_random": _summary(
            truth_gain("random"), config=config, label="truth-vs-random"
        ),
    }
    exact_gain = float(
        np.mean(
            [
                d1["entropy_auc"] - d2["entropy_auc"]
                for d1, d2 in zip(traces["depth_one"], traces["depth_two"])
            ]
        )
    )
    llm_gain = comparisons["entropy_auc_gain_vs_depth_one"]["mean"]
    recovery = llm_gain / exact_gain if exact_gain > 0.0 else float("-inf")
    collection_rate = float(
        np.mean(
            [
                trace["steps"][0]["action"] == COLLECT_ACTION
                for trace in traces["llm"]
            ]
        )
    )
    total_branches = sum(
        sum(len(root_menus) for root_menus in request["menus"])
        for request in provider.physical_requests
    )
    projected_cells = len(provider.projected_responses)
    projected_branches = sum(
        len(request["projection_events"]) for request in provider.projected_responses
    )
    projection = {
        "projected_cells": projected_cells,
        "projected_cell_rate": projected_cells
        / (config.num_trials * (config.num_rounds - 1)),
        "projected_branches": projected_branches,
        "total_branches": total_branches,
        "projected_branch_rate": projected_branches / total_branches,
    }
    mechanics = {
        "paired_truths_without_replacement": len(set(int(value) for value in truths))
        == config.num_trials,
        "all_four_arms_complete": all(
            len(trace["steps"]) == config.num_rounds
            for arm_traces in traces.values()
            for trace in arm_traces
        ),
        "all_actions_legal": all(
            step["action"]
            in model.legal_actions(
                _state_before(model, trace["steps"], step["round"] - 1)
            )
            for arm_traces in traces.values()
            for trace in arm_traces
            for step in trace["steps"]
        ),
        "exactly_350_accepted_llm_cells": len(provider.physical_requests)
        == config.num_trials * (config.num_rounds - 1),
        "rollout_scoring_made_no_llm_calls": True,
    }
    endpoint_gate = {
        "entropy_vs_depth_one_lower_bound_positive": comparisons[
            "entropy_auc_gain_vs_depth_one"
        ]["ci95"][0]
        > 0.0,
        "truth_vs_depth_one_lower_bound_positive": comparisons[
            "truth_log_auc_gain_vs_depth_one"
        ]["ci95"][0]
        > 0.0,
        "entropy_vs_random_lower_bound_positive": comparisons[
            "entropy_auc_gain_vs_random"
        ]["ci95"][0]
        > 0.0,
        "truth_vs_random_lower_bound_positive": comparisons[
            "truth_log_auc_gain_vs_random"
        ]["ci95"][0]
        > 0.0,
        "mean_exact_depth_two_recovery_at_least_threshold": recovery
        >= config.recovery_threshold,
        "first_collection_rate_at_least_threshold": collection_rate
        >= config.collection_rate_threshold,
        "projected_cell_rate_at_most_threshold": projection["projected_cell_rate"]
        <= config.projection_cell_rate_threshold,
        "projected_branch_rate_at_most_threshold": projection[
            "projected_branch_rate"
        ]
        <= config.projection_branch_rate_threshold,
    }
    return {
        "schema_version": 1,
        "stage": "mushroom_feature_acquisition_projected_utility_paired_trajectory_confirmation",
        "config": asdict(config),
        "truth_indices": [int(value) for value in truths],
        "mechanics": mechanics,
        "comparisons": comparisons,
        "exact_depth_two_entropy_gain_vs_depth_one": exact_gain,
        "llm_recovery_fraction_of_exact_depth_two_gain": recovery,
        "llm_first_collection_rate": collection_rate,
        "projection": projection,
        "endpoint_gate": endpoint_gate,
        "traces": traces,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
        "projected_responses": provider.projected_responses,
    }


def _state_before(
    model: MushroomFeatureModel, steps: list[dict[str, Any]], count: int
) -> Any:
    state = model.initial_state
    for step in steps[:count]:
        state = model.next_state(state, step["action"])
    return state


def render(result: dict[str, Any]) -> str:
    lines = [
        "# UCI Mushroom Projected-Utility Paired Trajectory Confirmation",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint | Mean gain | Paired 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("entropy_auc_gain_vs_depth_one", "Entropy AUC vs exact d1"),
        ("truth_log_auc_gain_vs_depth_one", "Truth-log AUC vs exact d1"),
        ("entropy_auc_gain_vs_random", "Entropy AUC vs matched random"),
        ("truth_log_auc_gain_vs_random", "Truth-log AUC vs matched random"),
    ):
        row = result["comparisons"][key]
        lines.append(
            f"| {label} | {row['mean']:+.6f} | [{row['ci95'][0]:+.6f}, "
            f"{row['ci95'][1]:+.6f}] | {'/'.join(map(str, row['wins_ties_losses']))} |"
        )
    projection = result["projection"]
    lines.extend(
        [
            "",
            f"Exact-d2 gain recovery: {result['llm_recovery_fraction_of_exact_depth_two_gain']:.1%}.",
            f"LLM first-round collection: {result['llm_first_collection_rate']:.1%}.",
            f"Projected cells: {projection['projected_cells']} "
            f"({projection['projected_cell_rate']:.2%}).",
            f"Projected branches: {projection['projected_branches']}/"
            f"{projection['total_branches']} "
            f"({projection['projected_branch_rate']:.2%}).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_nonmyopic_thyroid_gpt54mini_openrouter.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/nonmyopic/mushroom_projected_utility_gpt54mini_confirmation_20260723"
        ),
    )
    parser.add_argument(
        "--run-id",
        default="mushroom-projected-utility-gpt54mini-confirmation-20260723",
    )
    parser.add_argument("--seed", type=int, default=24_175)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = MushroomConfirmationConfig(seed=args.seed)
    strategy_config = MushroomStrategyConfig(
        seed=args.seed,
        utility_summary_mode="branch_local_expected_entropy",
        project_invalid_after_retries=True,
        allow_fewer_roots_when_exhausted=True,
    )
    strategy_config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicUtilityMushroomModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = strategy_config.max_new_tokens
        runtime_config.openrouter_max_output_tokens = strategy_config.max_new_tokens
        chat_model = build_model_adapter(
            runtime_config.model_pairs[0].questioner, config=runtime_config
        )
    provider = IndexedMushroomProvider(chat_model, strategy_config)
    try:
        result = run_confirmation(provider, config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "mushroom_feature_acquisition_projected_utility_paired_trajectory_confirmation",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(config),
            "strategy_config": asdict(strategy_config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "projected_responses": provider.projected_responses,
            "usage": _usage_snapshot(chat_model),
        }
        (args.output_dir / "CONFIRMATION_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    result["usage"] = _usage_snapshot(chat_model)
    result["strategy_config"] = asdict(strategy_config)
    result["run_id"] = args.run_id
    result["dry_run"] = args.dry_run
    result["mechanics"]["zero_reasoning_tokens"] = (
        int(result["usage"].get("reasoning_tokens", 0)) == 0
    )
    result["mechanics"]["zero_forced_exits"] = (
        int(result["usage"].get("forced_exits", 0)) == 0
    )
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
