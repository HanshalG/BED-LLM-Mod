"""Zero-LLM mechanics smoke for compact Rock Diagnosis strategies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import (
    RockDiagnosisModel,
    RockStrategyExecutor,
    get_paper_map,
    parse_rock_strategy,
    random_rock_strategy_text,
    score_rock_strategy_exact,
)


@dataclass(frozen=True)
class L0SmokeConfig:
    map_names: tuple[str, ...] = ("3-6", "5-7")
    num_trials_per_map: int = 5
    num_rounds: int = 4
    num_strategies: int = 4
    planning_horizon: int = 3
    seed: int = 10_031

    def validate(self) -> None:
        if not self.map_names:
            raise ValueError("map_names must not be empty")
        for map_name in self.map_names:
            get_paper_map(map_name)
        if min(self.num_trials_per_map, self.num_rounds, self.num_strategies, self.planning_horizon) <= 0:
            raise ValueError("trial, round, strategy, and horizon counts must be positive")


def _stable_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def _uniform(*parts: Any) -> float:
    return _stable_seed(*parts) / 2**64


def run_l0_smoke(config: L0SmokeConfig = L0SmokeConfig()) -> dict[str, Any]:
    config.validate()
    traces: list[dict[str, Any]] = []
    all_actions_legal = True
    all_scores_finite_nonnegative = True
    root_execution_matches_score = True
    all_candidate_sets_complete = True

    for map_name in config.map_names:
        model = RockDiagnosisModel(get_paper_map(map_name))
        executor = RockStrategyExecutor(model)
        for trial_index in range(config.num_trials_per_map):
            truth_rng = np.random.default_rng(_stable_seed(config.seed, map_name, "truth", trial_index))
            truth_index = int(truth_rng.integers(len(model.hidden_states)))
            belief = model.initial_belief.copy()
            position = model.map_spec.start_position
            history: tuple[tuple[str, str | None], ...] = ()
            check_counts: dict[tuple[tuple[int, int], int], int] = {}
            steps: list[dict[str, Any]] = []

            for round_index in range(config.num_rounds):
                strategy_rng = np.random.default_rng(
                    _stable_seed(config.seed, map_name, "strategies", trial_index, round_index)
                )
                strategies = [
                    parse_rock_strategy(
                        random_rock_strategy_text(model, strategy_rng, index=index),
                        model,
                    )
                    for index in range(config.num_strategies)
                ]
                horizon = min(config.planning_horizon, config.num_rounds - round_index)
                scores = [
                    score_rock_strategy_exact(
                        model,
                        strategy,
                        position=position,
                        belief=belief,
                        history=history,
                        horizon=horizon,
                    )
                    for strategy in strategies
                ]
                all_candidate_sets_complete = all_candidate_sets_complete and len(scores) == config.num_strategies
                all_scores_finite_nonnegative = all_scores_finite_nonnegative and all(
                    math.isfinite(score.eig) and score.eig >= 0.0 for score in scores
                )
                selected_index = max(range(len(scores)), key=lambda index: (scores[index].eig, -index))
                selected_strategy = strategies[selected_index]
                selected_score = scores[selected_index]
                action = executor.choose_action(
                    selected_strategy,
                    position=position,
                    belief=belief,
                    history=history,
                    strategy_step=0,
                )
                root_execution_matches_score = root_execution_matches_score and action == selected_score.root_action
                all_actions_legal = all_actions_legal and action in model.legal_actions(position)

                check_id = model.check_id(action)
                if check_id is None:
                    outcome: str | None = None
                else:
                    key = (position, check_id)
                    repeat_index = check_counts.get(key, 0)
                    check_counts[key] = repeat_index + 1
                    probability_good = float(model.likelihood_vector(position, action, "good")[truth_index])
                    outcome = (
                        "good"
                        if _uniform(config.seed, map_name, "observation", trial_index, position, check_id, repeat_index)
                        < probability_good
                        else "bad"
                    )
                next_belief = model.posterior(position, belief, action, outcome)
                steps.append(
                    {
                        "round": round_index,
                        "position": list(position),
                        "action": action,
                        "observation": outcome,
                        "selected_strategy_index": selected_index,
                        "selected_strategy": selected_strategy.raw_text,
                        "selected_exact_eig": selected_score.eig,
                        "candidate_exact_eig": [score.eig for score in scores],
                        "candidate_root_actions": [score.root_action for score in scores],
                        "expanded_nodes": [score.expanded_decision_nodes for score in scores],
                        "entropy_before": model.entropy(belief),
                        "entropy_after": model.entropy(next_belief),
                    }
                )
                history = history + ((action, outcome),)
                position = model.next_position(position, action)
                belief = next_belief

            traces.append(
                {
                    "map_name": map_name,
                    "trial_index": trial_index,
                    "truth_index": truth_index,
                    "final_entropy": model.entropy(belief),
                    "final_map_accuracy": float(model.decode_map_index(belief) == truth_index),
                    "final_truth_log_probability": float(
                        math.log(max(float(belief[truth_index]), np.finfo(float).tiny))
                    ),
                    "steps": steps,
                }
            )

    mechanics = {
        "zero_llm_calls": True,
        "parse_failures": 0,
        "execution_failures": 0,
        "all_selected_actions_legal": all_actions_legal,
        "all_exact_scores_finite_nonnegative": all_scores_finite_nonnegative,
        "all_candidate_sets_complete": all_candidate_sets_complete,
        "scored_root_matches_executed_root": root_execution_matches_score,
    }
    return {
        "schema_version": 1,
        "stage": "L0",
        "config": asdict(config),
        "num_trajectories": len(traces),
        "mechanics": mechanics,
        "gate_passed": all(bool(value) for key, value in mechanics.items() if key not in {"parse_failures", "execution_failures"})
        and mechanics["parse_failures"] == 0
        and mechanics["execution_failures"] == 0,
        "traces": traces,
    }


def render_report(summary: dict[str, Any]) -> str:
    config = summary["config"]
    mechanics = summary["mechanics"]
    lines = [
        "# StrategyEIG Revival L0 Mechanics Smoke",
        "",
        "This is a zero-LLM test of the registered Rock strategy grammar, fail-closed executor, and exact rollout-EIG scorer.",
        "",
        f"- Maps: `{', '.join(config['map_names'])}`.",
        f"- Trajectories: `{summary['num_trajectories']}` (`{config['num_trials_per_map']}` per map).",
        f"- Rounds / strategies / planning horizon: `{config['num_rounds']}` / `{config['num_strategies']}` / `{config['planning_horizon']}`.",
        f"- Zero LLM calls: `{mechanics['zero_llm_calls']}`.",
        f"- Parse / execution failures: `{mechanics['parse_failures']}` / `{mechanics['execution_failures']}`.",
        f"- All selected actions legal: `{mechanics['all_selected_actions_legal']}`.",
        f"- All exact scores finite and non-negative: `{mechanics['all_exact_scores_finite_nonnegative']}`.",
        f"- Scored roots equal executed roots: `{mechanics['scored_root_matches_executed_root']}`.",
        "",
        f"**L0 mechanics gate passed: `{summary['gate_passed']}`.**",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/rock_strategy_l0_smoke/20260716"),
    )
    args = parser.parse_args()
    summary = run_l0_smoke()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"gate_passed": summary["gate_passed"], "mechanics": summary["mechanics"]}, indent=2))
    if not summary["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
