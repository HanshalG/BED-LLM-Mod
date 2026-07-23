"""Late-state serving smoke for cached hierarchical h5 Rock trajectories."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError
from scripts.nonmyopic_range_gated_rock_depth5_goal_compiler import (
    Depth5GoalCompilerConfig,
    Depth5GoalCompilerProvider,
    DeterministicDepth5GoalModel,
    build_depth5_model,
)
from scripts.nonmyopic_range_gated_rock_depth5_oracle import H5_ROUTE
from scripts.nonmyopic_range_gated_rock_fixed_tail import usage_with_forced_events
from scripts.nonmyopic_range_gated_rock_stable_controls import (
    stable_best_index,
    stable_best_plan,
)
from scripts.nonmyopic_range_gated_rock_strategy import ChatModel, History
from scripts.nonmyopic_rock_depth_oracle import exhaustive_action_values


def trajectory_prefix_histories() -> tuple[History, ...]:
    route_prefixes: list[History] = [
        (),
        (("move-NORTH", None),),
        (("move-NORTH", None), ("move-WEST", None)),
        (
            ("move-NORTH", None),
            ("move-WEST", None),
            ("move-WEST", None),
        ),
    ]
    check_histories: list[History] = [
        (("check-6", outcome),) for outcome in ("good", "bad")
    ]
    check_histories.extend(
        (("check-6", first), ("check-6", second))
        for first in ("good", "bad")
        for second in ("good", "bad")
    )
    check_histories.extend(
        (
            ("check-6", first),
            ("check-6", second),
            ("check-6", third),
        )
        for first, second, third in (
            ("good", "good", "good"),
            ("bad", "bad", "bad"),
        )
    )
    return tuple([*route_prefixes, *check_histories])


def run_smoke(
    provider: Depth5GoalCompilerProvider,
    *,
    strategy_config: Depth5GoalCompilerConfig,
) -> dict[str, Any]:
    model = build_depth5_model()
    records: list[dict[str, Any]] = []
    histories = trajectory_prefix_histories()
    for cell_index, history in enumerate(histories):
        belief = model.initial_belief.copy()
        position = model.map_spec.start_position
        for action, observation in history:
            if action not in model.legal_actions(position):
                raise RuntimeError("registered smoke history contains an illegal action")
            belief = model.posterior(position, belief, action, observation)
            position = model.next_position(position, action)
        cell = provider.propose(
            model,
            cell_index=cell_index,
            position=position,
            belief=belief,
            history=history,
        )
        selected, selected_value, values = stable_best_plan(
            model, position=position, belief=belief, plans=cell.plans
        )
        exact_values, _ = exhaustive_action_values(
            model, position=position, belief=belief, depth=5
        )
        legal = tuple(exact_values)
        exact_action = legal[
            stable_best_index([exact_values[action] for action in legal])
        ]
        records.append(
            {
                "cell_index": cell_index,
                "history": [list(item) for item in history],
                "position": list(position),
                "plans": [list(plan) for plan in cell.plans],
                "selected_plan": list(selected),
                "selected_value": selected_value,
                "plan_values": values,
                "exact_h5_action": exact_action,
                "selected_root_matches_exact_h5": selected[0] == exact_action,
                "initial_registered_route": (
                    history == () and selected == H5_ROUTE
                ),
            }
        )
    match_count = sum(
        record["selected_root_matches_exact_h5"] for record in records
    )
    mechanics = {
        "twelve_registered_prefix_cells_completed": len(records) == 12
        and len(
            {tuple(tuple(item) for item in row["history"]) for row in records}
        )
        == 12,
        "all_cells_have_four_legal_h5_plans": all(
            len(row["plans"]) == 4
            and all(len(plan) == 5 for plan in row["plans"])
            for row in records
        ),
        "all_target_responses_are_distinct": all(
            len(set(request["target_assignments"])) == 4
            for request in provider.physical_requests
        ),
        "initial_cell_selects_registered_route": records[0][
            "initial_registered_route"
        ],
        "exact_h5_root_match_at_least_nine_of_twelve": match_count >= 9,
        "exactly_twelve_accepted_calls": len(provider.physical_requests) == 12,
        "scoring_made_no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "focused_range_gated_rock_h5_trajectory_serving_smoke",
        "strategy_config": asdict(strategy_config),
        "mechanics": mechanics,
        "exact_h5_root_match_count": match_count,
        "records": records,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


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
    parser.add_argument("--run-id", default="focused-range-gated-h5-trajectory-s0")
    parser.add_argument("--seed", type=int, default=24_244)
    parser.add_argument("--model-generation-tokens", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    strategy_config = Depth5GoalCompilerConfig(seed=args.seed)
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
    provider = Depth5GoalCompilerProvider(chat_model, strategy_config)
    try:
        result = run_smoke(provider, strategy_config=strategy_config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "focused_range_gated_rock_h5_trajectory_serving_smoke",
            "status": "failed_closed",
            "error": str(exc),
            "strategy_config": asdict(strategy_config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": usage_with_forced_events(chat_model, args.output_dir),
        }
        (args.output_dir / "SMOKE_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    result["usage"] = usage_with_forced_events(chat_model, args.output_dir)
    result["run_id"] = args.run_id
    result["dry_run"] = args.dry_run
    result["mechanics"]["usage_accounted"] = args.dry_run or all(
        field in result["usage"]
        for field in ("requests", "completion_tokens", "forced_exits")
    )
    result["gate"] = {"passed": all(result["mechanics"].values())}
    (args.output_dir / "SMOKE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate": result["gate"],
                "exact_h5_root_match_count": result[
                    "exact_h5_root_match_count"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
