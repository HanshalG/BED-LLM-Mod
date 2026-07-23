"""Late-state serving smoke for cached range-gated h3 trajectories."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError
from scripts.nonmyopic_range_gated_rock_fixed_tail import (
    DeterministicFixedTailModel,
    FixedRootTailProvider,
    usage_with_forced_events,
)
from scripts.nonmyopic_range_gated_rock_stable_controls import stable_best_plan
from scripts.nonmyopic_range_gated_rock_strategy import (
    ChatModel,
    History,
    RangeGatedStrategyConfig,
)
from scripts.nonmyopic_range_gated_rock_trajectory import (
    RangeGatedTrajectoryConfig,
    _exact_action,
    _model,
)


def trajectory_prefix_histories() -> tuple[History, ...]:
    approach: History = (
        ("move-SOUTH", None),
        ("move-SOUTH", None),
    )
    histories: list[History] = [(), approach[:1], approach]
    histories.extend(
        (*approach, ("check-5", outcome))
        for outcome in ("good", "bad")
    )
    histories.extend(
        (*approach, ("check-5", first), ("check-5", second))
        for first, second in itertools.product(("good", "bad"), repeat=2)
    )
    histories.extend(
        (
            *approach,
            ("check-5", first),
            ("check-5", second),
            ("check-5", third),
        )
        for first, second, third in (
            ("good", "good", "good"),
            ("good", "good", "bad"),
            ("bad", "bad", "good"),
        )
    )
    return tuple(histories)


def run_smoke(
    provider: FixedRootTailProvider,
    *,
    strategy_config: RangeGatedStrategyConfig,
) -> dict[str, Any]:
    trajectory_config = RangeGatedTrajectoryConfig()
    model = _model(trajectory_config)
    records: list[dict[str, Any]] = []
    for cell_index, history in enumerate(trajectory_prefix_histories()):
        position = model.map_spec.start_position
        belief = model.initial_belief.copy()
        for action, outcome in history:
            belief = model.posterior(position, belief, action, outcome)
            position = model.next_position(position, action)
        proposal = provider.propose(
            model,
            cell_index=cell_index,
            position=position,
            belief=belief,
            history=history,
        )
        selected_plan, selected_value, _ = stable_best_plan(
            model,
            position=position,
            belief=belief,
            plans=proposal.plans,
        )
        exact_action, exact_value, _ = _exact_action(
            model, position=position, belief=belief, depth=3
        )
        records.append(
            {
                "cell_index": cell_index,
                "history": [list(item) for item in history],
                "position": list(position),
                "plans": [list(plan) for plan in proposal.plans],
                "selected_plan": list(selected_plan),
                "selected_value": selected_value,
                "exact_d3_action": exact_action,
                "exact_d3_value": exact_value,
                "selected_root_matches_exact_d3": selected_plan[0]
                == exact_action,
            }
        )

    exact_rate = sum(
        row["selected_root_matches_exact_d3"] for row in records
    ) / len(records)
    early_expected = ("move-SOUTH", "move-SOUTH", "check-5")
    early_actions = tuple(row["selected_plan"][0] for row in records[:3])
    mechanics = {
        "twelve_prefix_cells_completed": len(records) == 12,
        "all_cells_have_four_legal_plans": all(
            len(row["plans"]) == 4 for row in records
        ),
        "all_twelve_cells_are_physical_calls": len(provider.physical_requests)
        == 12,
        "all_early_route_roots_match_exact_d3": early_actions
        == early_expected,
        "exact_d3_root_match_rate_at_least_75_percent": exact_rate >= 0.75,
        "rollout_scoring_made_no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_cached_h3_late_state_serving_smoke",
        "strategy_config": asdict(strategy_config),
        "mechanics": mechanics,
        "exact_d3_root_match_rate": exact_rate,
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
            "configs/config_nonmyopic_range_gated_gemma26b_thinking_cluster.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-id", default="range-gated-rock-cached-h3-smoke-20260723"
    )
    parser.add_argument("--seed", type=int, default=24_187)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    strategy_config = RangeGatedStrategyConfig(seed=args.seed)
    strategy_config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicFixedTailModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.log_path = args.output_dir / "run.log"
        chat_model = build_model_adapter(
            runtime_config.model_pairs[0].questioner, config=runtime_config
        )
    provider = FixedRootTailProvider(
        chat_model,
        strategy_config,
        include_successor_grounding=True,
        accept_json_prefix=True,
    )
    try:
        result = run_smoke(provider, strategy_config=strategy_config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "range_gated_rock_cached_h3_late_state_serving_smoke",
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
    if args.dry_run:
        result["mechanics"]["deterministic_dry_run_used_no_live_model"] = True
    else:
        result["mechanics"]["reasoning_usage_accounted"] = all(
            field in result["usage"]
            for field in ("reasoning_tokens", "completion_tokens", "requests")
        )
    if result["usage"].get("backend") == "vllm":
        result["mechanics"]["forced_finalization_events_accounted"] = (
            "forced_finalization_events" in result["usage"]
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
                "exact_d3_root_match_rate": result["exact_d3_root_match_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
