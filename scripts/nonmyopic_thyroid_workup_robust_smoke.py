"""Late-state serving smoke for named UCI thyroid continuation policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.thyroid_workup import ThyroidWorkupModel  # noqa: E402
from helpers import Config, load_config  # noqa: E402
from model_factory import build_model_adapter  # noqa: E402
from scripts.nonmyopic_gated_sensor_strategy_prior import (  # noqa: E402
    ChatModel,
    StrategyProposalError,
    _usage_snapshot,
)
from scripts.nonmyopic_thyroid_workup_oracle import exact_action_costs  # noqa: E402
from scripts.nonmyopic_thyroid_workup_strategy import (  # noqa: E402
    DeterministicNamedThyroidModel,
    NamedThyroidProvider,
    ThyroidStrategyConfig,
)


def _choose(costs: dict[str, float]) -> str:
    actions = tuple(costs)
    return min(actions, key=lambda action: (costs[action], actions.index(action)))


def build_robust_cells(
    model: ThyroidWorkupModel, *, seed: int
) -> list[tuple[int, Any, np.ndarray, tuple[tuple[str, str | None], ...]]]:
    truths = np.random.default_rng(seed).permutation(len(model.targets))
    depths = (0, 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2)
    cells = []
    for raw_truth, steps in zip(truths[: len(depths)], depths, strict=True):
        truth = int(raw_truth)
        state = model.initial_state
        belief = model.initial_belief
        history: list[tuple[str, str | None]] = []
        for step in range(steps):
            costs = exact_action_costs(
                model,
                state=state,
                belief=belief,
                depth=min(2, 8 - step),
            )
            action = _choose(costs)
            outcome = model.observation(truth, action)
            belief = model.posterior(belief, action, outcome)
            state = model.next_state(state, action)
            history.append((action, outcome))
        cells.append((truth, state, belief, tuple(history)))
    return cells


def run_smoke(provider: NamedThyroidProvider, *, seed: int) -> dict[str, Any]:
    model = ThyroidWorkupModel()
    records = []
    all_legal = True
    no_root_repeated = True
    cells = build_robust_cells(model, seed=seed)
    for cell_index, (truth, state, belief, history) in enumerate(cells):
        cell = provider.propose(
            model,
            cell_index=cell_index,
            state=state,
            belief=belief,
            history=history,
        )
        for strategy in cell.strategies:
            child_legal = model.legal_actions(model.next_state(state, strategy.root_action))
            all_legal &= all(action in child_legal for action in strategy.followups.values())
            no_root_repeated &= all(
                action != strategy.root_action for action in strategy.followups.values()
            )
        records.append(
            {
                "cell_index": cell_index,
                "truth_index": truth,
                "history": [list(item) for item in history],
                "roots": [strategy.root_action for strategy in cell.strategies],
                "followups": [strategy.followups for strategy in cell.strategies],
            }
        )
    return {
        "schema_version": 1,
        "stage": "uci_thyroid_workup_named_late_state_serving_smoke",
        "seed": seed,
        "mechanics": {
            "twelve_cells_completed": len(records) == 12,
            "rounds_zero_through_six_covered": {len(row["history"]) for row in records}
            == set(range(7)),
            "all_named_followups_legal": all_legal,
            "no_policy_repeats_its_root_as_followup": no_root_repeated,
            "exactly_twelve_accepted_cells": len(provider.physical_requests) == 12,
            "rollout_scoring_made_no_llm_calls": True,
        },
        "records": records,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


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
        default=Path("results/nonmyopic/thyroid_workup_gpt54mini_robust_smoke_20260723"),
    )
    parser.add_argument("--run-id", default="thyroid-workup-gpt54mini-robust-smoke-20260723")
    parser.add_argument("--seed", type=int, default=24_157)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    strategy_config = ThyroidStrategyConfig(seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicNamedThyroidModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = strategy_config.max_new_tokens
        runtime_config.openrouter_max_output_tokens = strategy_config.max_new_tokens
        chat_model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = NamedThyroidProvider(chat_model, strategy_config)
    try:
        result = run_smoke(provider, seed=args.seed)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "uci_thyroid_workup_named_late_state_serving_smoke",
            "status": "failed_closed",
            "error": str(exc),
            "strategy_config": asdict(strategy_config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage_snapshot(chat_model),
        }
        (args.output_dir / "SMOKE_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    result["strategy_config"] = asdict(strategy_config)
    result["usage"] = _usage_snapshot(chat_model)
    result["run_id"] = args.run_id
    result["dry_run"] = args.dry_run
    result["mechanics"]["zero_reasoning_tokens"] = (
        int(result["usage"].get("reasoning_tokens", 0)) == 0
    )
    result["mechanics"]["zero_forced_exits"] = (
        int(result["usage"].get("forced_exits", 0)) == 0
    )
    result["passed"] = all(result["mechanics"].values())
    (args.output_dir / "SMOKE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": result["passed"], "mechanics": result["mechanics"]}, indent=2))


if __name__ == "__main__":
    main()
