"""Serving and proposal-quality smoke for Rock branch-policy StrategyEIG."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_rock_strategy_prior import (
    ChatModel,
    DeterministicStrategyModel,
    History,
    L1Config,
    LLMRockStrategyProvider,
    StrategyProposalError,
)


def _exhaustive_d2_value(
    model: RockDiagnosisModel,
    position: tuple[int, int],
    belief: np.ndarray,
    horizon: int,
) -> float:
    values: list[float] = []
    for root_action in model.legal_actions(position):
        value = model.expected_information_gain(position, belief, root_action)
        if horizon > 1:
            child_position = model.next_position(position, root_action)
            continuation = 0.0
            for outcome in model.outcomes(root_action):
                probability = model.outcome_probability(position, belief, root_action, outcome)
                if probability <= 0.0:
                    continue
                posterior = model.posterior(position, belief, root_action, outcome)
                continuation += probability * max(
                    model.expected_information_gain(child_position, posterior, action)
                    for action in model.legal_actions(child_position)
                )
            value += continuation
        values.append(value)
    return max(values)


def _advance(
    model: RockDiagnosisModel,
    position: tuple[int, int],
    belief: np.ndarray,
    history: History,
    action: str,
    outcome: str | None,
) -> tuple[tuple[int, int], np.ndarray, History]:
    posterior = model.posterior(position, belief, action, outcome)
    return (
        model.next_position(position, action),
        posterior,
        history + ((action, outcome),),
    )


def _probe_states(map_name: str) -> list[tuple[tuple[int, int], np.ndarray, History]]:
    model = RockDiagnosisModel(get_paper_map(map_name))
    start = model.map_spec.start_position
    belief = model.initial_belief.copy()
    empty: History = ()
    states = [(start, belief, empty)]
    states.append(_advance(model, start, belief, empty, "move-EAST", None))
    states.append(_advance(model, start, belief, empty, "check-0", "good"))
    states.append(_advance(model, start, belief, empty, "check-0", "bad"))
    east_position, east_belief, east_history = states[1]
    states.append(
        _advance(model, east_position, east_belief, east_history, "check-0", "good")
    )
    states.append(
        _advance(model, east_position, east_belief, east_history, "check-0", "bad")
    )
    states.append(_advance(model, start, belief, empty, "check-1", "good"))
    states.append(_advance(model, start, belief, empty, "check-1", "bad"))
    states.append(_advance(model, start, belief, empty, "move-SOUTH", None))
    south_position, south_belief, south_history = states[8]
    states.append(
        _advance(
            model,
            south_position,
            south_belief,
            south_history,
            f"check-{min(5, model.num_rocks - 1)}",
            "good",
        )
    )
    return states


def run_smoke(
    model_adapter: ChatModel,
    *,
    num_strategies: int = 4,
    concurrency: int = 10,
    map_names: tuple[str, ...] = ("3-6", "5-7"),
    probe_states_per_map: int = 5,
) -> dict[str, Any]:
    if not 1 <= probe_states_per_map <= 10:
        raise ValueError("probe_states_per_map must be in [1, 10]")
    config = L1Config(
        map_names=map_names,
        num_trials_per_map=1,
        num_rounds=2,
        num_strategies=num_strategies,
        bootstrap_replicates=10,
        trial_concurrency=1,
        strategy_schema="branch_policy_v2",
    )
    provider = LLMRockStrategyProvider(model_adapter, config)
    jobs: list[tuple[str, int, tuple[int, int], np.ndarray, History, int]] = []
    for map_name in config.map_names:
        for state_index, (position, belief, history) in enumerate(
            _probe_states(map_name)[:probe_states_per_map]
        ):
            jobs.append(
                (
                    map_name,
                    state_index,
                    position,
                    belief,
                    history,
                    1 if (state_index + 1) % 5 == 0 else 2,
                )
            )

    def execute(job: tuple[str, int, tuple[int, int], np.ndarray, History, int]) -> dict[str, Any]:
        map_name, state_index, position, belief, history, horizon = job
        model = RockDiagnosisModel(get_paper_map(map_name))
        try:
            cell = provider.propose_strategies(
                model,
                map_name=map_name,
                trial_index=state_index,
                position=position,
                belief=belief,
                history=history,
                horizon=horizon,
            )
        except StrategyProposalError as exc:
            return {
                "map_name": map_name,
                "state_index": state_index,
                "position": list(position),
                "horizon": horizon,
                "status": "failed",
                "error": str(exc),
            }
        parsed = [json.loads(strategy.raw_text) for strategy in cell.strategies]
        move_policies = [item for item in parsed if item["root_action"].startswith("move-")]
        check_policies = [item for item in parsed if item["root_action"].startswith("check-")]
        best_index = max(range(len(cell.exact_scores)), key=lambda index: cell.exact_scores[index].eig)
        exhaustive_value = _exhaustive_d2_value(model, position, belief, horizon)
        return {
            "map_name": map_name,
            "state_index": state_index,
            "position": list(position),
            "horizon": horizon,
            "status": "passed",
            "strategies": parsed,
            "exact_eig": [float(score.eig) for score in cell.exact_scores],
            "root_actions": [str(score.root_action) for score in cell.exact_scores],
            "best_index": best_index,
            "best_root_action": str(cell.exact_scores[best_index].root_action),
            "best_exhaustive_fraction": (
                float(cell.exact_scores[best_index].eig / exhaustive_value)
                if exhaustive_value > 0.0
                else 1.0
            ),
            "move_policy_count": len(move_policies),
            "check_policy_count": len(check_policies),
            "move_then_check_count": sum(
                item["followups"].get("none", "").startswith("check-") for item in move_policies
            ),
            "contingent_check_count": sum(
                item["followups"].get("good") != item["followups"].get("bad")
                and set(item["followups"]) == {"good", "bad"}
                for item in check_policies
            ),
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(concurrency, len(jobs))) as executor:
        futures = {executor.submit(execute, job): job for job in jobs}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row["map_name"], row["state_index"]))
    passed = [row for row in rows if row["status"] == "passed"]
    mechanics = {
        "requested_cells": len(rows),
        "passed_cells": len(passed),
        "terminal_failures": len(rows) - len(passed),
        "raw_rejected_attempts": len(provider.invalid_responses),
        "parse_rate": len(passed) / len(rows),
        "all_cells_have_move_and_check_roots": all(
            row["move_policy_count"] > 0 and row["check_policy_count"] > 0 for row in passed
            if row["horizon"] > 1
        ) and len(passed) == len(rows),
        "all_move_cells_include_move_then_check": all(
            row["move_then_check_count"] > 0 for row in passed
            if row["horizon"] > 1
        ) and len(passed) == len(rows),
    }
    return {
        "schema_version": 1,
        "stage": "rock_branch_strategy_serving_smoke",
        "config": asdict(config),
        "mechanics": mechanics,
        "passed": all(
            [
                mechanics["parse_rate"] == 1.0,
                mechanics["all_cells_have_move_and_check_roots"],
                mechanics["all_move_cells_include_move_then_check"],
            ]
        ),
        "cells": rows,
        "accepted_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


def _usage(model_adapter: Any) -> dict[str, Any]:
    snapshot = getattr(model_adapter, "usage_snapshot", None)
    return snapshot() if callable(snapshot) else {"backend": "unknown"}


def render_report(summary: dict[str, Any]) -> str:
    mechanics = summary["mechanics"]
    lines = [
        "# Rock Branch-Strategy Serving Smoke",
        "",
        "This is an engineering-only interface and proposal-quality probe. It executes no paired policy trajectories.",
        "",
        f"- Passed: `{summary['passed']}`.",
        f"- Parse rate: `{mechanics['passed_cells']}/{mechanics['requested_cells']}`.",
        f"- Raw rejected attempts: `{mechanics['raw_rejected_attempts']}`.",
        f"- Every cell has move and check roots: `{mechanics['all_cells_have_move_and_check_roots']}`.",
        f"- Every cell includes a move-then-check policy: `{mechanics['all_move_cells_include_move_then_check']}`.",
        "",
        "| Map | State | Horizon | Position | Best root | Exhaustive fraction | Move / check policies | Move then check |",
        "| --- | ---: | ---: | --- | --- | ---: | --- | ---: |",
    ]
    for row in summary["cells"]:
        if row["status"] != "passed":
            lines.append(
                f"| {row['map_name']} | {row['state_index']} | {row['horizon']} | {row['position']} | FAILED | - | - | - |"
            )
            continue
        lines.append(
            f"| {row['map_name']} | {row['state_index']} | {row['horizon']} | {row['position']} | "
            f"{row['best_root_action']} | {row['best_exhaustive_fraction']:.3f} | "
            f"{row['move_policy_count']} / {row['check_policy_count']} | "
            f"{row['move_then_check_count']} |"
        )
    lines.extend(["", f"Usage: `{json.dumps(summary.get('usage', {}), sort_keys=True)}`.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_nonmyopic_rock_branch_strategy_openrouter.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/rock_branch_strategy_smoke/20260720"),
    )
    parser.add_argument("--run-id", default="nonmyopic-rock-branch-strategy-smoke-20260720")
    parser.add_argument("--num-strategies", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--maps",
        type=lambda value: tuple(part.strip() for part in value.split(",") if part.strip()),
        default=("3-6", "5-7"),
    )
    parser.add_argument("--probe-states-per-map", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        model_adapter: ChatModel = DeterministicStrategyModel()
    else:
        runtime: Config = load_config(args.config)
        runtime.run_id = args.run_id
        model_adapter = build_model_adapter(runtime.model_pairs[0].questioner, config=runtime)
    summary = run_smoke(
        model_adapter,
        num_strategies=args.num_strategies,
        concurrency=args.concurrency,
        map_names=args.maps,
        probe_states_per_map=args.probe_states_per_map,
    )
    summary["run_id"] = args.run_id
    summary["dry_run"] = args.dry_run
    summary["usage"] = _usage(model_adapter)
    (args.output_dir / "SMOKE.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SMOKE.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "mechanics": summary["mechanics"], "usage": summary["usage"]}, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
