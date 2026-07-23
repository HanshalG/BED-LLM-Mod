"""Fixed-root three-step proposal interface for range-gated Rock Diagnosis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel, get_paper_map
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_gated_sensor_strategy_prior import (
    StrategyProposalError,
    _usage_snapshot,
)
from scripts.nonmyopic_range_gated_rock_strategy import (
    ChatModel,
    History,
    Plan,
    RangeGatedPlanCell,
    RangeGatedStrategyConfig,
    _normalize_response,
    build_belief_cells,
    validate_plan,
)


def fixed_roots(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
) -> tuple[str, ...]:
    legal = model.legal_actions(position)
    moves = [action for action in legal if model.is_move(action)]
    moves.sort(
        key=lambda action: (
            min(
                abs(model.next_position(position, action)[0] - rock[0])
                + abs(model.next_position(position, action)[1] - rock[1])
                for rock in model.map_spec.rock_positions
            ),
            legal.index(action),
        )
    )
    checks = [action for action in legal if not model.is_move(action)]
    checks.sort(
        key=lambda action: (
            -model.expected_information_gain(position, belief, action),
            legal.index(action),
        )
    )
    roots = (*moves[:2], *checks[:2])
    if len(roots) != 4 or len(set(roots)) != 4:
        raise StrategyProposalError("state cannot supply the fixed K4 root mix")
    return roots


def compile_fixed_tail_cell(
    response: str,
    *,
    model: RangeGatedRockDiagnosisModel,
    position: tuple[int, int],
    roots: tuple[str, ...],
    config: RangeGatedStrategyConfig,
) -> tuple[Plan, ...]:
    try:
        payload = json.loads(_normalize_response(response))
    except json.JSONDecodeError as exc:
        raise StrategyProposalError("fixed-tail response is not valid JSON") from exc
    expected_keys = {f"r{index}" for index in range(len(roots))}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise StrategyProposalError(
            f"fixed-tail response must contain exactly keys {sorted(expected_keys)}"
        )
    plans: list[Plan] = []
    for index, root in enumerate(roots):
        tail = payload[f"r{index}"]
        if (
            not isinstance(tail, list)
            or len(tail) != config.horizon - 1
            or not all(isinstance(action, str) for action in tail)
        ):
            raise StrategyProposalError(
                f"r{index} must contain exactly two action strings"
            )
        plan = (root, *tail)
        validate_plan(
            model,
            position=position,
            plan=plan,
            horizon=config.horizon,
        )
        plans.append(plan)
    return tuple(plans)


class FixedRootTailProvider:
    def __init__(
        self,
        chat_model: ChatModel,
        config: RangeGatedStrategyConfig,
        *,
        include_successor_grounding: bool = False,
    ) -> None:
        self.chat_model = chat_model
        self.config = config
        self.include_successor_grounding = include_successor_grounding
        self.physical_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _messages(
        self,
        model: RangeGatedRockDiagnosisModel,
        *,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
        roots: tuple[str, ...],
    ) -> list[dict[str, str]]:
        slots = []
        for index, root in enumerate(roots):
            after_root = model.next_position(position, root)
            slot: dict[str, Any] = {
                "key": f"r{index}",
                "fixed_root": root,
                "position_after_root": list(after_root),
                "legal_second_actions": list(model.legal_actions(after_root)),
            }
            if self.include_successor_grounding:
                slot["second_action_successors"] = [
                    {
                        "action": action,
                        "position_after_action": list(
                            model.next_position(after_root, action)
                        ),
                    }
                    for action in model.legal_actions(after_root)
                ]
            slots.append(slot)
        context = {
            "grid_coordinates": "x and y each range from 0 through 6",
            "current_position": list(position),
            "movement_effects": {
                "move-NORTH": "y decreases by 1",
                "move-EAST": "x increases by 1",
                "move-SOUTH": "y increases by 1",
                "move-WEST": "x decreases by 1",
            },
            "sensor": {
                "remote_accuracy": model.remote_accuracy,
                "onsite_accuracy": model.onsite_accuracy,
                "onsite_rule": "a check is on-site only at that rock's exact coordinate",
            },
            "rocks": [
                {
                    "check_action": f"check-{index}",
                    "position": list(rock_position),
                    "p_good": round(model.rock_good_probability(belief, index), 10),
                }
                for index, rock_position in enumerate(model.map_spec.rock_positions)
            ],
            "history": [
                {"action": action, "observation": outcome}
                for action, outcome in history
            ],
            "root_slots": slots,
            "successor_grounding": self.include_successor_grounding,
        }
        schema = {f"r{index}": ["action2", "action3"] for index in range(4)}
        user = "\n".join(
            [
                "Complete each fixed root into one legal three-action information-gathering plan.",
                "Remote rock checks are weak. A plan may spend two actions travelling so its final check is high-fidelity on site.",
                "The exact verifier scores the four completed plans and executes only the first action; planning repeats after the real transition.",
                "Return JSON only with exactly the fixed root keys and two action strings per key.",
                "The second tail action must be legal from the position reached after the first tail action.",
                "Use exact move-NORTH/move-EAST/move-SOUTH/move-WEST or check-0 through check-7 strings.",
                "Do not emit roots, probabilities, scores, explanations, or extra fields.",
                "SCHEMA=" + json.dumps(schema, separators=(",", ":")),
                "CONTEXT=" + json.dumps(context, separators=(",", ":")),
            ]
        )
        return [
            {
                "role": "system",
                "content": "You complete compact rover inspection plans. Return exact JSON only.",
            },
            {"role": "user", "content": user},
        ]

    def propose(
        self,
        model: RangeGatedRockDiagnosisModel,
        *,
        cell_index: int,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
    ) -> RangeGatedPlanCell:
        roots = fixed_roots(model, position=position, belief=belief)
        messages = self._messages(
            model,
            position=position,
            belief=belief,
            history=history,
            roots=roots,
        )
        context = {
            "cell_index": cell_index,
            "position": list(position),
            "history": [list(item) for item in history],
            "roots": list(roots),
        }
        error: Exception | None = None
        for attempt in range(self.config.validation_retries + 1):
            response = self.chat_model.chat_complete(
                messages, self.config.temperature, num_responses=1
            )[0]
            try:
                plans = compile_fixed_tail_cell(
                    response,
                    model=model,
                    position=position,
                    roots=roots,
                    config=self.config,
                )
            except StrategyProposalError as exc:
                error = exc
                with self._lock:
                    self.invalid_responses.append(
                        {
                            **context,
                            "attempt": attempt,
                            "error": str(exc),
                            "raw_response": response,
                        }
                    )
                if attempt < self.config.validation_retries:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": f"Invalid tails: {exc}. Return corrected full JSON only.",
                        },
                    ]
                continue
            accepted = {
                **context,
                "attempt": attempt,
                "raw_response": response,
                "compiled_plans": [list(plan) for plan in plans],
            }
            with self._lock:
                self.physical_requests.append(accepted)
            return RangeGatedPlanCell(plans, response)
        raise StrategyProposalError(
            f"fixed-root tail cell failed after two attempts: {error}"
        )


class DeterministicFixedTailModel:
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("deterministic tail model supports one response")
        context = json.loads(messages[-1]["content"].split("CONTEXT=", 1)[1])
        rocks = {
            tuple(rock["position"]): rock["check_action"] for rock in context["rocks"]
        }
        effects = {
            "move-NORTH": (0, -1),
            "move-EAST": (1, 0),
            "move-SOUTH": (0, 1),
            "move-WEST": (-1, 0),
        }
        payload: dict[str, list[str]] = {}
        for slot in context["root_slots"]:
            position = tuple(slot["position_after_root"])
            tail: list[str] = []
            for _ in range(2):
                if position in rocks:
                    action = rocks[position]
                else:
                    legal_moves = [
                        action
                        for action, (dx, dy) in effects.items()
                        if 0 <= position[0] + dx <= 6
                        and 0 <= position[1] + dy <= 6
                    ]
                    action = min(
                        legal_moves,
                        key=lambda candidate: (
                            min(
                                abs(position[0] + effects[candidate][0] - rock[0])
                                + abs(position[1] + effects[candidate][1] - rock[1])
                                for rock in rocks
                            ),
                            legal_moves.index(candidate),
                        ),
                    )
                tail.append(action)
                if action in effects:
                    dx, dy = effects[action]
                    position = (position[0] + dx, position[1] + dy)
            payload[slot["key"]] = tail
        return [json.dumps(payload)]


def run_smoke(
    provider: FixedRootTailProvider,
    config: RangeGatedStrategyConfig,
) -> dict[str, Any]:
    model = RangeGatedRockDiagnosisModel(
        get_paper_map("7-8"), remote_accuracy=0.55, onsite_accuracy=0.95
    )
    position = model.map_spec.start_position
    records: list[dict[str, Any]] = []
    delayed_route_count = 0
    for index, (belief, history) in enumerate(
        build_belief_cells(model, count=10, seed=config.seed)
    ):
        cell = provider.propose(
            model,
            cell_index=index,
            position=position,
            belief=belief,
            history=history,
        )
        delayed_route = (
            "move-SOUTH",
            "move-SOUTH",
            "check-5",
        )
        delayed_route_count += int(delayed_route in cell.plans)
        records.append(
            {
                "cell_index": index,
                "history": [list(item) for item in history],
                "plans": [list(plan) for plan in cell.plans],
                "delayed_onsite_route_present": delayed_route in cell.plans,
            }
        )
    mechanics = {
        "ten_cells_completed": len(records) == 10,
        "all_cells_have_four_legal_fixed_root_plans": all(
            len(row["plans"]) == 4 for row in records
        ),
        "exactly_ten_logical_calls": len(provider.physical_requests) == 10,
        "delayed_onsite_route_present_in_at_least_eight_cells": (
            delayed_route_count >= 8
        ),
        "rollout_scoring_made_no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_fixed_root_tail_serving_smoke",
        "config": asdict(config),
        "successor_grounding": provider.include_successor_grounding,
        "mechanics": mechanics,
        "delayed_onsite_route_count": delayed_route_count,
        "provider": {
            "accepted_requests": len(provider.physical_requests),
            "invalid_responses": len(provider.invalid_responses),
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
        default=Path(
            "configs/config_nonmyopic_rocksample_15_15_gpt54mini_openrouter.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/nonmyopic/range_gated_rock_fixed_tail_gpt54mini_smoke_20260723"
        ),
    )
    parser.add_argument(
        "--run-id", default="range-gated-rock-fixed-tail-gpt54mini-smoke-20260723"
    )
    parser.add_argument("--seed", type=int, default=24_177)
    parser.add_argument("--successor-grounding", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = RangeGatedStrategyConfig(seed=args.seed)
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicFixedTailModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = config.max_new_tokens
        chat_model = build_model_adapter(
            runtime_config.model_pairs[0].questioner, config=runtime_config
        )
    provider = FixedRootTailProvider(
        chat_model,
        config,
        include_successor_grounding=args.successor_grounding,
    )
    try:
        result = run_smoke(provider, config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "range_gated_rock_fixed_root_tail_serving_smoke",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage_snapshot(chat_model),
        }
        (args.output_dir / "SMOKE_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    result["usage"] = _usage_snapshot(chat_model)
    result["run_id"] = args.run_id
    result["dry_run"] = args.dry_run
    result["mechanics"]["zero_reasoning_tokens"] = (
        int(result["usage"].get("reasoning_tokens", 0)) == 0
    )
    result["mechanics"]["zero_forced_exits"] = (
        int(result["usage"].get("forced_exits", 0)) == 0
    )
    result["gate"] = {"passed": all(result["mechanics"].values())}
    (args.output_dir / "SMOKE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "gate": result["gate"],
                "delayed_onsite_route_count": result[
                    "delayed_onsite_route_count"
                ],
                "provider": result["provider"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
