"""Fixed-root depth-four proposal gates for corner-start range-gated Rock Diagnosis."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError
from scripts.nonmyopic_range_gated_rock_fixed_tail import (
    fixed_roots,
    usage_with_forced_events,
)
from scripts.nonmyopic_range_gated_rock_proposal_gate import (
    _bootstrap_ci,
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
    _normalize_response,
    build_belief_cells,
    enumerate_legal_plans,
    plan_value,
    validate_plan,
)
from scripts.nonmyopic_rock_depth_oracle import exhaustive_action_values


CRITICAL_ROUTE: Plan = (
    "move-NORTH",
    "move-NORTH",
    "move-NORTH",
    "check-4",
)


@dataclass(frozen=True)
class RangeGatedDepth4StrategyConfig:
    num_plans: int = 4
    horizon: int = 4
    seed: int = 24_208
    temperature: float = 0.0
    validation_retries: int = 1
    max_new_tokens: int = 256

    def validate(self) -> None:
        if self.num_plans != 4 or self.horizon != 4:
            raise ValueError("the frozen depth-four interface uses K4 and horizon four")
        if self.validation_retries != 1:
            raise ValueError("the frozen interface permits exactly one validation retry")
        if self.max_new_tokens != 256:
            raise ValueError("the frozen interface uses a 256-token final output cap")


@dataclass(frozen=True)
class RangeGatedDepth4ProposalConfig:
    num_cells: int = 16
    seed: int = 24_209
    bootstrap_seed: int = 24_210
    bootstrap_replicates: int = 5_000
    route_selection_threshold: float = 0.75
    recovery_threshold: float = 0.60

    def validate(self) -> None:
        if self.num_cells != 16 or self.bootstrap_replicates != 5_000:
            raise ValueError("the frozen h4 proposal gate uses 16 cells and 5,000 bootstraps")
        if not 0.0 <= self.route_selection_threshold <= 1.0:
            raise ValueError("route selection threshold must be a probability")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery threshold must be a fraction")


def build_depth4_model() -> RangeGatedRockDiagnosisModel:
    return RangeGatedRockDiagnosisModel(
        replace(get_paper_map("7-8"), start_position=(6, 6)),
        remote_accuracy=0.55,
        onsite_accuracy=0.95,
    )


def _position_graph(
    model: RangeGatedRockDiagnosisModel,
    *,
    start: tuple[int, int],
    movement_depth: int = 2,
) -> list[dict[str, Any]]:
    positions = {start}
    frontier = {start}
    for _ in range(movement_depth):
        next_frontier: set[tuple[int, int]] = set()
        for position in frontier:
            for action in model.legal_actions(position):
                if model.is_move(action):
                    next_frontier.add(model.next_position(position, action))
        positions.update(next_frontier)
        frontier = next_frontier
    rows: list[dict[str, Any]] = []
    for position in sorted(positions):
        legal = model.legal_actions(position)
        rows.append(
            {
                "position": list(position),
                "legal_actions": list(legal),
                "transitions": [
                    {
                        "action": action,
                        "position_after_action": list(
                            model.next_position(position, action)
                        ),
                    }
                    for action in legal
                ],
            }
        )
    return rows


def compile_depth4_fixed_tail_cell(
    response: str,
    *,
    model: RangeGatedRockDiagnosisModel,
    position: tuple[int, int],
    roots: tuple[str, ...],
    config: RangeGatedDepth4StrategyConfig,
    accept_json_prefix: bool = False,
) -> tuple[Plan, ...]:
    normalized = _normalize_response(response)
    try:
        if accept_json_prefix:
            payload, _end = json.JSONDecoder().raw_decode(normalized)
        else:
            payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise StrategyProposalError("depth-four fixed-tail response is not valid JSON") from exc
    expected_keys = {f"r{index}" for index in range(len(roots))}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise StrategyProposalError(
            f"depth-four response must contain exactly keys {sorted(expected_keys)}"
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
                f"r{index} must contain exactly three action strings"
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


class Depth4FixedRootTailProvider:
    def __init__(
        self,
        chat_model: ChatModel,
        config: RangeGatedDepth4StrategyConfig,
        *,
        accept_json_prefix: bool = True,
    ) -> None:
        self.chat_model = chat_model
        self.config = config
        self.accept_json_prefix = accept_json_prefix
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
            slots.append(
                {
                    "key": f"r{index}",
                    "fixed_root": root,
                    "position_after_root": list(after_root),
                    "reachable_position_graph": _position_graph(
                        model, start=after_root
                    ),
                }
            )
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
        }
        schema = {f"r{index}": ["action2", "action3", "action4"] for index in range(4)}
        user = "\n".join(
            [
                "Complete each fixed root into one legal four-action information-gathering plan.",
                "Remote rock checks are weak. A plan may spend all three remaining actions travelling so its final check is high-fidelity on site.",
                "The exact verifier scores the four completed plans and executes only the first action; planning repeats after the real transition.",
                "Return JSON only with exactly the fixed root keys and three action strings per key.",
                "Every tail action must be legal from the position reached by the preceding actions.",
                "The reachable-position graph lists geometry and legality only; it contains no utility scores.",
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
                plans = compile_depth4_fixed_tail_cell(
                    response,
                    model=model,
                    position=position,
                    roots=roots,
                    config=self.config,
                    accept_json_prefix=self.accept_json_prefix,
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
            f"depth-four fixed-root tail cell failed after two attempts: {error}"
        )


class DeterministicDepth4TailModel:
    def chat_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        num_responses: int = 1,
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
            for _ in range(3):
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


def matched_random_depth4_plans(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    seed: int,
) -> tuple[Plan, ...]:
    rng = np.random.default_rng(seed)
    plans: list[Plan] = []
    for root in fixed_roots(model, position=position, belief=belief):
        plan = [root]
        current = model.next_position(position, root)
        for _ in range(3):
            legal = model.legal_actions(current)
            action = legal[int(rng.integers(len(legal)))]
            plan.append(action)
            current = model.next_position(current, action)
        plans.append(tuple(plan))
    return tuple(plans)


def _summary(
    values: list[float],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    label: str,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "ci95": _bootstrap_ci(
            array,
            seed=_stable_seed(bootstrap_seed, "range-gated-h4-bootstrap", label),
            replicates=bootstrap_replicates,
        ),
        "wins_ties_losses": [
            int(np.sum(array > 1e-12)),
            int(np.sum(np.abs(array) <= 1e-12)),
            int(np.sum(array < -1e-12)),
        ],
        "values": array.tolist(),
    }


def _score_cell(
    model: RangeGatedRockDiagnosisModel,
    *,
    cell_index: int,
    belief: np.ndarray,
    history: History,
    llm_plans: tuple[Plan, ...],
    config: RangeGatedDepth4ProposalConfig,
    exhaustive_plans: tuple[Plan, ...],
) -> dict[str, Any]:
    position = model.map_spec.start_position
    roots = fixed_roots(model, position=position, belief=belief)
    if tuple(plan[0] for plan in llm_plans) != roots:
        raise RuntimeError("compiled plans do not preserve the machine-fixed roots")
    llm_plan, llm_value, llm_values = stable_best_plan(
        model, position=position, belief=belief, plans=llm_plans
    )
    random_plans = matched_random_depth4_plans(
        model,
        position=position,
        belief=belief,
        seed=_stable_seed(config.seed, "depth4-fixed-root-random", cell_index),
    )
    random_plan, random_value, random_values = stable_best_plan(
        model, position=position, belief=belief, plans=random_plans
    )
    truncated = tuple(plan[:3] for plan in llm_plans)
    shared_plan, _shared_h3_value, shared_values = stable_best_plan(
        model, position=position, belief=belief, plans=truncated
    )
    shared_full_plan = llm_plans[truncated.index(shared_plan)]
    shared_full_value = plan_value(
        model, position=position, belief=belief, plan=shared_full_plan
    )
    d3_values, _ = exhaustive_action_values(
        model, position=position, belief=belief, depth=3
    )
    legal = model.legal_actions(position)
    d3_vector = [d3_values[action] for action in legal]
    d3_root = legal[stable_best_index(d3_vector)]
    strong_d3_plans = tuple(
        plan for plan in exhaustive_plans if plan[0] == d3_root
    )
    strong_d3_plan, strong_d3_value, strong_d3_values = stable_best_plan(
        model, position=position, belief=belief, plans=strong_d3_plans
    )
    exact_plan, exact_value, _exact_values = stable_best_plan(
        model, position=position, belief=belief, plans=exhaustive_plans
    )
    opportunity = exact_value - strong_d3_value
    if opportunity <= 1e-12 or exact_plan != CRITICAL_ROUTE:
        raise RuntimeError("cell is not the registered strict depth-four route opportunity")
    return {
        "cell_index": cell_index,
        "history": [list(item) for item in history],
        "fixed_roots": list(roots),
        "llm_plans": [list(plan) for plan in llm_plans],
        "llm_plan_values": llm_values,
        "llm_selected_plan": list(llm_plan),
        "llm_value": llm_value,
        "matched_random_plans": [list(plan) for plan in random_plans],
        "matched_random_plan_values": random_values,
        "matched_random_selected_plan": list(random_plan),
        "matched_random_value": random_value,
        "shared_h3_plan_values": shared_values,
        "shared_h3_selected_full_plan": list(shared_full_plan),
        "shared_h3_selected_full_value": shared_full_value,
        "strong_d3_root": d3_root,
        "strong_d3_h4_plan": list(strong_d3_plan),
        "strong_d3_h4_plan_values": strong_d3_values,
        "strong_d3_h4_value": strong_d3_value,
        "exact_h4_plan": list(exact_plan),
        "exact_h4_value": exact_value,
        "h4_opportunity": opportunity,
        "recovery_fraction": (llm_value - strong_d3_value) / opportunity,
    }


def run_smoke(
    provider: Depth4FixedRootTailProvider,
    strategy_config: RangeGatedDepth4StrategyConfig,
) -> dict[str, Any]:
    model = build_depth4_model()
    position = model.map_spec.start_position
    exhaustive_plans = enumerate_legal_plans(
        model, position=position, horizon=4
    )
    records: list[dict[str, Any]] = []
    for cell_index, (belief, history) in enumerate(
        build_belief_cells(model, count=10, seed=strategy_config.seed)
    ):
        cell = provider.propose(
            model,
            cell_index=cell_index,
            position=position,
            belief=belief,
            history=history,
        )
        selected, _value, _values = stable_best_plan(
            model, position=position, belief=belief, plans=cell.plans
        )
        exact, _exact_value, _exact_values = stable_best_plan(
            model, position=position, belief=belief, plans=exhaustive_plans
        )
        records.append(
            {
                "cell_index": cell_index,
                "history": [list(item) for item in history],
                "plans": [list(plan) for plan in cell.plans],
                "critical_route_present": CRITICAL_ROUTE in cell.plans,
                "selected_plan": list(selected),
                "exact_h4_plan": list(exact),
                "selected_root_matches_exact": selected[0] == exact[0],
            }
        )
    route_count = sum(row["critical_route_present"] for row in records)
    root_match_count = sum(row["selected_root_matches_exact"] for row in records)
    mechanics = {
        "ten_distinct_cells_completed": len(records) == 10
        and len(
            {tuple(tuple(item) for item in row["history"]) for row in records}
        )
        == 10,
        "all_cells_have_four_legal_fixed_root_plans": all(
            len(row["plans"]) == 4 for row in records
        ),
        "exactly_ten_accepted_calls": len(provider.physical_requests) == 10,
        "critical_route_present_in_at_least_eight_cells": route_count >= 8,
        "exact_h4_root_selected_in_at_least_eight_cells": root_match_count >= 8,
        "scoring_made_no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_depth4_fixed_tail_serving_smoke",
        "config": asdict(strategy_config),
        "mechanics": mechanics,
        "critical_route_count": route_count,
        "exact_root_match_count": root_match_count,
        "records": records,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


def run_proposal_gate(
    provider: Depth4FixedRootTailProvider,
    strategy_config: RangeGatedDepth4StrategyConfig,
    gate_config: RangeGatedDepth4ProposalConfig,
) -> dict[str, Any]:
    strategy_config.validate()
    gate_config.validate()
    model = build_depth4_model()
    position = model.map_spec.start_position
    exhaustive_plans = enumerate_legal_plans(
        model, position=position, horizon=4
    )
    records: list[dict[str, Any]] = []
    for cell_index, (belief, history) in enumerate(
        build_belief_cells(model, count=gate_config.num_cells, seed=gate_config.seed)
    ):
        cell = provider.propose(
            model,
            cell_index=cell_index,
            position=position,
            belief=belief,
            history=history,
        )
        records.append(
            _score_cell(
                model,
                cell_index=cell_index,
                belief=belief,
                history=history,
                llm_plans=cell.plans,
                config=gate_config,
                exhaustive_plans=exhaustive_plans,
            )
        )
    summary_kwargs = {
        "bootstrap_seed": gate_config.bootstrap_seed,
        "bootstrap_replicates": gate_config.bootstrap_replicates,
    }
    comparisons = {
        "llm_minus_matched_random": _summary(
            [row["llm_value"] - row["matched_random_value"] for row in records],
            label="llm-minus-matched-random",
            **summary_kwargs,
        ),
        "llm_minus_shared_h3": _summary(
            [
                row["llm_value"] - row["shared_h3_selected_full_value"]
                for row in records
            ],
            label="llm-minus-shared-h3",
            **summary_kwargs,
        ),
        "llm_minus_strong_d3": _summary(
            [row["llm_value"] - row["strong_d3_h4_value"] for row in records],
            label="llm-minus-strong-d3",
            **summary_kwargs,
        ),
        "recovery_fraction": _summary(
            [row["recovery_fraction"] for row in records],
            label="recovery-fraction",
            **summary_kwargs,
        ),
        "exact_h4_root_selection_rate": float(
            np.mean(
                [
                    row["llm_selected_plan"][0] == row["exact_h4_plan"][0]
                    for row in records
                ]
            )
        ),
        "exact_h4_route_selection_rate": float(
            np.mean(
                [row["llm_selected_plan"] == row["exact_h4_plan"] for row in records]
            )
        ),
    }
    mechanics = {
        "sixteen_distinct_cells_resolved": len(records) == gate_config.num_cells
        and len(
            {tuple(tuple(item) for item in row["history"]) for row in records}
        )
        == gate_config.num_cells,
        "all_cells_are_strict_h4_opportunities": all(
            row["h4_opportunity"] > 1e-12 for row in records
        ),
        "all_exact_h4_plans_are_critical_route": all(
            row["exact_h4_plan"] == list(CRITICAL_ROUTE) for row in records
        ),
        "all_roots_match_fixed_interface": all(
            [plan[0] for plan in row["llm_plans"]] == row["fixed_roots"]
            for row in records
        ),
        "all_controls_exactly_scored": True,
        "scoring_made_no_llm_calls": True,
    }
    endpoint_gate = {
        "matched_random_lower_bound_positive": comparisons[
            "llm_minus_matched_random"
        ]["ci95"][0]
        > 0.0,
        "shared_h3_lower_bound_positive": comparisons["llm_minus_shared_h3"][
            "ci95"
        ][0]
        > 0.0,
        "strong_d3_lower_bound_positive": comparisons["llm_minus_strong_d3"][
            "ci95"
        ][0]
        > 0.0,
        "route_selection_at_least_threshold": comparisons[
            "exact_h4_route_selection_rate"
        ]
        >= gate_config.route_selection_threshold,
        "mean_recovery_at_least_threshold": comparisons["recovery_fraction"][
            "mean"
        ]
        >= gate_config.recovery_threshold,
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_depth4_fixed_tail_proposal_gate",
        "strategy_config": asdict(strategy_config),
        "config": asdict(gate_config),
        "mechanics": mechanics,
        "comparisons": comparisons,
        "endpoint_gate": endpoint_gate,
        "records": records,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Range-Gated Rock Depth-Four Fixed-Tail Gate",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint | Mean | 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("llm_minus_matched_random", "LLM h4 - identical-root random h4"),
        ("llm_minus_shared_h3", "LLM h4 - shared-plan h3"),
        ("llm_minus_strong_d3", "LLM h4 - strongest exact d3 root"),
        ("recovery_fraction", "Exact h4 opportunity recovery"),
    ):
        row = result["comparisons"][key]
        lines.append(
            f"| {label} | {row['mean']:+.6f} | "
            f"[{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] | "
            f"{'/'.join(str(value) for value in row['wins_ties_losses'])} |"
        )
    lines.extend(
        [
            "",
            f"- Exact h4 route selection: "
            f"`{result['comparisons']['exact_h4_route_selection_rate']:.3f}`.",
            f"- Mechanics: `{result['mechanics']}`.",
            f"- Endpoint gates: `{result['endpoint_gate']}`.",
            f"- Usage: `{result['usage']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "proposal"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=24_210)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-generation-tokens", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    default_seed = 24_208 if args.stage == "smoke" else 24_209
    seed = args.seed if args.seed is not None else default_seed
    strategy_config = RangeGatedDepth4StrategyConfig(seed=seed)
    strategy_config.validate()
    gate_config = RangeGatedDepth4ProposalConfig(
        seed=seed,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicDepth4TailModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.log_path = args.output_dir / "run.log"
        runtime_config.location_max_new_tokens = args.model_generation_tokens
        chat_model = build_model_adapter(
            runtime_config.model_pairs[0].questioner,
            config=runtime_config,
        )
    provider = Depth4FixedRootTailProvider(chat_model, strategy_config)
    try:
        if args.stage == "smoke":
            result = run_smoke(provider, strategy_config)
        else:
            result = run_proposal_gate(provider, strategy_config, gate_config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": f"range_gated_rock_depth4_{args.stage}",
            "status": "failed_closed",
            "error": str(exc),
            "strategy_config": asdict(strategy_config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": usage_with_forced_events(chat_model, args.output_dir),
        }
        (args.output_dir / f"{args.stage.upper()}_FAILURE.json").write_text(
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
    if args.stage == "smoke":
        result["gate"] = {"passed": all(result["mechanics"].values())}
        stem = "SMOKE"
    else:
        result["gate"] = {
            "passed": all(result["mechanics"].values())
            and all(result["endpoint_gate"].values())
        }
        stem = "PROPOSAL"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.stage == "proposal":
        (args.output_dir / "PROPOSAL.md").write_text(
            render_report(result), encoding="utf-8"
        )
    print(json.dumps({"gate": result["gate"], "usage": result["usage"]}, indent=2))


if __name__ == "__main__":
    main()
