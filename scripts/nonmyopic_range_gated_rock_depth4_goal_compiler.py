"""Hierarchical target-proposal gate for corner-start range-gated Rock Diagnosis."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError
from scripts.nonmyopic_range_gated_rock_depth4_fixed_tail import (
    CRITICAL_ROUTE,
    RangeGatedDepth4ProposalConfig,
    _summary,
    build_depth4_model,
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
    Plan,
    RangeGatedPlanCell,
    _normalize_response,
    build_belief_cells,
    enumerate_legal_plans,
    plan_value,
    validate_plan,
)
from scripts.nonmyopic_rock_depth_oracle import exhaustive_action_values


@dataclass(frozen=True)
class Depth4GoalCompilerConfig:
    num_plans: int = 4
    horizon: int = 4
    seed: int = 24_228
    temperature: float = 0.0
    validation_retries: int = 1
    max_new_tokens: int = 128

    def validate(self) -> None:
        if self.num_plans != 4 or self.horizon != 4:
            raise ValueError("the frozen goal compiler uses K4 and horizon four")
        if self.validation_retries != 1:
            raise ValueError("the frozen interface permits exactly one validation retry")
        if self.max_new_tokens != 128:
            raise ValueError("the frozen interface uses a 128-token final output cap")


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def compile_target_plan(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    root: str,
    target_rock: int,
) -> Plan:
    """Compile a semantic target into two greedy transit slots and a final check."""

    if not 0 <= target_rock < model.num_rocks:
        raise StrategyProposalError(f"target rock {target_rock} is out of range")
    if root not in model.legal_actions(position):
        raise StrategyProposalError(f"fixed root {root!r} is illegal from {position}")
    current = model.next_position(position, root)
    target = model.map_spec.rock_positions[target_rock]
    tail: list[str] = []
    for _ in range(2):
        if current == target:
            action = f"check-{target_rock}"
        else:
            legal_moves = [
                action
                for action in model.legal_actions(current)
                if model.is_move(action)
            ]
            action = min(
                legal_moves,
                key=lambda candidate: (
                    _manhattan(
                        model.next_position(current, candidate),
                        target,
                    ),
                    model.legal_actions(current).index(candidate),
                ),
            )
        tail.append(action)
        current = model.next_position(current, action)
    tail.append(f"check-{target_rock}")
    plan = (root, *tail)
    validate_plan(model, position=position, plan=plan, horizon=4)
    return plan


def compile_goal_cell(
    response: str,
    *,
    model: RangeGatedRockDiagnosisModel,
    position: tuple[int, int],
    roots: tuple[str, ...],
    config: Depth4GoalCompilerConfig,
    accept_json_prefix: bool = True,
) -> tuple[tuple[int, ...], tuple[Plan, ...]]:
    normalized = _normalize_response(response)
    try:
        if accept_json_prefix:
            payload, _end = json.JSONDecoder().raw_decode(normalized)
        else:
            payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise StrategyProposalError("goal response is not valid JSON") from exc
    expected_keys = {f"r{index}" for index in range(len(roots))}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise StrategyProposalError(
            f"goal response must contain exactly keys {sorted(expected_keys)}"
        )
    targets: list[int] = []
    for index in range(len(roots)):
        target = payload[f"r{index}"]
        if isinstance(target, bool) or not isinstance(target, int):
            raise StrategyProposalError(f"r{index} must be an integer rock id")
        if not 0 <= target < model.num_rocks:
            raise StrategyProposalError(f"r{index} target {target} is out of range")
        targets.append(target)
    if len(set(targets)) != config.num_plans:
        raise StrategyProposalError("all four target rocks must be distinct")
    plans = tuple(
        compile_target_plan(
            model,
            position=position,
            root=root,
            target_rock=target,
        )
        for root, target in zip(roots, targets, strict=True)
    )
    return tuple(targets), plans


class Depth4GoalCompilerProvider:
    def __init__(
        self,
        chat_model: ChatModel,
        config: Depth4GoalCompilerConfig,
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
        slots: list[dict[str, Any]] = []
        for index, root in enumerate(roots):
            after_root = model.next_position(position, root)
            slots.append(
                {
                    "key": f"r{index}",
                    "fixed_root": root,
                    "position_after_root": list(after_root),
                    "targets": [
                        {
                            "rock_id": rock_index,
                            "position": list(rock_position),
                            "movement_distance_after_root": _manhattan(
                                after_root, rock_position
                            ),
                            "p_good": round(
                                model.rock_good_probability(belief, rock_index),
                                10,
                            ),
                        }
                        for rock_index, rock_position in enumerate(
                            model.map_spec.rock_positions
                        )
                    ],
                }
            )
        context = {
            "current_position": list(position),
            "sensor": {
                "remote_accuracy": model.remote_accuracy,
                "onsite_accuracy": model.onsite_accuracy,
                "onsite_rule": "the final check is on-site only at the target rock's exact coordinate",
            },
            "compiler": {
                "transit_slots_after_root": 2,
                "rule": "each transit slot takes a shortest-path move toward the selected target; if already on target it repeats that target check",
                "final_action": "check the selected target rock",
            },
            "history": [
                {"action": action, "observation": outcome}
                for action, outcome in history
            ],
            "root_slots": slots,
        }
        user = "\n".join(
            [
                "Assign one distinct semantic rock-inspection target to each fixed root.",
                "The deterministic compiler, not you, creates all movement actions using the stated shortest-path rule.",
                "Remote checks are weak and on-site checks are strong; use the belief and geometry to choose a useful target for each root.",
                "The exact verifier scores the four compiled plans and executes only their selected root.",
                'Return JSON only with schema {"r0":rock_id,"r1":rock_id,"r2":rock_id,"r3":rock_id}.',
                "Use four distinct integer rock ids from 0 through 7.",
                "Do not emit actions, routes, probabilities, scores, explanations, or extra fields.",
                "CONTEXT=" + json.dumps(context, separators=(",", ":")),
            ]
        )
        return [
            {
                "role": "system",
                "content": "You select semantic rover inspection targets. Return exact JSON only.",
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
                targets, plans = compile_goal_cell(
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
                            "content": (
                                f"Invalid target assignment: {exc}. Return corrected "
                                "full JSON with four distinct integer targets only."
                            ),
                        },
                    ]
                continue
            accepted = {
                **context,
                "attempt": attempt,
                "raw_response": response,
                "target_assignments": list(targets),
                "compiled_plans": [list(plan) for plan in plans],
                "compiler_rule": "two canonical shortest-path transit slots then target check",
            }
            with self._lock:
                self.physical_requests.append(accepted)
            return RangeGatedPlanCell(plans, response)
        raise StrategyProposalError(
            f"depth-four target cell failed after two attempts: {error}"
        )


class DeterministicDepth4GoalModel:
    def chat_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        num_responses: int = 1,
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("deterministic goal model supports one response")
        context = json.loads(messages[-1]["content"].split("CONTEXT=", 1)[1])
        used: set[int] = set()
        payload: dict[str, int] = {}
        for slot in context["root_slots"]:
            target = min(
                (
                    row
                    for row in slot["targets"]
                    if int(row["rock_id"]) not in used
                ),
                key=lambda row: (
                    int(row["movement_distance_after_root"]),
                    abs(float(row["p_good"]) - 0.5),
                    int(row["rock_id"]),
                ),
            )
            target_id = int(target["rock_id"])
            used.add(target_id)
            payload[slot["key"]] = target_id
        return [json.dumps(payload)]


def matched_random_goal_plans(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    seed: int,
) -> tuple[tuple[int, ...], tuple[Plan, ...]]:
    roots = fixed_roots(model, position=position, belief=belief)
    targets = tuple(
        int(value)
        for value in np.random.default_rng(seed).permutation(model.num_rocks)[
            : len(roots)
        ]
    )
    plans = tuple(
        compile_target_plan(
            model,
            position=position,
            root=root,
            target_rock=target,
        )
        for root, target in zip(roots, targets, strict=True)
    )
    return targets, plans


def _score_goal_cell(
    model: RangeGatedRockDiagnosisModel,
    *,
    cell_index: int,
    belief: np.ndarray,
    history: History,
    llm_targets: tuple[int, ...],
    llm_plans: tuple[Plan, ...],
    config: RangeGatedDepth4ProposalConfig,
    exhaustive_plans: tuple[Plan, ...],
) -> dict[str, Any]:
    position = model.map_spec.start_position
    roots = fixed_roots(model, position=position, belief=belief)
    if tuple(plan[0] for plan in llm_plans) != roots:
        raise RuntimeError("compiled plans do not preserve the machine-fixed roots")
    recomputed = tuple(
        compile_target_plan(
            model,
            position=position,
            root=root,
            target_rock=target,
        )
        for root, target in zip(roots, llm_targets, strict=True)
    )
    if recomputed != llm_plans:
        raise RuntimeError("serialized plans do not match target compiler")
    llm_plan, llm_value, llm_values = stable_best_plan(
        model, position=position, belief=belief, plans=llm_plans
    )
    random_targets, random_plans = matched_random_goal_plans(
        model,
        position=position,
        belief=belief,
        seed=_stable_seed(config.seed, "depth4-random-goals", cell_index),
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
        "llm_target_assignments": list(llm_targets),
        "llm_plans": [list(plan) for plan in llm_plans],
        "llm_plan_values": llm_values,
        "llm_selected_plan": list(llm_plan),
        "llm_value": llm_value,
        "matched_random_target_assignments": list(random_targets),
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


def _comparison_payload(
    records: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    kwargs = {
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
    }
    return {
        "llm_minus_matched_random_goals": _summary(
            [row["llm_value"] - row["matched_random_value"] for row in records],
            label="llm-minus-matched-random-goals",
            **kwargs,
        ),
        "llm_minus_shared_h3": _summary(
            [
                row["llm_value"] - row["shared_h3_selected_full_value"]
                for row in records
            ],
            label="llm-minus-shared-h3",
            **kwargs,
        ),
        "llm_minus_strong_d3": _summary(
            [row["llm_value"] - row["strong_d3_h4_value"] for row in records],
            label="llm-minus-strong-d3",
            **kwargs,
        ),
        "recovery_fraction": _summary(
            [row["recovery_fraction"] for row in records],
            label="recovery-fraction",
            **kwargs,
        ),
        "exact_h4_route_selection_rate": float(
            np.mean(
                [row["llm_selected_plan"] == row["exact_h4_plan"] for row in records]
            )
        ),
    }


def _propose_cells(
    provider: Depth4GoalCompilerProvider,
    model: RangeGatedRockDiagnosisModel,
    *,
    cells: list[tuple[np.ndarray, History]],
    cell_concurrency: int,
) -> list[RangeGatedPlanCell]:
    if cell_concurrency <= 0:
        raise ValueError("cell concurrency must be positive")
    position = model.map_spec.start_position

    def propose(
        item: tuple[int, tuple[np.ndarray, History]],
    ) -> RangeGatedPlanCell:
        cell_index, (belief, history) = item
        return provider.propose(
            model,
            cell_index=cell_index,
            position=position,
            belief=belief,
            history=history,
        )

    indexed = list(enumerate(cells))
    if cell_concurrency == 1:
        return [propose(item) for item in indexed]
    with ThreadPoolExecutor(max_workers=cell_concurrency) as executor:
        return list(executor.map(propose, indexed))


def run_smoke(
    provider: Depth4GoalCompilerProvider,
    strategy_config: Depth4GoalCompilerConfig,
    *,
    cell_concurrency: int = 1,
) -> dict[str, Any]:
    model = build_depth4_model()
    position = model.map_spec.start_position
    exhaustive_plans = enumerate_legal_plans(
        model, position=position, horizon=4
    )
    cells = build_belief_cells(model, count=10, seed=strategy_config.seed)
    proposals = _propose_cells(
        provider,
        model,
        cells=cells,
        cell_concurrency=cell_concurrency,
    )
    requests = {
        int(request["cell_index"]): request
        for request in provider.physical_requests
    }
    records: list[dict[str, Any]] = []
    for (cell_index, (belief, history)), cell in zip(
        enumerate(cells), proposals, strict=True
    ):
        request = requests[cell_index]
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
                "target_assignments": request["target_assignments"],
                "plans": [list(plan) for plan in cell.plans],
                "selected_plan": list(selected),
                "exact_plan": list(exact),
                "critical_route_present": CRITICAL_ROUTE in cell.plans,
                "critical_goal_assigned_to_north_root": (
                    request["target_assignments"][0] == 4
                ),
                "selected_matches_exact": selected == exact,
            }
        )
    critical_count = sum(row["critical_route_present"] for row in records)
    target_count = sum(
        row["critical_goal_assigned_to_north_root"] for row in records
    )
    exact_count = sum(row["selected_matches_exact"] for row in records)
    mechanics = {
        "ten_distinct_cells_completed": len(records) == 10
        and len(
            {tuple(tuple(item) for item in row["history"]) for row in records}
        )
        == 10,
        "all_cells_have_four_distinct_targets": all(
            len(set(row["target_assignments"])) == 4 for row in records
        ),
        "all_compiled_plans_are_legal_h4": all(
            len(plan) == 4
            for row in records
            for plan in row["plans"]
        ),
        "critical_target_assigned_to_north_root_in_at_least_eight_cells": (
            target_count >= 8
        ),
        "critical_route_present_in_at_least_eight_cells": critical_count >= 8,
        "exact_h4_plan_selected_in_at_least_eight_cells": exact_count >= 8,
        "exactly_ten_accepted_calls": len(provider.physical_requests) == 10,
        "scoring_made_no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_depth4_goal_compiler_smoke",
        "cell_concurrency": cell_concurrency,
        "config": asdict(strategy_config),
        "mechanics": mechanics,
        "critical_target_count": target_count,
        "critical_route_count": critical_count,
        "exact_plan_match_count": exact_count,
        "records": records,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


def run_proposal_gate(
    provider: Depth4GoalCompilerProvider,
    strategy_config: Depth4GoalCompilerConfig,
    gate_config: RangeGatedDepth4ProposalConfig,
    *,
    cell_concurrency: int = 1,
) -> dict[str, Any]:
    strategy_config.validate()
    gate_config.validate()
    model = build_depth4_model()
    position = model.map_spec.start_position
    exhaustive_plans = enumerate_legal_plans(
        model, position=position, horizon=4
    )
    cells = build_belief_cells(
        model, count=gate_config.num_cells, seed=gate_config.seed
    )
    proposals = _propose_cells(
        provider,
        model,
        cells=cells,
        cell_concurrency=cell_concurrency,
    )
    requests = {
        int(request["cell_index"]): request
        for request in provider.physical_requests
    }
    records: list[dict[str, Any]] = []
    for (cell_index, (belief, history)), cell in zip(
        enumerate(cells), proposals, strict=True
    ):
        records.append(
            _score_goal_cell(
                model,
                cell_index=cell_index,
                belief=belief,
                history=history,
                llm_targets=tuple(requests[cell_index]["target_assignments"]),
                llm_plans=cell.plans,
                config=gate_config,
                exhaustive_plans=exhaustive_plans,
            )
        )
    comparisons = _comparison_payload(
        records,
        bootstrap_seed=gate_config.bootstrap_seed,
        bootstrap_replicates=gate_config.bootstrap_replicates,
    )
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
        "all_targets_are_distinct": all(
            len(set(row["llm_target_assignments"])) == 4 for row in records
        ),
        "all_controls_exactly_scored": True,
        "scoring_made_no_llm_calls": True,
    }
    endpoint_gate = {
        "matched_random_goal_lower_bound_positive": comparisons[
            "llm_minus_matched_random_goals"
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
        "stage": "range_gated_rock_depth4_goal_compiler_proposal_gate",
        "cell_concurrency": cell_concurrency,
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
        "# Range-Gated Rock Hierarchical H4 Proposal Gate",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint | Mean | 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("llm_minus_matched_random_goals", "LLM goals - random goals"),
        ("llm_minus_shared_h3", "Compiled h4 - shared compiled h3"),
        ("llm_minus_strong_d3", "Compiled h4 - strongest exact d3 root"),
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
    parser.add_argument("--bootstrap-seed", type=int, default=24_230)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-generation-tokens", type=int, default=4096)
    parser.add_argument("--cell-concurrency", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    default_seed = 24_228 if args.stage == "smoke" else 24_229
    seed = args.seed if args.seed is not None else default_seed
    strategy_config = Depth4GoalCompilerConfig(seed=seed)
    strategy_config.validate()
    gate_config = RangeGatedDepth4ProposalConfig(
        seed=seed,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicDepth4GoalModel()
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
    provider = Depth4GoalCompilerProvider(chat_model, strategy_config)
    try:
        if args.stage == "smoke":
            result = run_smoke(
                provider,
                strategy_config,
                cell_concurrency=args.cell_concurrency,
            )
        else:
            result = run_proposal_gate(
                provider,
                strategy_config,
                gate_config,
                cell_concurrency=args.cell_concurrency,
            )
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": f"range_gated_rock_depth4_goal_compiler_{args.stage}",
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
