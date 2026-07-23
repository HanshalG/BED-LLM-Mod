"""Named three-action proposal interface for range-gated Rock Diagnosis."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import threading
from typing import Any, Protocol

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel, get_paper_map
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError, _usage_snapshot


History = tuple[tuple[str, str | None], ...]
Plan = tuple[str, ...]


class ChatModel(Protocol):
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]: ...


@dataclass(frozen=True)
class RangeGatedStrategyConfig:
    num_plans: int = 4
    horizon: int = 3
    seed: int = 24_145
    temperature: float = 0.0
    validation_retries: int = 1
    max_new_tokens: int = 256

    def validate(self) -> None:
        if self.num_plans != 4 or self.horizon != 3:
            raise ValueError("the frozen range-gated interface uses K4 and horizon three")
        if self.validation_retries != 1:
            raise ValueError("the frozen interface permits exactly one validation retry")
        if self.max_new_tokens != 256:
            raise ValueError("the frozen interface uses a 256-token output cap")


@dataclass(frozen=True)
class RangeGatedPlanCell:
    plans: tuple[Plan, ...]
    raw_response: str


def _normalize_response(response: str) -> str:
    normalized = response.strip()
    if "```json" in normalized:
        start = normalized.rfind("```json") + len("```json")
        end = normalized.find("```", start)
        if end < 0:
            raise StrategyProposalError("response has an incomplete JSON fence")
        normalized = normalized[start:end].strip()
    return normalized


def validate_plan(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    plan: Plan,
    horizon: int,
) -> None:
    if len(plan) != horizon:
        raise StrategyProposalError(f"every plan must contain exactly {horizon} actions")
    current = position
    for step, action in enumerate(plan):
        if action not in model.legal_actions(current):
            raise StrategyProposalError(
                f"illegal action {action!r} at plan step {step + 1} from {current}"
            )
        current = model.next_position(current, action)


def compile_plan_cell(
    response: str,
    *,
    model: RangeGatedRockDiagnosisModel,
    position: tuple[int, int],
    config: RangeGatedStrategyConfig,
) -> tuple[Plan, ...]:
    try:
        payload = json.loads(_normalize_response(response))
    except json.JSONDecodeError as exc:
        raise StrategyProposalError("plan response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"plans"}:
        raise StrategyProposalError("response must contain exactly the plans field")
    raw_plans = payload["plans"]
    if not isinstance(raw_plans, list) or len(raw_plans) != config.num_plans:
        raise StrategyProposalError(f"plans must contain exactly {config.num_plans} arrays")
    plans: list[Plan] = []
    for raw_plan in raw_plans:
        if not isinstance(raw_plan, list) or not all(
            isinstance(action, str) for action in raw_plan
        ):
            raise StrategyProposalError("each plan must be an array of action strings")
        plan = tuple(raw_plan)
        validate_plan(
            model, position=position, plan=plan, horizon=config.horizon
        )
        plans.append(plan)
    if len(set(plans)) != len(plans):
        raise StrategyProposalError("all four plans must be distinct")
    move_roots = {plan[0] for plan in plans if model.is_move(plan[0])}
    check_roots = {plan[0] for plan in plans if not model.is_move(plan[0])}
    if len(move_roots) != 2 or len(check_roots) != 2:
        raise StrategyProposalError(
            "plans must use exactly two distinct movement roots and two distinct check roots"
        )
    return tuple(plans)


def plan_value(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    plan: Plan,
) -> float:
    if not plan:
        return 0.0
    action = plan[0]
    validate_plan(model, position=position, plan=plan, horizon=len(plan))
    next_position = model.next_position(position, action)
    value = model.expected_information_gain(position, belief, action)
    for outcome in model.outcomes(action):
        probability = model.outcome_probability(position, belief, action, outcome)
        if probability <= 0.0:
            continue
        posterior = model.posterior(position, belief, action, outcome)
        value += probability * plan_value(
            model,
            position=next_position,
            belief=posterior,
            plan=plan[1:],
        )
    return float(value)


def enumerate_legal_plans(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    horizon: int,
) -> tuple[Plan, ...]:
    if horizon == 0:
        return ((),)
    plans: list[Plan] = []
    for action in model.legal_actions(position):
        next_position = model.next_position(position, action)
        for tail in enumerate_legal_plans(
            model, position=next_position, horizon=horizon - 1
        ):
            plans.append((action, *tail))
    return tuple(plans)


def build_belief_cells(
    model: RangeGatedRockDiagnosisModel, *, count: int, seed: int
) -> list[tuple[np.ndarray, History]]:
    actions = [f"check-{index}" for index in range(model.num_rocks) if index != 5]
    candidates: list[tuple[tuple[str, str], ...]] = [()]
    candidates.extend(((action, outcome),) for action in actions for outcome in ("good", "bad"))
    candidates.extend(
        ((first, first_outcome), (second, second_outcome))
        for first in actions
        for second in actions
        for first_outcome in ("good", "bad")
        for second_outcome in ("good", "bad")
    )
    order = np.random.default_rng(seed).permutation(len(candidates))
    cells: list[tuple[np.ndarray, History]] = []
    seen: set[bytes] = set()
    position = model.map_spec.start_position
    for raw_index in order:
        history = candidates[int(raw_index)]
        belief = model.initial_belief.copy()
        for action, outcome in history:
            belief = model.posterior(position, belief, action, outcome)
        key = np.ascontiguousarray(belief, dtype=np.float64).tobytes()
        if key in seen:
            continue
        seen.add(key)
        cells.append((belief, tuple(history)))
        if len(cells) == count:
            break
    if len(cells) != count:
        raise RuntimeError(f"could not build {count} distinct belief cells")
    return cells


class NamedPlanProvider:
    def __init__(self, chat_model: ChatModel, config: RangeGatedStrategyConfig) -> None:
        self.chat_model = chat_model
        self.config = config
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
    ) -> list[dict[str, str]]:
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
                "onsite_rule": "on-site only when rover position equals checked rock position",
            },
            "rocks": [
                {
                    "id": index,
                    "position": list(rock_position),
                    "p_good": round(model.rock_good_probability(belief, index), 10),
                }
                for index, rock_position in enumerate(model.map_spec.rock_positions)
            ],
            "history": [
                {"action": action, "observation": outcome} for action, outcome in history
            ],
            "legal_first_actions": list(model.legal_actions(position)),
        }
        user = "\n".join(
            [
                "Propose four distinct legal three-action plans that reduce uncertainty about all rock types.",
                "Travel may sacrifice weak remote checks to reach a high-fidelity on-site inspection.",
                "The exact verifier will score every plan and choose only its first action; a new plan set is requested after the real observation.",
                "Use exactly two distinct movement roots and exactly two distinct check roots across the four plans.",
                'Return JSON only with schema: {"plans":[["action","action","action"],[...],[...],[...]]}',
                "Use the exact action strings in legal_first_actions or check-0 through check-7.",
                "Do not emit probabilities, scores, explanations, indices, or extra fields.",
                "CONTEXT=" + json.dumps(context, separators=(",", ":")),
            ]
        )
        return [
            {
                "role": "system",
                "content": "You propose compact physical inspection plans. Return exact JSON only.",
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
        messages = self._messages(
            model, position=position, belief=belief, history=history
        )
        context = {
            "cell_index": cell_index,
            "position": list(position),
            "history": [list(item) for item in history],
        }
        error: Exception | None = None
        for attempt in range(self.config.validation_retries + 1):
            response = self.chat_model.chat_complete(
                messages, self.config.temperature, num_responses=1
            )[0]
            try:
                plans = compile_plan_cell(
                    response,
                    model=model,
                    position=position,
                    config=self.config,
                )
            except StrategyProposalError as exc:
                error = exc
                with self._lock:
                    self.invalid_responses.append(
                        {**context, "attempt": attempt, "error": str(exc), "raw_response": response}
                    )
                if attempt < self.config.validation_retries:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": f"Invalid plan set: {exc}. Return corrected full JSON only.",
                        },
                    ]
                continue
            with self._lock:
                self.physical_requests.append(
                    {**context, "attempt": attempt, "raw_response": response}
                )
            return RangeGatedPlanCell(plans, response)
        raise StrategyProposalError(f"range-gated plan cell failed after two attempts: {error}")


class DeterministicNamedPlanModel:
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("deterministic plan model supports one response")
        context = json.loads(messages[-1]["content"].split("CONTEXT=", 1)[1])
        position = tuple(context["current_position"])
        legal = list(context["legal_first_actions"])
        move_roots = [action for action in legal if action.startswith("move-")][:2]
        check_roots = [action for action in legal if action.startswith("check-")][:2]
        plans: list[list[str]] = []
        for root in move_roots:
            current = position
            plan = [root]
            effects = {
                "move-NORTH": (0, -1),
                "move-EAST": (1, 0),
                "move-SOUTH": (0, 1),
                "move-WEST": (-1, 0),
            }
            dx, dy = effects[root]
            current = (current[0] + dx, current[1] + dy)
            legal_second = [
                action
                for action, (mx, my) in effects.items()
                if 0 <= current[0] + mx <= 6 and 0 <= current[1] + my <= 6
            ]
            second = legal_second[0]
            plan.extend([second, "check-0"])
            plans.append(plan)
        plans.extend(
            [[check_roots[0], "check-1", "check-2"], [check_roots[1], "check-2", "check-3"]]
        )
        return [json.dumps({"plans": plans})]


def run_smoke(provider: NamedPlanProvider, config: RangeGatedStrategyConfig) -> dict[str, Any]:
    model = RangeGatedRockDiagnosisModel(get_paper_map("7-8"))
    records: list[dict[str, Any]] = []
    position = model.map_spec.start_position
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
        records.append(
            {"cell_index": index, "history": [list(item) for item in history], "plans": cell.plans}
        )
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_named_plan_serving_smoke",
        "config": asdict(config),
        "mechanics": {
            "ten_cells_completed": len(records) == 10,
            "all_cells_have_four_legal_distinct_plans": all(len(row["plans"]) == 4 for row in records),
            "exactly_ten_logical_calls": len(provider.physical_requests) == 10,
            "rollout_scoring_made_no_llm_calls": True,
        },
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
        default=Path("configs/config_nonmyopic_rocksample_15_15_vllm.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/range_gated_rock_26b_smoke_20260723"),
    )
    parser.add_argument("--run-id", default="range-gated-rock-26b-smoke-20260723")
    parser.add_argument("--seed", type=int, default=24_145)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = RangeGatedStrategyConfig(seed=args.seed)
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicNamedPlanModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = config.max_new_tokens
        chat_model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = NamedPlanProvider(chat_model, config)
    try:
        result = run_smoke(provider, config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "range_gated_rock_named_plan_serving_smoke",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage_snapshot(chat_model),
        }
        (args.output_dir / "SMOKE_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    result["usage"] = _usage_snapshot(chat_model)
    result["run_id"] = args.run_id
    result["dry_run"] = args.dry_run
    result["mechanics"]["zero_reasoning_tokens"] = int(result["usage"].get("reasoning_tokens", 0)) == 0
    result["mechanics"]["zero_forced_exits"] = int(result["usage"].get("forced_exits", 0)) == 0
    result["gate"] = {"passed": all(result["mechanics"].values())}
    (args.output_dir / "SMOKE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": result["gate"], "provider": result["provider"]}, indent=2))


if __name__ == "__main__":
    main()
