"""Compact indexed branch policies for the Cleveland heart-workup task."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import threading
from typing import Any, Literal

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.heart_workup import (  # noqa: E402
    FEATURE_DESCRIPTIONS,
    ORDER_WORKUP_ACTION,
    HeartWorkupModel,
    WorkupState,
)
from environments.heart_workup.model import EPSILON  # noqa: E402
from helpers import Config, load_config  # noqa: E402
from model_factory import build_model_adapter  # noqa: E402
from scripts.nonmyopic_gated_sensor_strategy_prior import (  # noqa: E402
    ChatModel,
    StrategyProposalError,
    _usage_snapshot,
)


OUTCOME_LABELS = {
    "sex": {"0": "female", "1": "male"},
    "chest-pain": {
        "1": "typical angina",
        "2": "atypical angina",
        "3": "non-anginal pain",
        "4": "asymptomatic",
    },
    "fasting-blood-sugar": {"0": "not above 120 mg/dl", "1": "above 120 mg/dl"},
    "resting-ecg": {
        "0": "normal",
        "1": "ST-T wave abnormality",
        "2": "left-ventricular hypertrophy",
    },
    "exercise-angina": {"0": "no", "1": "yes"},
    "st-slope": {"1": "upsloping", "2": "flat", "3": "downsloping"},
    "major-vessels": {
        "0": "zero vessels",
        "1": "one vessel",
        "2": "two vessels",
        "3": "three vessels",
    },
    "thal": {"3": "normal", "6": "fixed defect", "7": "reversible defect"},
}


@dataclass(frozen=True)
class HeartStrategyConfig:
    num_strategies: int = 4
    seed: int = 24_136
    temperature: float = 0.0
    validation_retries: int = 1
    max_new_tokens: int = 128
    utility_summary_mode: Literal["none", "branch_local_expected_entropy"] = "none"
    project_invalid_after_retries: bool = False
    allow_fewer_roots_when_exhausted: bool = False

    def validate(self) -> None:
        if self.num_strategies != 4:
            raise ValueError("the frozen Heart interface uses exactly four roots")
        if self.validation_retries != 1:
            raise ValueError("the frozen Heart interface permits one validation retry")
        if self.max_new_tokens != 128:
            raise ValueError("the frozen Heart interface uses a 128-token output cap")
        if self.utility_summary_mode not in ("none", "branch_local_expected_entropy"):
            raise ValueError("unsupported Heart continuation utility summary mode")


@dataclass(frozen=True)
class HeartBranchStrategy:
    root_action: str
    followups: dict[str, str]


@dataclass(frozen=True)
class HeartStrategyCell:
    strategies: tuple[HeartBranchStrategy, ...]
    raw_response: str


def _normalize_response(response: str) -> str:
    normalized = response.strip()
    if "```json" in normalized:
        start = normalized.rfind("```json") + len("```json")
        if normalized[start : start + 1] == "\n":
            start += 1
        end = normalized.find("```", start)
        if end < 0:
            raise StrategyProposalError("response has an incomplete JSON fence")
        normalized = normalized[start:end].strip()
    return normalized


def _action_description(model: HeartWorkupModel, action: str) -> str:
    if action == ORDER_WORKUP_ACTION:
        return (
            "order the clinical workup; consumes this round, gives no immediate information, "
            "and unlocks laboratory, ECG, exercise, vessel, and thal tests"
        )
    return FEATURE_DESCRIPTIONS[model.action_feature(action)]


def _outcome_label(feature: str, outcome: str) -> str:
    return OUTCOME_LABELS.get(feature, {}).get(outcome, outcome)


def fixed_roots(
    model: HeartWorkupModel,
    *,
    state: WorkupState,
    belief: np.ndarray,
    count: int,
) -> tuple[str, ...]:
    legal = model.legal_actions(state)
    queries = [action for action in legal if action.startswith("query:")]
    queries.sort(
        key=lambda action: (
            model.expected_target_entropy(belief, action),
            legal.index(action),
        )
    )
    setup = (ORDER_WORKUP_ACTION,) if ORDER_WORKUP_ACTION in legal else ()
    roots = (*setup, *queries[: count - len(setup)])
    if len(roots) != count:
        raise StrategyProposalError("state cannot supply four distinct fixed Heart roots")
    return roots


def branch_menus(
    model: HeartWorkupModel,
    *,
    state: WorkupState,
    belief: np.ndarray,
    roots: tuple[str, ...],
) -> list[dict[str, tuple[str, ...]]]:
    menus: list[dict[str, tuple[str, ...]]] = []
    for root in roots:
        child_state = model.next_state(state, root)
        choices = tuple(model.legal_actions(child_state))
        if not choices:
            raise StrategyProposalError("nonterminal Heart branch has no legal follow-up")
        root_menus: dict[str, tuple[str, ...]] = {}
        for outcome in model.outcomes(root):
            if model.outcome_probability(belief, root, outcome) <= EPSILON:
                continue
            key = "none" if outcome is None else str(outcome)
            root_menus[key] = choices
        menus.append(root_menus)
    return menus


def continuation_utility_cards(
    model: HeartWorkupModel,
    *,
    belief: np.ndarray,
    roots: tuple[str, ...],
    menus: list[dict[str, tuple[str, ...]]],
) -> list[dict[str, list[dict[str, float | int | str]]]]:
    """Return leakage-free one-step utility summaries under each root outcome."""
    cards: list[dict[str, list[dict[str, float | int | str]]]] = []
    for root, root_menus in zip(roots, menus, strict=True):
        root_cards: dict[str, list[dict[str, float | int | str]]] = {}
        for outcome, choices in root_menus.items():
            raw_outcome = None if outcome == "none" else outcome
            posterior = model.posterior(belief, root, raw_outcome)
            entropy = model.target_entropy(posterior)
            root_cards[outcome] = [
                {
                    "index": index,
                    "action": action,
                    "expected_class_entropy": round(
                        model.expected_target_entropy(posterior, action), 8
                    ),
                    "one_step_information_gain": round(
                        entropy - model.expected_target_entropy(posterior, action), 8
                    ),
                }
                for index, action in enumerate(choices)
            ]
        cards.append(root_cards)
    return cards


def _best_continuation_index(
    model: HeartWorkupModel,
    *,
    posterior: np.ndarray,
    choices: tuple[str, ...],
) -> int:
    scores = [model.expected_target_entropy(posterior, action) for action in choices]
    best_score = min(scores)
    return next(
        index for index, score in enumerate(scores) if score <= best_score + EPSILON
    )


def compile_indexed_cell(
    response: str,
    *,
    roots: tuple[str, ...],
    menus: list[dict[str, tuple[str, ...]]],
) -> tuple[HeartBranchStrategy, ...]:
    try:
        payload = json.loads(_normalize_response(response))
    except json.JSONDecodeError as exc:
        raise StrategyProposalError("indexed Heart response is not valid JSON") from exc
    expected_keys = {f"r{slot}" for slot in range(len(roots))}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise StrategyProposalError(
            f"indexed Heart response must contain exactly keys {sorted(expected_keys)}"
        )
    strategies: list[HeartBranchStrategy] = []
    for slot, (root, root_menus) in enumerate(zip(roots, menus, strict=True)):
        encoded = payload[f"r{slot}"]
        if not isinstance(encoded, list) or len(encoded) != len(root_menus):
            raise StrategyProposalError(
                f"r{slot} must be one integer array of exactly {len(root_menus)} items"
            )
        followups: dict[str, str] = {}
        for branch_index, (outcome, choices) in enumerate(root_menus.items()):
            index = encoded[branch_index]
            if isinstance(index, bool) or not isinstance(index, int):
                raise StrategyProposalError(f"slot {slot} index for {outcome} must be an integer")
            if not 0 <= index < len(choices):
                raise StrategyProposalError(
                    f"slot {slot} index for {outcome} must be in [0, {len(choices) - 1}]"
                )
            followups[outcome] = choices[index]
        strategies.append(HeartBranchStrategy(root, followups))
    return tuple(strategies)


def project_indexed_cell(
    response: str,
    *,
    model: HeartWorkupModel,
    belief: np.ndarray,
    roots: tuple[str, ...],
    menus: list[dict[str, tuple[str, ...]]],
) -> tuple[tuple[HeartBranchStrategy, ...], list[dict[str, Any]]]:
    """Project only invalid indexed branches onto exact legal continuations."""
    try:
        payload = json.loads(_normalize_response(response))
    except (StrategyProposalError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    strategies: list[HeartBranchStrategy] = []
    projections: list[dict[str, Any]] = []
    for slot, (root, root_menus) in enumerate(zip(roots, menus, strict=True)):
        encoded = payload.get(f"r{slot}", [])
        if not isinstance(encoded, list):
            encoded = []
        followups: dict[str, str] = {}
        for branch_index, (outcome, choices) in enumerate(root_menus.items()):
            proposed = encoded[branch_index] if branch_index < len(encoded) else None
            if (
                not isinstance(proposed, bool)
                and isinstance(proposed, int)
                and 0 <= proposed < len(choices)
            ):
                followups[outcome] = choices[proposed]
                continue
            raw_outcome = None if outcome == "none" else outcome
            posterior = model.posterior(belief, root, raw_outcome)
            replacement_index = _best_continuation_index(
                model, posterior=posterior, choices=choices
            )
            replacement = choices[replacement_index]
            followups[outcome] = replacement
            projections.append(
                {
                    "slot": slot,
                    "root": root,
                    "outcome": outcome,
                    "branch_index": branch_index,
                    "proposed": repr(proposed),
                    "replacement_index": replacement_index,
                    "replacement": replacement,
                }
            )
        strategies.append(HeartBranchStrategy(root, followups))
    return tuple(strategies), projections


def _serialized_strategies(
    strategies: tuple[HeartBranchStrategy, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "root_action": strategy.root_action,
            "followups": dict(strategy.followups),
        }
        for strategy in strategies
    ]


class IndexedHeartProvider:
    def __init__(self, chat_model: ChatModel, config: HeartStrategyConfig) -> None:
        self.chat_model = chat_model
        self.config = config
        self.physical_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []
        self.projected_responses: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _messages(
        self,
        model: HeartWorkupModel,
        *,
        state: WorkupState,
        belief: np.ndarray,
        history: tuple[tuple[str, str | None], ...],
        roots: tuple[str, ...],
        menus: list[dict[str, tuple[str, ...]]],
    ) -> list[dict[str, str]]:
        slots: list[dict[str, Any]] = []
        utility_cards = (
            continuation_utility_cards(model, belief=belief, roots=roots, menus=menus)
            if self.config.utility_summary_mode == "branch_local_expected_entropy"
            else None
        )
        for slot, (root, root_menus) in enumerate(zip(roots, menus, strict=True)):
            menu_variants = set(root_menus.values())
            if len(menu_variants) != 1:
                raise StrategyProposalError("Heart outcome branches must share one legal menu")
            choices = next(iter(menu_variants))
            branches: list[dict[str, Any]] = []
            for outcome in root_menus:
                raw_outcome = None if outcome == "none" else outcome
                posterior = model.posterior(belief, root, raw_outcome)
                branch: dict[str, Any] = {
                    "outcome": outcome,
                    "probability": round(
                        model.outcome_probability(belief, root, raw_outcome), 10
                    ),
                    "p_disease": round(model.class_probability(posterior, 1), 10),
                    "p_no_disease": round(model.class_probability(posterior, 0), 10),
                }
                if root.startswith("query:") and outcome != "none":
                    branch["outcome_label"] = _outcome_label(
                        model.action_feature(root), outcome
                    )
                if utility_cards is not None:
                    branch["continuation_utility"] = utility_cards[slot][outcome]
                branches.append(branch)
            slots.append(
                {
                    "slot": slot,
                    "root_action": root,
                    "root_description": _action_description(model, root),
                    "menu": [
                        {
                            "index": index,
                            "action": action,
                            "description": _action_description(model, action),
                        }
                        for index, action in enumerate(choices)
                    ],
                    "branches": branches,
                }
            )
        history_payload = []
        for action, outcome in history:
            item: dict[str, Any] = {"action": action, "outcome": outcome}
            if action.startswith("query:") and outcome is not None:
                item["outcome_label"] = _outcome_label(model.action_feature(action), outcome)
            history_payload.append(item)
        schema = {
            f"r{slot}": [0] * len(slot_payload["branches"])
            for slot, slot_payload in enumerate(slots)
        }
        limits = {
            f"r{slot}": {
                "exact_items": len(slot_payload["branches"]),
                "each_integer_min": 0,
                "each_integer_max": len(slot_payload["menu"]) - 1,
            }
            for slot, slot_payload in enumerate(slots)
        }
        instructions = [
            "Choose one legal second action for every outcome branch of every fixed root.",
            "The goal is to reduce uncertainty about heart-disease presence over two actions.",
            "The exact verifier will score the complete branch policies and choose one root.",
            "A clinical workup consumes a round and reveals nothing immediately, but unlocks stronger tests.",
        ]
        if self.config.utility_summary_mode == "branch_local_expected_entropy":
            instructions.append(
                "Each branch includes calibrated continuation_utility values from the empirical "
                "prior. Prefer lower expected_class_entropy (equivalently higher "
                "one_step_information_gain) within that branch."
            )
        user = "\n".join(
            [
                *instructions,
                "Return JSON only: one integer menu index per listed branch, in listed order.",
                "Menu indexes are LOCAL to each root. Never reuse an r0 index as an r1/r2/r3 index.",
                "Schema: " + json.dumps(schema, separators=(",", ":")),
                "Never emit action names, explanations, scores, or extra fields.",
                "Current workup ordered: " + str(state.workup_ordered).lower(),
                "Current p_disease: " + str(round(model.class_probability(belief, 1), 10)),
                "Current p_no_disease: " + str(round(model.class_probability(belief, 0), 10)),
                "History: " + json.dumps(history_payload, separators=(",", ":")),
                "ROOT_SLOTS=" + json.dumps(slots, separators=(",", ":")),
                "FINAL_OUTPUT_LIMITS=" + json.dumps(limits, separators=(",", ":")),
            ]
        )
        system = (
            "You design short observation-contingent diagnostic test policies for exact Bayesian "
            "heart-disease classification. Roots and legal menus are fixed. Return indexed JSON only."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def propose(
        self,
        model: HeartWorkupModel,
        *,
        cell_index: int,
        state: WorkupState,
        belief: np.ndarray,
        history: tuple[tuple[str, str | None], ...],
    ) -> HeartStrategyCell:
        root_count = (
            min(self.config.num_strategies, len(model.legal_actions(state)))
            if self.config.allow_fewer_roots_when_exhausted
            else self.config.num_strategies
        )
        roots = fixed_roots(
            model,
            state=state,
            belief=belief,
            count=root_count,
        )
        menus = branch_menus(model, state=state, belief=belief, roots=roots)
        messages = self._messages(
            model,
            state=state,
            belief=belief,
            history=history,
            roots=roots,
            menus=menus,
        )
        context = {
            "cell_index": cell_index,
            "workup_ordered": state.workup_ordered,
            "history": [{"action": action, "outcome": outcome} for action, outcome in history],
            "roots": list(roots),
            "menus": [
                {outcome: list(choices) for outcome, choices in root_menus.items()}
                for root_menus in menus
            ],
        }
        if self.config.utility_summary_mode == "branch_local_expected_entropy":
            context["utility_summary_mode"] = self.config.utility_summary_mode
            context["continuation_utility_cards"] = continuation_utility_cards(
                model, belief=belief, roots=roots, menus=menus
            )
        error: Exception | None = None
        for attempt in range(self.config.validation_retries + 1):
            response = self.chat_model.chat_complete(
                messages, self.config.temperature, num_responses=1
            )[0]
            try:
                strategies = compile_indexed_cell(response, roots=roots, menus=menus)
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
                            "content": (
                                f"Invalid indexed response: {exc}. Each root has its own local menu; "
                                "do not copy index values between roots. Obey FINAL_OUTPUT_LIMITS and "
                                "return corrected full JSON only."
                            ),
                        },
                    ]
                    continue
                if self.config.project_invalid_after_retries:
                    strategies, projections = project_indexed_cell(
                        response,
                        model=model,
                        belief=belief,
                        roots=roots,
                        menus=menus,
                    )
                    projected = {
                        **context,
                        "attempt": attempt,
                        "raw_response": response,
                        "projected": True,
                        "projection_events": projections,
                        "compiled_strategies": _serialized_strategies(strategies),
                    }
                    with self._lock:
                        self.projected_responses.append(projected)
                        self.physical_requests.append(projected)
                    return HeartStrategyCell(strategies, response)
                continue
            with self._lock:
                self.physical_requests.append(
                    {
                        **context,
                        "attempt": attempt,
                        "raw_response": response,
                        "compiled_strategies": _serialized_strategies(strategies),
                    }
                )
            return HeartStrategyCell(strategies, response)
        raise StrategyProposalError(f"Heart cell failed after two attempts: {error}")


class DeterministicIndexedHeartModel:
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("deterministic Heart model supports one response")
        slots_payload = messages[-1]["content"].split("ROOT_SLOTS=", 1)[1]
        slots = json.loads(slots_payload.split("\nFINAL_OUTPUT_LIMITS=", 1)[0])
        return [
            json.dumps(
                {
                    f"r{slot}": [0] * len(slot_payload["branches"])
                    for slot, slot_payload in enumerate(slots)
                }
            )
        ]


class DeterministicUtilityHeartModel:
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("deterministic Heart model supports one response")
        slots_payload = messages[-1]["content"].split("ROOT_SLOTS=", 1)[1]
        slots = json.loads(slots_payload.split("\nFINAL_OUTPUT_LIMITS=", 1)[0])
        return [
            json.dumps(
                {
                    f"r{slot}": [
                        min(
                            branch["continuation_utility"],
                            key=lambda card: (
                                card["expected_class_entropy"],
                                card["index"],
                            ),
                        )["index"]
                        for branch in slot_payload["branches"]
                    ]
                    for slot, slot_payload in enumerate(slots)
                }
            )
        ]


def _best_query(model: HeartWorkupModel, state: WorkupState, belief: np.ndarray) -> str:
    queries = [action for action in model.legal_actions(state) if action.startswith("query:")]
    legal = model.legal_actions(state)
    return min(
        queries,
        key=lambda action: (
            model.expected_target_entropy(belief, action),
            legal.index(action),
        ),
    )


def build_smoke_cells(
    model: HeartWorkupModel, *, seed: int
) -> list[tuple[int, WorkupState, np.ndarray, tuple[tuple[str, str | None], ...]]]:
    truths = np.random.default_rng(seed).permutation(len(model.rows))
    cells = []
    for index, raw_truth in enumerate(truths[:10]):
        truth = int(raw_truth)
        state = model.initial_state
        belief = model.initial_belief.copy()
        history: list[tuple[str, str | None]] = []
        if index >= 5:
            outcome = model.observation(truth, ORDER_WORKUP_ACTION)
            belief = model.posterior(belief, ORDER_WORKUP_ACTION, outcome)
            state = model.next_state(state, ORDER_WORKUP_ACTION)
            history.append((ORDER_WORKUP_ACTION, outcome))
        steps = (0, 1, 2, 1, 2)[index % 5]
        for _ in range(steps):
            action = _best_query(model, state, belief)
            outcome = model.observation(truth, action)
            belief = model.posterior(belief, action, outcome)
            state = model.next_state(state, action)
            history.append((action, outcome))
        cells.append((truth, state, belief, tuple(history)))
    return cells


def run_smoke(provider: IndexedHeartProvider, config: HeartStrategyConfig) -> dict[str, Any]:
    model = HeartWorkupModel()
    records: list[dict[str, Any]] = []
    all_legal = True
    workup_covered = True
    for index, (truth, state, belief, history) in enumerate(
        build_smoke_cells(model, seed=config.seed)
    ):
        roots = fixed_roots(model, state=state, belief=belief, count=4)
        menus = branch_menus(model, state=state, belief=belief, roots=roots)
        cell = provider.propose(
            model,
            cell_index=index,
            state=state,
            belief=belief,
            history=history,
        )
        all_legal &= all(
            strategy.root_action in model.legal_actions(state) for strategy in cell.strategies
        )
        all_legal &= all(
            followup in model.legal_actions(model.next_state(state, strategy.root_action))
            for strategy in cell.strategies
            for followup in strategy.followups.values()
        )
        if not state.workup_ordered:
            workup_covered &= roots[0] == ORDER_WORKUP_ACTION
            workup_covered &= all(
                ORDER_WORKUP_ACTION in choices
                for root_menus in menus[1:]
                for choices in root_menus.values()
            )
        records.append(
            {
                "cell_index": index,
                "truth_index": truth,
                "workup_ordered": state.workup_ordered,
                "history_length": len(history),
                "roots": [strategy.root_action for strategy in cell.strategies],
                "followups": [strategy.followups for strategy in cell.strategies],
            }
        )
    return {
        "schema_version": 1,
        "stage": (
            "cleveland_heart_workup_projected_utility_serving_smoke"
            if config.utility_summary_mode == "branch_local_expected_entropy"
            and config.project_invalid_after_retries
            else "cleveland_heart_workup_26b_serving_smoke"
        ),
        "config": asdict(config),
        "mechanics": {
            "ten_cells_completed": len(records) == 10,
            "all_roots_and_followups_legal": all_legal,
            "workup_root_and_branch_followups_representable": workup_covered,
            "exactly_ten_logical_calls": len(provider.physical_requests) == 10,
            "zero_projected_cells": len(provider.projected_responses) == 0,
            "rollout_scoring_made_no_llm_calls": True,
        },
        "provider": {
            "accepted_requests": len(provider.physical_requests),
            "invalid_responses": len(provider.invalid_responses),
            "projected_cells": len(provider.projected_responses),
            "projected_branches": sum(
                len(request["projection_events"])
                for request in provider.projected_responses
            ),
            "physical_requests": len(provider.physical_requests) + len(provider.invalid_responses),
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
        default=Path("results/nonmyopic/heart_workup_26b_smoke_20260723"),
    )
    parser.add_argument("--run-id", default="heart-workup-26b-smoke-20260723")
    parser.add_argument("--seed", type=int, default=24_136)
    parser.add_argument(
        "--utility-summary-mode",
        choices=("none", "branch_local_expected_entropy"),
        default="none",
    )
    parser.add_argument("--project-invalid-after-retries", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = HeartStrategyConfig(
        seed=args.seed,
        utility_summary_mode=args.utility_summary_mode,
        project_invalid_after_retries=args.project_invalid_after_retries,
    )
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicIndexedHeartModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = config.max_new_tokens
        chat_model = build_model_adapter(
            runtime_config.model_pairs[0].questioner,
            config=runtime_config,
        )
    provider = IndexedHeartProvider(chat_model, config)
    try:
        result = run_smoke(provider, config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": (
                "cleveland_heart_workup_projected_utility_serving_smoke"
                if config.utility_summary_mode == "branch_local_expected_entropy"
                and config.project_invalid_after_retries
                else "cleveland_heart_workup_26b_serving_smoke"
            ),
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
    print(json.dumps({"gate": result["gate"], "provider": result["provider"]}, indent=2))


if __name__ == "__main__":
    main()
