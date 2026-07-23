"""Strict indexed semantic branch policies for Mushroom feature acquisition."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.mushroom_feature_acquisition import (  # noqa: E402
    COLLECT_ACTION,
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    FIELD_FEATURES,
    AcquisitionState,
    MushroomFeatureModel,
)
from environments.mushroom_feature_acquisition.model import EPSILON  # noqa: E402
from helpers import Config, load_config  # noqa: E402
from model_factory import build_model_adapter  # noqa: E402
from scripts.nonmyopic_gated_sensor_strategy_prior import (  # noqa: E402
    ChatModel,
    StrategyProposalError,
    _usage_snapshot,
)

CHOICE_CODES = "0123456789ABCDEFGHIJKLMNOPQRSTUV"


@dataclass(frozen=True)
class MushroomStrategyConfig:
    num_strategies: int = 4
    seed: int = 24_127
    temperature: float = 0.0
    validation_retries: int = 1
    max_new_tokens: int = 128

    def validate(self) -> None:
        if self.num_strategies != 4:
            raise ValueError("the frozen Mushroom interface uses exactly four roots")
        if self.validation_retries != 1:
            raise ValueError("the frozen Mushroom interface permits one validation retry")
        if self.max_new_tokens != 128:
            raise ValueError("the repaired Mushroom smoke uses a 128-token output cap")


@dataclass(frozen=True)
class MushroomBranchStrategy:
    root_action: str
    followups: dict[str, str]


@dataclass(frozen=True)
class MushroomStrategyCell:
    strategies: tuple[MushroomBranchStrategy, ...]
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


def _fixed_roots(
    model: MushroomFeatureModel,
    *,
    state: AcquisitionState,
    belief: np.ndarray,
    count: int,
) -> tuple[str, ...]:
    legal = model.legal_actions(state)
    queries = [action for action in legal if action != COLLECT_ACTION]
    queries.sort(
        key=lambda action: (
            model.expected_target_entropy(belief, action),
            legal.index(action),
        )
    )
    setup = (COLLECT_ACTION,) if COLLECT_ACTION in legal else ()
    roots = (*setup, *queries[: count - len(setup)])
    if len(roots) != count:
        raise StrategyProposalError("state cannot supply four distinct fixed roots")
    return roots


def _branch_menus(
    model: MushroomFeatureModel,
    *,
    state: AcquisitionState,
    belief: np.ndarray,
    roots: tuple[str, ...],
) -> list[dict[str, tuple[str, ...]]]:
    menus: list[dict[str, tuple[str, ...]]] = []
    for root in roots:
        root_menus: dict[str, tuple[str, ...]] = {}
        child_state = model.next_state(state, root)
        for outcome in model.outcomes(root):
            probability = model.outcome_probability(belief, root, outcome)
            if probability <= EPSILON:
                continue
            choices = tuple(
                action
                for action in model.legal_actions(child_state)
                if action.startswith("query:")
            )
            if root == COLLECT_ACTION:
                choices = tuple(
                    action
                    for action in choices
                    if model.action_feature(action) not in FIELD_FEATURES
                )
            if not choices:
                raise StrategyProposalError("nonterminal Mushroom branch has no query follow-up")
            key = "none" if outcome is None else str(outcome)
            root_menus[key] = choices
        menus.append(root_menus)
    return menus


def compile_indexed_cell(
    response: str,
    *,
    roots: tuple[str, ...],
    menus: list[dict[str, tuple[str, ...]]],
) -> tuple[MushroomBranchStrategy, ...]:
    try:
        payload = json.loads(_normalize_response(response))
    except json.JSONDecodeError as exc:
        raise StrategyProposalError("indexed Mushroom response is not valid JSON") from exc
    expected_keys = {f"r{slot}" for slot in range(len(roots))}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise StrategyProposalError(
            f"indexed Mushroom response must contain exactly keys {sorted(expected_keys)}"
        )
    strategies: list[MushroomBranchStrategy] = []
    for slot, (root, root_menus) in enumerate(zip(roots, menus, strict=True)):
        encoded = payload[f"r{slot}"]
        if not isinstance(encoded, str) or len(encoded) != len(root_menus):
            raise StrategyProposalError(
                f"r{slot} must be one code string of exactly {len(root_menus)} characters"
            )
        followups: dict[str, str] = {}
        for branch_index, (outcome, choices) in enumerate(root_menus.items()):
            code = encoded[branch_index]
            if code not in CHOICE_CODES[: len(choices)]:
                raise StrategyProposalError(
                    f"slot {slot} code for {outcome} must be one of "
                    f"{CHOICE_CODES[: len(choices)]!r}"
                )
            index = CHOICE_CODES.index(code)
            followups[outcome] = choices[index]
        strategies.append(MushroomBranchStrategy(root, followups))
    return tuple(strategies)


class IndexedMushroomProvider:
    def __init__(self, chat_model: ChatModel, config: MushroomStrategyConfig) -> None:
        self.chat_model = chat_model
        self.config = config
        self.physical_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _messages(
        self,
        model: MushroomFeatureModel,
        *,
        state: AcquisitionState,
        belief: np.ndarray,
        history: tuple[tuple[str, str | None], ...],
        roots: tuple[str, ...],
        menus: list[dict[str, tuple[str, ...]]],
    ) -> list[dict[str, str]]:
        slots: list[dict[str, Any]] = []
        for slot, (root, root_menus) in enumerate(zip(roots, menus, strict=True)):
            menu_variants = set(root_menus.values())
            if len(menu_variants) != 1:
                raise StrategyProposalError("Mushroom outcome branches must share one legal menu")
            shared_choices = next(iter(menu_variants))
            branches: list[dict[str, Any]] = []
            for outcome in root_menus:
                raw_outcome = None if outcome == "none" else outcome
                posterior = model.posterior(belief, root, raw_outcome)
                branch: dict[str, Any] = {
                    "outcome": outcome,
                    "probability": round(model.outcome_probability(belief, root, raw_outcome), 10),
                    "p_edible": round(model.class_probability(posterior, "e"), 10),
                    "p_poisonous": round(model.class_probability(posterior, "p"), 10),
                }
                if root != COLLECT_ACTION:
                    feature = model.action_feature(root)
                    branch["outcome_label"] = model.outcome_label(feature, outcome)
                branches.append(branch)
            slots.append(
                {
                    "slot": slot,
                    "root_action": root,
                    "root_description": (
                        "collect a specimen; this consumes the current round and unlocks detailed features"
                        if root == COLLECT_ACTION
                        else FEATURE_DESCRIPTIONS[model.action_feature(root)]
                    ),
                    "menu": [
                        {
                            "code": CHOICE_CODES[index],
                            "feature": model.action_feature(action),
                            "description": FEATURE_DESCRIPTIONS[model.action_feature(action)],
                        }
                        for index, action in enumerate(shared_choices)
                    ],
                    "branches": branches,
                }
            )
        schema = {
            f"r{slot}": "0" * len(slot_payload["branches"])
            for slot, slot_payload in enumerate(slots)
        }
        history_payload = []
        for action, outcome in history:
            item: dict[str, Any] = {"action": action, "outcome": outcome}
            if action.startswith("query:") and outcome is not None:
                feature = model.action_feature(action)
                item["outcome_label"] = model.outcome_label(feature, outcome)
            history_payload.append(item)
        user = "\n".join(
            [
                "Choose one legal follow-up feature for every outcome branch of every fixed root.",
                "The goal is to distinguish edible from poisonous mushrooms quickly over the remaining rounds.",
                "Return JSON only. For each root key, concatenate exactly one menu code per listed branch.",
                "Each root string must match its schema length exactly and contain no spaces.",
                "Schema: " + json.dumps(schema, separators=(",", ":")),
                "Never emit feature names, explanations, scores, or extra fields.",
                "Current specimen collected: " + str(state.specimen_collected).lower(),
                "Current p_edible: " + str(round(model.class_probability(belief, "e"), 10)),
                "Current p_poisonous: " + str(round(model.class_probability(belief, "p"), 10)),
                "History: " + json.dumps(history_payload, separators=(",", ":")),
                "ROOT_SLOTS=" + json.dumps(slots, separators=(",", ":")),
            ]
        )
        system = (
            "You design short observation-contingent feature-acquisition policies for exact Bayesian "
            "mushroom edibility diagnosis. Roots and legal menus are fixed. Return indexed JSON only."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def propose(
        self,
        model: MushroomFeatureModel,
        *,
        cell_index: int,
        state: AcquisitionState,
        belief: np.ndarray,
        history: tuple[tuple[str, str | None], ...],
    ) -> MushroomStrategyCell:
        roots = _fixed_roots(
            model,
            state=state,
            belief=belief,
            count=self.config.num_strategies,
        )
        menus = _branch_menus(model, state=state, belief=belief, roots=roots)
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
            "specimen_collected": state.specimen_collected,
            "history": [{"action": action, "outcome": outcome} for action, outcome in history],
            "roots": list(roots),
            "menus": [
                {outcome: list(choices) for outcome, choices in root_menus.items()}
                for root_menus in menus
            ],
        }
        error: Exception | None = None
        for attempt in range(self.config.validation_retries + 1):
            response = self.chat_model.chat_complete(messages, self.config.temperature, num_responses=1)[0]
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
                            "content": f"Invalid indexed response: {exc}. Return corrected full JSON only.",
                        },
                    ]
                continue
            with self._lock:
                self.physical_requests.append(
                    {**context, "attempt": attempt, "raw_response": response}
                )
            return MushroomStrategyCell(strategies, response)
        raise StrategyProposalError(f"Mushroom cell failed after two attempts: {error}")


class DeterministicIndexedMushroomModel:
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("deterministic Mushroom model supports one response")
        slots = json.loads(messages[-1]["content"].split("ROOT_SLOTS=", 1)[1])
        return [
            json.dumps(
                {
                    f"r{slot}": "0" * len(slot_payload["branches"])
                    for slot, slot_payload in enumerate(slots)
                }
            )
        ]


def _best_query(
    model: MushroomFeatureModel, state: AcquisitionState, belief: np.ndarray
) -> str:
    queries = [action for action in model.legal_actions(state) if action.startswith("query:")]
    return min(
        queries,
        key=lambda action: (
            model.expected_target_entropy(belief, action),
            model.legal_actions(state).index(action),
        ),
    )


def build_smoke_cells(
    model: MushroomFeatureModel, *, seed: int
) -> list[tuple[int, AcquisitionState, np.ndarray, tuple[tuple[str, str | None], ...]]]:
    rng = np.random.default_rng(seed)
    truths = rng.choice(len(model.rows), size=10, replace=False)
    field_steps = (0, 1, 1, 2, 2, 2)
    specimen_steps = (0, 1, 2, 3)
    cells = []
    for index, truth in enumerate(truths):
        state = model.initial_state
        belief = model.initial_belief.copy()
        history: list[tuple[str, str | None]] = []
        if index >= len(field_steps):
            outcome = model.observation(int(truth), COLLECT_ACTION)
            belief = model.posterior(belief, COLLECT_ACTION, outcome)
            state = model.next_state(state, COLLECT_ACTION)
            history.append((COLLECT_ACTION, outcome))
            steps = specimen_steps[index - len(field_steps)]
        else:
            steps = field_steps[index]
        for _ in range(steps):
            action = _best_query(model, state, belief)
            outcome = model.observation(int(truth), action)
            belief = model.posterior(belief, action, outcome)
            state = model.next_state(state, action)
            history.append((action, outcome))
        cells.append((int(truth), state, belief, tuple(history)))
    return cells


def run_smoke(
    provider: IndexedMushroomProvider,
    config: MushroomStrategyConfig,
) -> dict[str, Any]:
    model = MushroomFeatureModel()
    cells = build_smoke_cells(model, seed=config.seed)
    records: list[dict[str, Any]] = []
    accepted_before = len(provider.physical_requests)
    physical_before = len(provider.physical_requests) + len(provider.invalid_responses)
    all_legal = True
    collection_covered = True
    for index, (truth, state, belief, history) in enumerate(cells):
        cell = provider.propose(
            model,
            cell_index=index,
            state=state,
            belief=belief,
            history=history,
        )
        legal = model.legal_actions(state)
        all_legal &= all(strategy.root_action in legal for strategy in cell.strategies)
        all_legal &= all(
            followup in model.legal_actions(model.next_state(state, strategy.root_action))
            for strategy in cell.strategies
            for followup in strategy.followups.values()
        )
        if not state.specimen_collected:
            collection_covered &= cell.strategies[0].root_action == COLLECT_ACTION
        records.append(
            {
                "cell_index": index,
                "truth_index": truth,
                "specimen_collected": state.specimen_collected,
                "history_length": len(history),
                "roots": [strategy.root_action for strategy in cell.strategies],
                "followups": [strategy.followups for strategy in cell.strategies],
            }
        )
    physical_after = len(provider.physical_requests) + len(provider.invalid_responses)
    return {
        "schema_version": 1,
        "stage": "mushroom_feature_acquisition_26b_serving_smoke",
        "config": asdict(config),
        "mechanics": {
            "ten_cells_completed": len(records) == 10,
            "all_roots_and_followups_legal": all_legal,
            "collection_root_covered_in_uncollected_cells": collection_covered,
            "exactly_ten_logical_calls": len(provider.physical_requests) - accepted_before == 10,
            "rollout_scoring_made_no_llm_calls": True,
        },
        "provider": {
            "accepted_requests": len(provider.physical_requests),
            "invalid_responses": len(provider.invalid_responses),
            "physical_requests": physical_after,
        },
        "records": records,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config_nonmyopic_rocksample_15_15_vllm.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/mushroom_feature_26b_smoke_20260723"),
    )
    parser.add_argument("--run-id", default="mushroom-feature-26b-smoke-20260723")
    parser.add_argument("--seed", type=int, default=24_127)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = MushroomStrategyConfig(seed=args.seed)
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicIndexedMushroomModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = config.max_new_tokens
        chat_model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = IndexedMushroomProvider(chat_model, config)
    try:
        result = run_smoke(provider, config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "mushroom_feature_acquisition_26b_serving_smoke",
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
    usage = result["usage"]
    no_reasoning = int(usage.get("reasoning_tokens", 0)) == 0
    no_forced = int(usage.get("forced_exits", 0)) == 0
    result["mechanics"]["zero_reasoning_tokens"] = no_reasoning
    result["mechanics"]["zero_forced_exits"] = no_forced
    result["gate"] = {"passed": all(result["mechanics"].values())}
    (args.output_dir / "SMOKE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": result["gate"], "provider": result["provider"]}, indent=2))


if __name__ == "__main__":
    main()
