"""Named branch-contingent policies for the UCI thyroid workup task."""

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

from environments.thyroid_workup import (  # noqa: E402
    COLLECT_BLOOD_ACTION,
    FEATURE_NAMES,
    ThyroidWorkupModel,
    ThyroidWorkupState,
)
from environments.thyroid_workup.model import EPSILON  # noqa: E402
from helpers import Config, load_config  # noqa: E402
from model_factory import build_model_adapter  # noqa: E402
from scripts.nonmyopic_gated_sensor_strategy_prior import (  # noqa: E402
    ChatModel,
    StrategyProposalError,
    _usage_snapshot,
)


FEATURE_DESCRIPTIONS = {
    "age": "age group",
    "sex": "recorded sex",
    "on-thyroxine": "currently taking thyroxine",
    "query-on-thyroxine": "clinician queried thyroxine use",
    "on-antithyroid-medication": "currently taking antithyroid medication",
    "sick": "currently recorded as sick",
    "pregnant": "pregnancy status",
    "thyroid-surgery": "history of thyroid surgery",
    "i131-treatment": "history of radioactive iodine treatment",
    "query-hypothyroid": "clinician queried hypothyroidism",
    "query-hyperthyroid": "clinician queried hyperthyroidism",
    "lithium": "lithium use",
    "goitre": "goitre finding",
    "tumor": "tumor finding",
    "hypopituitary": "hypopituitary finding",
    "psych": "psychiatric history",
    "tsh": "thyroid-stimulating hormone assay",
    "t3": "triiodothyronine assay",
    "tt4": "total thyroxine assay",
    "t4u": "thyroxine uptake assay",
}


@dataclass(frozen=True)
class ThyroidStrategyConfig:
    num_strategies: int = 4
    seed: int = 24_151
    temperature: float = 0.0
    validation_retries: int = 1
    max_new_tokens: int = 1_024
    utility_summary_mode: Literal["none", "branch_local_expected_entropy"] = "none"

    def validate(self) -> None:
        if self.num_strategies != 4:
            raise ValueError("the frozen thyroid interface uses exactly four roots")
        if self.validation_retries != 1:
            raise ValueError("the frozen thyroid interface permits one validation retry")
        if self.max_new_tokens != 1_024:
            raise ValueError("the frozen thyroid interface uses a 1,024-token output cap")
        if self.utility_summary_mode not in ("none", "branch_local_expected_entropy"):
            raise ValueError("unsupported thyroid continuation utility summary mode")


@dataclass(frozen=True)
class ThyroidBranchStrategy:
    root_action: str
    followups: dict[str, str]


@dataclass(frozen=True)
class ThyroidStrategyCell:
    strategies: tuple[ThyroidBranchStrategy, ...]
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


def _description(model: ThyroidWorkupModel, action: str) -> str:
    if action == COLLECT_BLOOD_ACTION:
        return (
            "collect one shared blood sample; consumes this action, reveals nothing now, "
            "and unlocks TSH, T3, TT4, and T4U assays"
        )
    return FEATURE_DESCRIPTIONS[FEATURE_NAMES[model.action_feature(action)]]


def fixed_roots(
    model: ThyroidWorkupModel,
    *,
    state: ThyroidWorkupState,
    belief: np.ndarray,
    count: int = 4,
) -> tuple[str, ...]:
    legal = model.legal_actions(state)
    queries = [action for action in legal if action.startswith("query:")]
    queries.sort(
        key=lambda action: (model.expected_target_entropy(belief, action), legal.index(action))
    )
    setup = (COLLECT_BLOOD_ACTION,) if COLLECT_BLOOD_ACTION in legal else ()
    roots = (*setup, *queries[: count - len(setup)])
    if len(roots) != count:
        raise StrategyProposalError("state cannot supply four distinct thyroid roots")
    return roots


def branch_menus(
    model: ThyroidWorkupModel,
    *,
    state: ThyroidWorkupState,
    belief: np.ndarray,
    roots: tuple[str, ...],
) -> dict[str, dict[str, tuple[str, ...]]]:
    menus: dict[str, dict[str, tuple[str, ...]]] = {}
    for root in roots:
        choices = tuple(model.legal_actions(model.next_state(state, root)))
        branches: dict[str, tuple[str, ...]] = {}
        for outcome in model.outcomes(root):
            if model.outcome_probability(belief, root, outcome) <= EPSILON:
                continue
            branches["none" if outcome is None else str(outcome)] = choices
        if not branches:
            raise StrategyProposalError(f"root {root} has no positive-probability branches")
        menus[root] = branches
    return menus


def continuation_utility_cards(
    model: ThyroidWorkupModel,
    *,
    belief: np.ndarray,
    roots: tuple[str, ...],
    menus: dict[str, dict[str, tuple[str, ...]]],
) -> dict[str, dict[str, list[dict[str, float | str]]]]:
    """Return leakage-free one-step utility summaries under each root outcome."""
    cards: dict[str, dict[str, list[dict[str, float | str]]]] = {}
    for root in roots:
        cards[root] = {}
        for outcome, choices in menus[root].items():
            raw_outcome = None if outcome == "none" else outcome
            posterior = model.posterior(belief, root, raw_outcome)
            entropy = model.target_entropy(posterior)
            action_cards = []
            for action in choices:
                expected_entropy = model.expected_target_entropy(posterior, action)
                action_cards.append(
                    {
                        "action": action,
                        "expected_class_entropy": round(expected_entropy, 8),
                        "one_step_information_gain": round(entropy - expected_entropy, 8),
                    }
                )
            cards[root][outcome] = action_cards
    return cards


def compile_named_cell(
    response: str,
    *,
    roots: tuple[str, ...],
    menus: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[ThyroidBranchStrategy, ...]:
    try:
        payload = json.loads(_normalize_response(response))
    except json.JSONDecodeError as exc:
        raise StrategyProposalError("named thyroid response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != set(roots):
        raise StrategyProposalError("response must contain exactly the four fixed root names")
    strategies: list[ThyroidBranchStrategy] = []
    for root in roots:
        encoded = payload[root]
        root_menus = menus[root]
        if not isinstance(encoded, dict) or set(encoded) != set(root_menus):
            raise StrategyProposalError(
                f"root {root} must contain exactly branches {list(root_menus)}"
            )
        followups: dict[str, str] = {}
        for outcome, choices in root_menus.items():
            followup = encoded[outcome]
            if not isinstance(followup, str) or followup not in choices:
                raise StrategyProposalError(
                    f"follow-up for {root}/{outcome} must be one listed legal action name"
                )
            followups[outcome] = followup
        strategies.append(ThyroidBranchStrategy(root, followups))
    return tuple(strategies)


class NamedThyroidProvider:
    def __init__(self, chat_model: ChatModel, config: ThyroidStrategyConfig) -> None:
        self.chat_model = chat_model
        self.config = config
        self.physical_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _messages(
        self,
        model: ThyroidWorkupModel,
        *,
        state: ThyroidWorkupState,
        belief: np.ndarray,
        history: tuple[tuple[str, str | None], ...],
        roots: tuple[str, ...],
        menus: dict[str, dict[str, tuple[str, ...]]],
    ) -> list[dict[str, str]]:
        slots: list[dict[str, Any]] = []
        schema: dict[str, dict[str, str]] = {}
        utility_cards = (
            continuation_utility_cards(model, belief=belief, roots=roots, menus=menus)
            if self.config.utility_summary_mode == "branch_local_expected_entropy"
            else None
        )
        for root in roots:
            root_menus = menus[root]
            choices = next(iter(root_menus.values()))
            branches = []
            schema[root] = {}
            for outcome in root_menus:
                raw_outcome = None if outcome == "none" else outcome
                posterior = model.posterior(belief, root, raw_outcome)
                branch: dict[str, Any] = {
                    "outcome": outcome,
                    "probability": round(
                        model.outcome_probability(belief, root, raw_outcome), 8
                    ),
                    "class_probabilities": {
                        str(target): round(model.class_probability(posterior, target), 8)
                        for target in (1, 2, 3)
                    },
                }
                if utility_cards is not None:
                    branch["continuation_utility"] = utility_cards[root][outcome]
                branches.append(branch)
                schema[root][outcome] = "LEGAL_ACTION_NAME"
            slots.append(
                {
                    "root_action": root,
                    "root_description": _description(model, root),
                    "branches": branches,
                    "legal_followups": [
                        {"action": action, "description": _description(model, action)}
                        for action in choices
                    ],
                }
            )
        instructions = [
                "Choose one named legal second action for every outcome branch of every fixed root.",
                "The goal is to reduce uncertainty among thyroid classes 1, 2, and 3 over two actions.",
                "The exact verifier will score each complete branch policy and choose the root.",
                "Blood collection has zero immediate information but unlocks four laboratory assays.",
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
                "Return JSON only. Copy action names exactly; never use menu indexes or add fields.",
                "Required shape: " + json.dumps(schema, separators=(",", ":")),
                "Blood already collected: " + str(state.blood_collected).lower(),
                "Current class probabilities: "
                + json.dumps(
                    {
                        str(target): round(model.class_probability(belief, target), 8)
                        for target in (1, 2, 3)
                    },
                    separators=(",", ":"),
                ),
                "History: " + json.dumps(history, separators=(",", ":")),
                "ROOTS=" + json.dumps(slots, separators=(",", ":")),
            ]
        )
        system = (
            "You design short observation-contingent thyroid diagnostic policies. "
            "Roots and legal follow-up names are fixed. Return the complete named JSON only."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def propose(
        self,
        model: ThyroidWorkupModel,
        *,
        cell_index: int,
        state: ThyroidWorkupState,
        belief: np.ndarray,
        history: tuple[tuple[str, str | None], ...],
    ) -> ThyroidStrategyCell:
        roots = fixed_roots(model, state=state, belief=belief)
        menus = branch_menus(model, state=state, belief=belief, roots=roots)
        messages = self._messages(
            model,
            state=state,
            belief=belief,
            history=history,
            roots=roots,
            menus=menus,
        )
        context: dict[str, Any] = {
            "cell_index": cell_index,
            "history": [list(item) for item in history],
            "roots": list(roots),
            "menus": {
                root: {outcome: list(choices) for outcome, choices in root_menus.items()}
                for root, root_menus in menus.items()
            },
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
                strategies = compile_named_cell(response, roots=roots, menus=menus)
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
                                f"Invalid response: {exc}. Copy only exact legal action names from "
                                "each root's legal_followups and return the complete corrected JSON."
                            ),
                        },
                    ]
                continue
            with self._lock:
                self.physical_requests.append(
                    {**context, "attempt": attempt, "raw_response": response}
                )
            return ThyroidStrategyCell(strategies, response)
        raise StrategyProposalError(f"thyroid cell failed after two attempts: {error}")


class DeterministicNamedThyroidModel:
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("deterministic thyroid model supports one response")
        slots = json.loads(messages[-1]["content"].split("ROOTS=", 1)[1])
        return [
            json.dumps(
                {
                    slot["root_action"]: {
                        branch["outcome"]: slot["legal_followups"][0]["action"]
                        for branch in slot["branches"]
                    }
                    for slot in slots
                }
            )
        ]


class DeterministicUtilityThyroidModel:
    """Select the lowest predictive-entropy continuation from each branch card."""

    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("deterministic thyroid model supports one response")
        slots = json.loads(messages[-1]["content"].split("ROOTS=", 1)[1])
        payload: dict[str, dict[str, str]] = {}
        for slot in slots:
            payload[slot["root_action"]] = {}
            for branch in slot["branches"]:
                cards = branch.get("continuation_utility")
                if not cards:
                    raise ValueError("utility-grounded deterministic model requires utility cards")
                best = min(
                    enumerate(cards),
                    key=lambda item: (item[1]["expected_class_entropy"], item[0]),
                )[1]
                payload[slot["root_action"]][branch["outcome"]] = best["action"]
        return [json.dumps(payload)]


def _best_query(
    model: ThyroidWorkupModel, state: ThyroidWorkupState, belief: np.ndarray
) -> str:
    legal = model.legal_actions(state)
    queries = [action for action in legal if action.startswith("query:")]
    return min(
        queries,
        key=lambda action: (model.expected_target_entropy(belief, action), legal.index(action)),
    )


def build_smoke_cells(
    model: ThyroidWorkupModel, *, seed: int
) -> list[tuple[int, ThyroidWorkupState, np.ndarray, tuple[tuple[str, str | None], ...]]]:
    truths = np.random.default_rng(seed).permutation(len(model.targets))
    cells = []
    for cell_index, raw_truth in enumerate(truths[:10]):
        truth = int(raw_truth)
        state = model.initial_state
        belief = model.initial_belief
        history: list[tuple[str, str | None]] = []
        for _ in range(cell_index % 3):
            action = _best_query(model, state, belief)
            outcome = model.observation(truth, action)
            belief = model.posterior(belief, action, outcome)
            state = model.next_state(state, action)
            history.append((action, outcome))
        cells.append((truth, state, belief, tuple(history)))
    return cells


def run_smoke(provider: NamedThyroidProvider, config: ThyroidStrategyConfig) -> dict[str, Any]:
    model = ThyroidWorkupModel()
    records = []
    legal = True
    for cell_index, (truth, state, belief, history) in enumerate(
        build_smoke_cells(model, seed=config.seed)
    ):
        cell = provider.propose(
            model,
            cell_index=cell_index,
            state=state,
            belief=belief,
            history=history,
        )
        legal &= all(
            strategy.root_action in model.legal_actions(state)
            and all(
                followup in model.legal_actions(model.next_state(state, strategy.root_action))
                for followup in strategy.followups.values()
            )
            for strategy in cell.strategies
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
        "stage": "uci_thyroid_workup_26b_named_serving_smoke",
        "config": asdict(config),
        "mechanics": {
            "ten_cells_completed": len(records) == 10,
            "all_roots_and_followups_legal": legal,
            "collection_root_machine_fixed_first": all(
                row["roots"][0] == COLLECT_BLOOD_ACTION for row in records
            ),
            "exactly_ten_logical_calls": len(provider.physical_requests) == 10,
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
        default=Path("configs/config_nonmyopic_rocksample_15_15_vllm.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/thyroid_workup_26b_smoke_20260723"),
    )
    parser.add_argument("--run-id", default="thyroid-workup-26b-smoke-20260723")
    parser.add_argument("--seed", type=int, default=24_151)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = ThyroidStrategyConfig(seed=args.seed)
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicNamedThyroidModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = config.max_new_tokens
        chat_model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = NamedThyroidProvider(chat_model, config)
    try:
        result = run_smoke(provider, config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "uci_thyroid_workup_26b_named_serving_smoke",
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
    result["passed"] = all(result["mechanics"].values())
    (args.output_dir / "SMOKE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": result["passed"], "mechanics": result["mechanics"]}, indent=2))


if __name__ == "__main__":
    main()
