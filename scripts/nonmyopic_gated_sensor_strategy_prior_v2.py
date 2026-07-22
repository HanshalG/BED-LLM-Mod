"""Indexed branch-policy StrategyEIG on exact gated sensor diagnosis."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.gated_sensor import (
    GatedBranchStrategy,
    GatedSensorModel,
    SensorState,
    score_gated_strategy_exact,
)
from environments.gated_sensor.model import EPSILON
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_gated_sensor_strategy_prior import (
    ARMS,
    ArmName,
    ChatModel,
    PolicyState,
    Selection,
    StrategyCell,
    StrategyProposalError,
    _apply_selection,
    _bootstrap_ci,
    _choose,
    _comparison,
    _exhaustive_selection,
    _predicate_summary,
    _stable_seed,
    _usage_snapshot,
)


@dataclass(frozen=True)
class IndexedStrategyConfig:
    interface_version: str = "indexed_branch_v2"
    num_trials: int = 30
    num_rounds: int = 8
    num_strategies: int = 6
    seed: int = 24_093
    bootstrap_replicates: int = 10_000
    trial_concurrency: int = 32
    temperature: float = 0.0
    validation_retries: int = 1
    screen_accuracy: float = 0.65
    precise_accuracy: float = 0.95

    def validate(self) -> None:
        if self.interface_version != "indexed_branch_v2":
            raise ValueError("interface_version must be indexed_branch_v2")
        if min(
            self.num_trials,
            self.num_rounds,
            self.num_strategies,
            self.bootstrap_replicates,
            self.trial_concurrency,
        ) <= 0:
            raise ValueError("trial, round, strategy, bootstrap, and concurrency counts must be positive")
        if self.num_strategies < 4:
            raise ValueError("num_strategies must cover all initial activation roots and one measurement root")
        if self.validation_retries != 1:
            raise ValueError("indexed v2 permits exactly one validation retry")
        GatedSensorModel(
            screen_accuracy=self.screen_accuracy,
            precise_accuracy=self.precise_accuracy,
        )


def _fixed_roots(
    model: GatedSensorModel,
    *,
    state: SensorState,
    belief: np.ndarray,
    horizon: int,
    count: int,
) -> tuple[str, ...]:
    """Cover all setup roots, then fill with the strongest distinct d1 roots."""

    legal = model.legal_actions(state)
    activations = tuple(action for action in legal if action.startswith("activate:")) if horizon > 1 else ()
    measurements = [action for action in legal if not action.startswith("activate:")]
    measurements.sort(
        key=lambda action: (-model.expected_information_gain(belief, action), legal.index(action))
    )
    needed = count - len(activations)
    if needed <= 0 or needed > len(measurements):
        raise StrategyProposalError("candidate count cannot cover fixed indexed roots")
    return (*activations, *measurements[:needed])


def _branch_menus(
    model: GatedSensorModel,
    *,
    state: SensorState,
    roots: tuple[str, ...],
    horizon: int,
) -> list[dict[str, list[str]]]:
    menus: list[dict[str, list[str]]] = []
    for root in roots:
        root_menus: dict[str, list[str]] = {}
        if horizon > 1:
            child_state = model.next_state(state, root)
            choices = model.legal_actions(child_state)
            if root.startswith("activate:"):
                choices = tuple(action for action in choices if action.startswith("precise:"))
            for outcome in model.outcomes(root):
                key = "none" if outcome is None else outcome
                root_menus[key] = list(choices)
        menus.append(root_menus)
    return menus


def _normalize_response(response: str) -> str:
    normalized = response.strip()
    if "```json\n" in normalized:
        start = normalized.rfind("```json\n") + len("```json\n")
        end = normalized.find("\n```", start)
        if end < 0:
            raise StrategyProposalError("response has an incomplete JSON fence")
        normalized = normalized[start:end]
    return normalized


def compile_indexed_cell(
    response: str,
    *,
    roots: tuple[str, ...],
    menus: list[dict[str, list[str]]],
) -> tuple[GatedBranchStrategy, ...]:
    """Compile branch-local integer choices into executable legal policies."""

    try:
        payload = json.loads(_normalize_response(response))
    except json.JSONDecodeError as exc:
        raise StrategyProposalError("indexed response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"choices"}:
        raise StrategyProposalError("indexed response must contain exactly choices")
    rows = payload["choices"]
    if not isinstance(rows, list) or len(rows) != len(roots):
        raise StrategyProposalError(f"expected exactly {len(roots)} indexed choice rows")
    strategies: list[GatedBranchStrategy] = []
    for slot, (row, root, root_menus) in enumerate(zip(rows, roots, menus, strict=True)):
        expected_width = 2 if root_menus else 0
        if not isinstance(row, list) or len(row) != expected_width:
            raise StrategyProposalError(
                f"slot {slot} must contain exactly {expected_width} integer choices"
            )
        if any(isinstance(index, bool) or not isinstance(index, int) for index in row):
            raise StrategyProposalError(f"slot {slot} choices must all be integers")
        followups: dict[str, str] = {}
        for branch_index, (outcome, choices) in enumerate(root_menus.items()):
            index = row[branch_index]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(choices):
                raise StrategyProposalError(
                    f"slot {slot} index for {outcome} must be in [0, {len(choices) - 1}]"
                )
            followups[outcome] = choices[index]
        name = f"indexed-slot-{slot}"
        description = "Branch-local choices compiled from the model's fixed-shape integer row."
        executable = {
            "name": name,
            "description": description,
            "root_action": root,
            "followups": followups,
        }
        strategies.append(
            GatedBranchStrategy(
                name=name,
                description=description,
                root_action=root,
                followups=followups,
                raw_text=json.dumps(executable, sort_keys=True, separators=(",", ":")),
            )
        )
    return tuple(strategies)


class IndexedStrategyProvider:
    """Generate branch-local menu indices for machine-assigned legal roots."""

    def __init__(self, chat_model: ChatModel, config: IndexedStrategyConfig) -> None:
        self.chat_model = chat_model
        self.config = config
        self._cache: dict[tuple[Any, ...], StrategyCell] = {}
        self._lock = threading.Lock()
        self.logical_calls = 0
        self.cache_hits = 0
        self.physical_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []
        self.local_terminal_cells = 0

    def _messages(
        self,
        model: GatedSensorModel,
        *,
        state: SensorState,
        belief: np.ndarray,
        history: tuple[tuple[str, str | None], ...],
        roots: tuple[str, ...],
        menus: list[dict[str, list[str]]],
    ) -> list[dict[str, str]]:
        slots = [
            {
                "slot": slot,
                "root_action": root,
                "followup_menus": {
                    outcome: [{"index": index, "action": action} for index, action in enumerate(choices)]
                    for outcome, choices in root_menus.items()
                },
            }
            for slot, (root, root_menus) in enumerate(zip(roots, menus, strict=True))
        ]
        system = (
            "You choose observation-contingent follow-ups for exact Bayesian fault diagnosis. "
            "Roots and legal branch menus are machine assigned. Return JSON only and choose integer indices."
        )
        user = "\n".join(
            [
                "INTERFACE=indexed_branch_v2",
                f"Return exactly {len(roots)} strategies in slot order.",
                'Schema: {"choices":[[0,0],[1,0],[2,0],[0,1],[3,2],[1,0]]}',
                "Return one row per slot and exactly two integers per row. For a one-branch activation "
                "slot, the first integer selects the none branch and the second integer is ignored padding. "
                "For a measurement slot, the integers select positive then negative. "
                "Never emit names, action strings, outcome keys, roots, explanations, or extra fields.",
                "Favor nonredundant tests that reduce remaining posterior uncertainty. Positive and negative "
                "branches may need different follow-ups.",
                "Current active panel: " + (state.active_panel or "none"),
                "Current exact predicate marginals: "
                + json.dumps(_predicate_summary(model, belief), separators=(",", ":")),
                "Exact posterior over fault codes (bits are ordered bit-0 through bit-4): "
                + json.dumps(
                    [
                        {
                            "bits": "".join(str(bit) for bit in hidden),
                            "probability": round(float(probability), 10),
                        }
                        for hidden, probability in zip(model.hidden_states, belief)
                        if probability > 1e-12
                    ],
                    separators=(",", ":"),
                ),
                "History: "
                + json.dumps(
                    [{"action": action, "observation": outcome} for action, outcome in history],
                    separators=(",", ":"),
                ),
                "INDEXED_SLOTS="
                + json.dumps(
                    [
                        {
                            **slot,
                            "branch_order": list(slot["followup_menus"]),
                        }
                        for slot in slots
                    ],
                    separators=(",", ":"),
                ),
            ]
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def propose(
        self,
        model: GatedSensorModel,
        *,
        trial_index: int,
        state: SensorState,
        belief: np.ndarray,
        history: tuple[tuple[str, str | None], ...],
        horizon: int,
    ) -> StrategyCell:
        key = (trial_index, state.active_panel, history, horizon)
        with self._lock:
            self.logical_calls += 1
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
        if cached is not None:
            return StrategyCell(cached.strategies, cached.scores, cached.raw_response, True)
        roots = _fixed_roots(
            model,
            state=state,
            belief=belief,
            horizon=horizon,
            count=self.config.num_strategies,
        )
        menus = _branch_menus(model, state=state, roots=roots, horizon=horizon)
        if horizon <= 1:
            response = json.dumps({"choices": [[] for _root in roots]}, separators=(",", ":"))
            strategies = compile_indexed_cell(response, roots=roots, menus=menus)
            scores = tuple(
                score_gated_strategy_exact(model, strategy, state=state, belief=belief, horizon=1)
                for strategy in strategies
            )
            cell = StrategyCell(strategies, scores, response, False)
            with self._lock:
                self.local_terminal_cells += 1
                self._cache[key] = cell
            return cell
        messages = self._messages(
            model,
            state=state,
            belief=belief,
            history=history,
            roots=roots,
            menus=menus,
        )
        context = {
            "trial_index": trial_index,
            "active_panel": state.active_panel,
            "history": [{"action": action, "observation": outcome} for action, outcome in history],
            "horizon": horizon,
            "roots": list(roots),
            "menus": menus,
        }
        error: Exception | None = None
        for attempt in range(self.config.validation_retries + 1):
            response = self.chat_model.chat_complete(messages, self.config.temperature, num_responses=1)[0]
            try:
                strategies = compile_indexed_cell(response, roots=roots, menus=menus)
                scores = tuple(
                    score_gated_strategy_exact(
                        model,
                        strategy,
                        state=state,
                        belief=belief,
                        horizon=horizon,
                    )
                    for strategy in strategies
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
                            "content": f"Invalid indexed cell: {exc}. Return the corrected full JSON only.",
                        },
                    ]
                continue
            cell = StrategyCell(strategies, scores, response, False)
            with self._lock:
                self.physical_requests.append({**context, "attempt": attempt, "raw_response": response})
                self._cache[key] = cell
            return cell
        raise StrategyProposalError(f"indexed cell failed after two attempts: {error}")


class DeterministicIndexedModel:
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("deterministic indexed model supports one response")
        slots = json.loads(messages[-1]["content"].split("INDEXED_SLOTS=", 1)[1])
        choices = [[0, 0] for _slot in slots]
        return [json.dumps({"choices": choices})]


def _strategy_selection_v2(
    model: GatedSensorModel,
    provider: IndexedStrategyProvider,
    state: PolicyState,
    *,
    trial_index: int,
    horizon: int,
    myopic: bool,
) -> Selection:
    cell = provider.propose(
        model,
        trial_index=trial_index,
        state=state.sensor_state,
        belief=state.belief,
        history=state.history,
        horizon=horizon,
    )
    scores = tuple(
        model.expected_information_gain(state.belief, strategy.root_action)
        if myopic
        else exact_score.eig
        for strategy, exact_score in zip(cell.strategies, cell.scores, strict=True)
    )
    index = _choose(scores)
    strategy = cell.strategies[index]
    return Selection(
        action=strategy.root_action,
        planning_score=scores[index],
        immediate_eig=model.expected_information_gain(state.belief, strategy.root_action),
        candidate_roots=tuple(candidate.root_action for candidate in cell.strategies),
        candidate_scores=scores,
        selected_strategy=strategy.raw_text,
        candidate_strategies=tuple(candidate.raw_text for candidate in cell.strategies),
        scorer_units=(len(cell.strategies) if myopic else sum(score.expanded_decision_nodes for score in cell.scores)),
        logical_llm_calls=int(horizon > 1),
    )


def _random_selection_v2(
    model: GatedSensorModel,
    state: PolicyState,
    config: IndexedStrategyConfig,
    *,
    trial_index: int,
    round_index: int,
    horizon: int,
) -> Selection:
    roots = _fixed_roots(
        model,
        state=state.sensor_state,
        belief=state.belief,
        horizon=horizon,
        count=config.num_strategies,
    )
    menus = _branch_menus(model, state=state.sensor_state, roots=roots, horizon=horizon)
    rng = np.random.default_rng(_stable_seed(config.seed, "indexed-random", trial_index, round_index))
    choices = []
    for root_menus in menus:
        row = [int(rng.integers(len(branch_choices))) for branch_choices in root_menus.values()]
        if root_menus and len(row) == 1:
            row.append(0)
        choices.append(row)
    strategies = compile_indexed_cell(
        json.dumps({"choices": choices}),
        roots=roots,
        menus=menus,
    )
    scores = tuple(
        score_gated_strategy_exact(
            model,
            strategy,
            state=state.sensor_state,
            belief=state.belief,
            horizon=horizon,
        )
        for strategy in strategies
    )
    index = _choose(tuple(score.eig for score in scores))
    strategy = strategies[index]
    return Selection(
        action=strategy.root_action,
        planning_score=scores[index].eig,
        immediate_eig=model.expected_information_gain(state.belief, strategy.root_action),
        candidate_roots=roots,
        candidate_scores=tuple(score.eig for score in scores),
        selected_strategy=strategy.raw_text,
        candidate_strategies=tuple(candidate.raw_text for candidate in strategies),
        scorer_units=sum(score.expanded_decision_nodes for score in scores),
        logical_llm_calls=0,
    )


def _run_trial_v2(
    trial_index: int,
    *,
    model: GatedSensorModel,
    provider: IndexedStrategyProvider,
    config: IndexedStrategyConfig,
) -> dict[str, Any]:
    truth_rng = np.random.default_rng(_stable_seed(config.seed, "indexed-truth", trial_index))
    truth_index = int(truth_rng.integers(len(model.hidden_states)))
    states = {arm: PolicyState(model.initial_belief.copy(), model.initial_state) for arm in ARMS}
    for round_index in range(config.num_rounds):
        horizon = min(2, config.num_rounds - round_index)
        for arm in ARMS:
            state = states[arm]
            if arm == "strategy_eig":
                selection = _strategy_selection_v2(
                    model, provider, state, trial_index=trial_index, horizon=horizon, myopic=False
                )
            elif arm == "shared_d1":
                selection = _strategy_selection_v2(
                    model, provider, state, trial_index=trial_index, horizon=horizon, myopic=True
                )
            elif arm == "random_strategy":
                selection = _random_selection_v2(
                    model,
                    state,
                    config,
                    trial_index=trial_index,
                    round_index=round_index,
                    horizon=horizon,
                )
            elif arm == "exhaustive_d1":
                selection = _exhaustive_selection(model, state, depth=1)
            else:
                selection = _exhaustive_selection(model, state, depth=horizon)
            _apply_selection(
                model,
                state,
                selection,
                arm=arm,
                truth_index=truth_index,
                config=config,
                trial_index=trial_index,
                round_index=round_index,
            )
    traces: dict[str, Any] = {}
    for arm, state in states.items():
        entropy = [step["entropy_after"] for step in state.steps]
        truth = [step["truth_log_probability"] for step in state.steps]
        traces[arm] = {
            "arm": arm,
            "trial_index": trial_index,
            "truth_index": truth_index,
            "entropy_auc": float(np.mean(entropy)),
            "truth_log_probability_auc": float(np.mean(truth)),
            "final_entropy": entropy[-1],
            "final_truth_log_probability": truth[-1],
            "final_map_accuracy": float(model.decode_map_index(state.belief) == truth_index),
            "steps": state.steps,
        }
    return traces


def run_experiment_v2(provider: IndexedStrategyProvider, config: IndexedStrategyConfig) -> dict[str, Any]:
    config.validate()
    model = GatedSensorModel(
        screen_accuracy=config.screen_accuracy,
        precise_accuracy=config.precise_accuracy,
    )
    with ThreadPoolExecutor(max_workers=config.trial_concurrency) as executor:
        trials = list(
            executor.map(
                lambda index: _run_trial_v2(index, model=model, provider=provider, config=config),
                range(config.num_trials),
            )
        )
    traces = {arm: [trial[arm] for trial in trials] for arm in ARMS}
    comparisons = {
        f"strategy_eig_minus_{control}": _comparison(
            traces["strategy_eig"],
            traces[control],
            label=f"indexed-strategy-minus-{control}",
            config=config,
        )
        for control in ("shared_d1", "exhaustive_d1", "random_strategy", "exhaustive_d2")
    }
    mechanics = {
        "paired_trials_and_truths": all(
            [(trace["trial_index"], trace["truth_index"]) for trace in traces[arm]]
            == [(trace["trial_index"], trace["truth_index"]) for trace in traces["strategy_eig"]]
            for arm in ARMS
        ),
        "all_selected_actions_legal": all(
            step["action"] in model.legal_actions(SensorState(step["state_before"]))
            for arm_traces in traces.values()
            for trace in arm_traces
            for step in trace["steps"]
        ),
        "initial_activation_roots_shared_with_random": all(
            strategy["steps"][0]["candidate_roots"] == random["steps"][0]["candidate_roots"]
            for strategy, random in zip(traces["strategy_eig"], traces["random_strategy"], strict=True)
        ),
        "rollout_scoring_made_no_llm_calls": True,
        "terminal_selection_made_no_llm_calls": True,
    }
    primary = ("shared_d1", "exhaustive_d1", "random_strategy")
    gate = all(mechanics.values()) and all(
        comparisons[f"strategy_eig_minus_{control}"]["entropy_auc_gain_ci95"][0] > 0.0
        for control in primary
    )
    return {
        "schema_version": 2,
        "config": asdict(config),
        "comparisons": comparisons,
        "mechanics": mechanics,
        "gate": {
            "passed": gate,
            "rule": "indexed StrategyEIG entropy-AUC lower CI exceeds shared d1, exhaustive d1, and matched random",
        },
        "provider": {
            "logical_calls": provider.logical_calls,
            "physical_requests": len(provider.physical_requests) + len(provider.invalid_responses),
            "accepted_requests": len(provider.physical_requests),
            "invalid_responses": len(provider.invalid_responses),
            "cache_hits": provider.cache_hits,
            "local_terminal_cells": provider.local_terminal_cells,
        },
        "traces": traces,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_nonmyopic_gated_sensor_openrouter.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/gated_sensor_strategy_v2_20260722"),
    )
    parser.add_argument("--run-id", default="nonmyopic-gated-sensor-strategy-v2-20260722")
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--num-strategies", type=int, default=6)
    parser.add_argument("--seed", type=int, default=24_093)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--trial-concurrency", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = IndexedStrategyConfig(
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        num_strategies=args.num_strategies,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        trial_concurrency=args.trial_concurrency,
    )
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicIndexedModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        chat_model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = IndexedStrategyProvider(chat_model, config)
    try:
        summary = run_experiment_v2(provider, config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 2,
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage_snapshot(chat_model),
        }
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    summary["usage"] = _usage_snapshot(chat_model)
    summary["run_id"] = args.run_id
    summary["dry_run"] = args.dry_run
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"gate": summary["gate"], "provider": summary["provider"]}, indent=2))


if __name__ == "__main__":
    main()
