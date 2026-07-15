"""Exploratory LLM candidate-proposal pilot on the frozen UCI Zoo 20Q matrix.

The LLM is deliberately constrained to proposing legal trait IDs. Answers, posterior
updates, exact EIG, and MAP decoding are all deterministic programmatic operations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_oracle_control import (
    TRAITS,
    FrozenMatrix,
    _entropy,
    _posterior_after,
    expected_information_gain,
    load_frozen_matrix,
)


History = tuple[tuple[int, bool], ...]


class CandidateProposalError(RuntimeError):
    """The LLM did not return a complete legal candidate cell."""


class ChatModel(Protocol):
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]: ...


@dataclass(frozen=True)
class PilotConfig:
    num_trials: int = 8
    num_rounds: int = 6
    candidate_width: int = 3
    seed: int = 1304
    bootstrap_replicates: int = 10_000
    temperature: float = 0.7
    candidate_retries: int = 1

    def validate(self) -> None:
        if not 1 <= self.num_trials <= 10:
            raise ValueError("num_trials must be in [1, 10] for the exploration protocol")
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        if self.candidate_width <= 0:
            raise ValueError("candidate_width must be positive")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if self.candidate_retries < 0:
            raise ValueError("candidate_retries must be non-negative")


@dataclass(frozen=True)
class CandidatePool:
    trial_index: int
    history: History
    label: str
    trait_ids: tuple[str, ...]
    actions: tuple[int, ...]
    raw_response: str
    cache_hit: bool


@dataclass
class ArmCounters:
    logical_candidate_calls: int = 0
    cache_hits: int = 0


@dataclass(frozen=True)
class Selection:
    action: int
    candidate_pool: tuple[int, ...]
    base_candidate_pool: tuple[int, ...]
    immediate_scores: dict[int, float]
    total_scores: dict[int, float]
    candidate_call_budget: int
    virtual_depth_two_call_budget: int


@dataclass(frozen=True)
class StepTrace:
    action: int
    trait_id: str
    answer: bool
    candidate_pool: tuple[str, ...]
    base_candidate_pool: tuple[str, ...]
    selected_eig: float
    selection_score: float
    support_size: int
    map_correct: float
    entropy: float
    candidate_call_budget: int
    virtual_depth_two_call_budget: int


@dataclass(frozen=True)
class PolicyTrace:
    arm: str
    target_index: int
    target_name: str
    steps: tuple[StepTrace, ...]
    logical_candidate_calls: int
    candidate_cache_hits: int

    @property
    def accuracy(self) -> tuple[float, ...]:
        return tuple(step.map_correct for step in self.steps)

    @property
    def entropy(self) -> tuple[float, ...]:
        return tuple(step.entropy for step in self.steps)


def _history_text(history: History, matrix: FrozenMatrix) -> str:
    if not history:
        return "No traits have been asked."
    return "\n".join(
        f"- {matrix.traits[action]}: {'yes' if answer else 'no'}" for action, answer in history
    )


def parse_candidate_trait_ids(
    response: str,
    *,
    allowed_traits: tuple[str, ...],
    asked_actions: set[int],
    expected_count: int,
) -> tuple[int, ...]:
    """Strictly parse one complete legal LLM candidate cell.

    Returning a smaller pool, an unknown ID, or a duplicate would silently alter the
    constrained environment, so all of those conditions are terminal for a proposal.
    """

    normalized = response.strip()
    fence_start = "```json\n"
    fence_end = "\n```"
    if normalized.startswith(fence_start):
        if not normalized.endswith(fence_end):
            raise CandidateProposalError("response has an incomplete JSON fence")
        normalized = normalized[len(fence_start) : -len(fence_end)]
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise CandidateProposalError("response is not a JSON object") from exc
    if not isinstance(payload, dict) or set(payload) != {"trait_ids"}:
        raise CandidateProposalError("response must be exactly {'trait_ids': [...]}" )
    trait_ids = payload["trait_ids"]
    if not isinstance(trait_ids, list) or not all(isinstance(item, str) for item in trait_ids):
        raise CandidateProposalError("trait_ids must be a JSON list of strings")
    if len(trait_ids) != expected_count:
        raise CandidateProposalError(f"expected exactly {expected_count} trait IDs")
    if len(set(trait_ids)) != len(trait_ids):
        raise CandidateProposalError("trait IDs must be distinct")
    index = {trait: action for action, trait in enumerate(allowed_traits)}
    unknown = [trait for trait in trait_ids if trait not in index]
    if unknown:
        raise CandidateProposalError(f"unknown trait IDs: {unknown}")
    actions = tuple(index[trait] for trait in trait_ids)
    repeated = [trait for trait, action in zip(trait_ids, actions) if action in asked_actions]
    if repeated:
        raise CandidateProposalError(f"trait IDs were already asked: {repeated}")
    return actions


class LLMCandidateProvider:
    """Caches valid LLM proposal cells by trial, exact history, and proposal label."""

    def __init__(
        self,
        model: ChatModel,
        matrix: FrozenMatrix,
        config: PilotConfig,
    ) -> None:
        self.model = model
        self.matrix = matrix
        self.config = config
        self._cache: dict[tuple[int, History, str, tuple[int, ...]], CandidatePool] = {}
        self.physical_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []

    def _messages(
        self,
        *,
        history: History,
        support: np.ndarray,
        label: str,
        avoid_actions: tuple[int, ...],
    ) -> list[dict[str, str]]:
        asked_actions = {action for action, _answer in history}
        legal_unasked = [trait for action, trait in enumerate(self.matrix.traits) if action not in asked_actions]
        support_names = [self.matrix.names[int(item)] for item in support]
        system = (
            "You propose candidates for a fixed 20 Questions game. You may only choose from the legal trait IDs. "
            "Return exactly one JSON object and no prose: {\"trait_ids\":[\"id1\",\"id2\",\"id3\"]}."
        )
        prompt_lines = [
                f"Candidate cell: {label}.",
                f"Choose exactly {self.config.candidate_width} distinct unasked legal trait IDs.",
                "A separate exact program will evaluate EIG and choose among your candidates.",
                "Observed history:",
                _history_text(history, self.matrix),
                "Consistent target names under the exact uniform posterior:",
                ", ".join(support_names),
                "Unasked legal trait IDs:",
                ", ".join(legal_unasked),
        ]
        if avoid_actions:
            avoid_traits = [self.matrix.traits[action] for action in avoid_actions]
            prompt_lines.extend(
                [
                    "For this width-expansion cell, avoid re-proposing these IDs whenever at least "
                    f"{self.config.candidate_width} other unasked legal IDs remain:",
                    ", ".join(avoid_traits),
                ]
            )
        user = "\n".join(prompt_lines)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def propose(
        self,
        *,
        trial_index: int,
        history: History,
        support: np.ndarray,
        label: str,
        avoid_actions: tuple[int, ...] = (),
    ) -> CandidatePool:
        key = (trial_index, history, label, avoid_actions)
        cached = self._cache.get(key)
        if cached is not None:
            return replace(cached, cache_hit=True)

        messages = self._messages(
            history=history, support=support, label=label, avoid_actions=avoid_actions
        )
        asked_actions = {action for action, _answer in history}
        last_error: CandidateProposalError | None = None
        for attempt in range(self.config.candidate_retries + 1):
            responses = self.model.chat_complete(messages, self.config.temperature, num_responses=1)
            if len(responses) != 1:
                raise CandidateProposalError("candidate model did not return exactly one response")
            response = responses[0]
            try:
                actions = parse_candidate_trait_ids(
                    response,
                    allowed_traits=self.matrix.traits,
                    asked_actions=asked_actions,
                    expected_count=self.config.candidate_width,
                )
            except CandidateProposalError as exc:
                last_error = exc
                self.invalid_responses.append(
                    {
                        "trial_index": trial_index,
                        "history": _serialize_history(history, self.matrix),
                        "label": label,
                        "attempt": attempt,
                        "error": str(exc),
                        "raw_response": response,
                    }
                )
                continue
            pool = CandidatePool(
                trial_index=trial_index,
                history=history,
                label=label,
                trait_ids=tuple(self.matrix.traits[action] for action in actions),
                actions=actions,
                raw_response=response,
                cache_hit=False,
            )
            self._cache[key] = pool
            self.physical_requests.append(
                {
                    "trial_index": trial_index,
                    "history": _serialize_history(history, self.matrix),
                    "label": label,
                    "attempt": attempt,
                    "trait_ids": list(pool.trait_ids),
                    "raw_response": response,
                    "support_size": int(len(support)),
                }
            )
            return pool
        raise CandidateProposalError(
            f"candidate cell trial={trial_index}, label={label} failed after "
            f"{self.config.candidate_retries + 1} attempt(s): {last_error}"
        )


class DeterministicCandidateModel:
    """Offline mechanics-only model for the no-spend dry run."""

    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("dry-run model only supports one response")
        marker = "Unasked legal trait IDs:\n"
        content = messages[-1]["content"]
        legal = content.split(marker, maxsplit=1)[1].split("\n", maxsplit=1)[0].split(", ")
        avoid_marker = "For this width-expansion cell, avoid re-proposing these IDs whenever at least "
        avoid: list[str] = []
        if avoid_marker in content:
            avoid = content.rsplit("\n", maxsplit=1)[1].split(", ")
        novel = [trait for trait in legal if trait not in avoid]
        chosen = novel[:3]
        if len(chosen) < 3:
            chosen.extend(trait for trait in legal if trait not in chosen)
        return [json.dumps({"trait_ids": chosen[:3]})]

    def usage_snapshot(self) -> dict[str, Any]:
        return {"backend": "dry_run", "requests": 0, "cost_usd": 0.0}


def _request_pool(
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    *,
    trial_index: int,
    history: History,
    support: np.ndarray,
    label: str,
    avoid_actions: tuple[int, ...] = (),
) -> CandidatePool:
    counters.logical_candidate_calls += 1
    pool = provider.propose(
        trial_index=trial_index,
        history=history,
        support=support,
        label=label,
        avoid_actions=avoid_actions,
    )
    if pool.cache_hit:
        counters.cache_hits += 1
    return pool


def _choose(pool: tuple[int, ...], scores: dict[int, float]) -> int:
    if not pool:
        raise CandidateProposalError("empty candidate pool cannot be scored")
    return max(pool, key=lambda action: (scores[action], -action))


def _one_step_selection(
    matrix: FrozenMatrix,
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    *,
    trial_index: int,
    history: History,
    support: np.ndarray,
) -> Selection:
    before = counters.logical_candidate_calls
    base = _request_pool(
        provider, counters, trial_index=trial_index, history=history, support=support, label="root"
    )
    scores = {action: expected_information_gain(matrix, support, action) for action in base.actions}
    return Selection(
        action=_choose(base.actions, scores),
        candidate_pool=base.actions,
        base_candidate_pool=base.actions,
        immediate_scores=scores,
        total_scores=scores,
        candidate_call_budget=counters.logical_candidate_calls - before,
        virtual_depth_two_call_budget=1,
    )


def _feasible_branch_count(matrix: FrozenMatrix, support: np.ndarray, root_actions: tuple[int, ...]) -> int:
    return sum(
        int(bool(len(_posterior_after(matrix, support, action, answer))))
        for action in root_actions
        for answer in (False, True)
    )


def _depth_two_selection(
    matrix: FrozenMatrix,
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    *,
    trial_index: int,
    history: History,
    support: np.ndarray,
    remaining_rounds: int,
) -> Selection:
    before = counters.logical_candidate_calls
    base = _request_pool(
        provider, counters, trial_index=trial_index, history=history, support=support, label="root"
    )
    immediate = {action: expected_information_gain(matrix, support, action) for action in base.actions}
    if remaining_rounds == 1:
        return Selection(
            action=_choose(base.actions, immediate),
            candidate_pool=base.actions,
            base_candidate_pool=base.actions,
            immediate_scores=immediate,
            total_scores=immediate,
            candidate_call_budget=counters.logical_candidate_calls - before,
            virtual_depth_two_call_budget=1,
        )

    scores: dict[int, float] = {}
    support_size = len(support)
    feasible_branches = 0
    for root_action in base.actions:
        continuation = 0.0
        for answer in (False, True):
            next_support = _posterior_after(matrix, support, root_action, answer)
            if not len(next_support):
                continue
            feasible_branches += 1
            next_history = history + ((root_action, answer),)
            future = _request_pool(
                provider,
                counters,
                trial_index=trial_index,
                history=next_history,
                support=next_support,
                label="root",
            )
            future_scores = [expected_information_gain(matrix, next_support, action) for action in future.actions]
            continuation += (len(next_support) / support_size) * max(future_scores)
        scores[root_action] = immediate[root_action] + continuation
    budget = counters.logical_candidate_calls - before
    expected_budget = 1 + feasible_branches
    if budget != expected_budget:
        raise AssertionError("depth-two logical call allocation drifted")
    return Selection(
        action=_choose(base.actions, scores),
        candidate_pool=base.actions,
        base_candidate_pool=base.actions,
        immediate_scores=immediate,
        total_scores=scores,
        candidate_call_budget=budget,
        virtual_depth_two_call_budget=expected_budget,
    )


def _matched_width_selection(
    matrix: FrozenMatrix,
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    *,
    trial_index: int,
    history: History,
    support: np.ndarray,
    remaining_rounds: int,
) -> Selection:
    """Spend the virtual d2 candidate-call allocation on wider current-state coverage."""

    before = counters.logical_candidate_calls
    base = _request_pool(
        provider, counters, trial_index=trial_index, history=history, support=support, label="root"
    )
    virtual_budget = 1
    if remaining_rounds > 1:
        virtual_budget += _feasible_branch_count(matrix, support, base.actions)
    pools = [base.actions]
    seen_actions = set(base.actions)
    for sample_index in range(1, virtual_budget):
        extra = _request_pool(
            provider,
            counters,
            trial_index=trial_index,
            history=history,
            support=support,
            label=f"width:{sample_index}",
            avoid_actions=tuple(sorted(seen_actions)),
        )
        pools.append(extra.actions)
        seen_actions.update(extra.actions)
    candidate_pool = tuple(dict.fromkeys(action for pool in pools for action in pool))
    scores = {action: expected_information_gain(matrix, support, action) for action in candidate_pool}
    budget = counters.logical_candidate_calls - before
    if budget != virtual_budget:
        raise AssertionError("matched-width logical call allocation drifted")
    return Selection(
        action=_choose(candidate_pool, scores),
        candidate_pool=candidate_pool,
        base_candidate_pool=base.actions,
        immediate_scores=scores,
        total_scores=scores,
        candidate_call_budget=budget,
        virtual_depth_two_call_budget=virtual_budget,
    )


def run_policy(
    matrix: FrozenMatrix,
    provider: LLMCandidateProvider,
    config: PilotConfig,
    *,
    arm: str,
    trial_index: int,
    target: int,
) -> PolicyTrace:
    if arm not in {"d1_shared", "d2", "d1_matched_width"}:
        raise ValueError(f"unknown arm: {arm}")
    support = np.arange(len(matrix.names), dtype=int)
    history: History = ()
    counters = ArmCounters()
    steps: list[StepTrace] = []
    for round_index in range(config.num_rounds):
        remaining_rounds = config.num_rounds - round_index
        if arm == "d1_shared":
            selection = _one_step_selection(
                matrix, provider, counters, trial_index=trial_index, history=history, support=support
            )
        elif arm == "d2":
            selection = _depth_two_selection(
                matrix,
                provider,
                counters,
                trial_index=trial_index,
                history=history,
                support=support,
                remaining_rounds=remaining_rounds,
            )
        else:
            selection = _matched_width_selection(
                matrix,
                provider,
                counters,
                trial_index=trial_index,
                history=history,
                support=support,
                remaining_rounds=remaining_rounds,
            )
        answer = bool(matrix.values[target, selection.action])
        support = _posterior_after(matrix, support, selection.action, answer)
        history = history + ((selection.action, answer),)
        map_correct = float(int(support[0]) == target)
        steps.append(
            StepTrace(
                action=selection.action,
                trait_id=matrix.traits[selection.action],
                answer=answer,
                candidate_pool=tuple(matrix.traits[action] for action in selection.candidate_pool),
                base_candidate_pool=tuple(matrix.traits[action] for action in selection.base_candidate_pool),
                selected_eig=selection.immediate_scores[selection.action],
                selection_score=selection.total_scores[selection.action],
                support_size=int(len(support)),
                map_correct=map_correct,
                entropy=_entropy(support),
                candidate_call_budget=selection.candidate_call_budget,
                virtual_depth_two_call_budget=selection.virtual_depth_two_call_budget,
            )
        )
    return PolicyTrace(
        arm=arm,
        target_index=target,
        target_name=matrix.names[target],
        steps=tuple(steps),
        logical_candidate_calls=counters.logical_candidate_calls,
        candidate_cache_hits=counters.cache_hits,
    )


def _bootstrap_mean_ci(values: np.ndarray, *, replicates: int, seed: int) -> tuple[float, float]:
    if not len(values):
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 512):
        batch = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        samples[start : start + batch] = np.mean(values[indices], axis=1)
    return (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def _policy_summary(traces: list[PolicyTrace]) -> dict[str, Any]:
    accuracy = np.asarray([trace.accuracy for trace in traces], dtype=float)
    entropy = np.asarray([trace.entropy for trace in traces], dtype=float)
    pool_sizes = np.asarray([[len(step.candidate_pool) for step in trace.steps] for trace in traces], dtype=float)
    calls = np.asarray([[step.candidate_call_budget for step in trace.steps] for trace in traces], dtype=float)
    return {
        "accuracy_auc_mean": float(np.mean(accuracy)),
        "final_accuracy_mean": float(np.mean(accuracy[:, -1])),
        "final_entropy_mean": float(np.mean(entropy[:, -1])),
        "round_accuracy_mean": [float(item) for item in np.mean(accuracy, axis=0)],
        "round_entropy_mean": [float(item) for item in np.mean(entropy, axis=0)],
        "mean_unique_pool_size": float(np.mean(pool_sizes)),
        "mean_logical_candidate_calls_per_decision": float(np.mean(calls)),
        "logical_candidate_calls": int(sum(trace.logical_candidate_calls for trace in traces)),
    }


def _paired_summary(
    first: list[PolicyTrace], second: list[PolicyTrace], *, config: PilotConfig, label: str
) -> dict[str, Any]:
    first_accuracy = np.asarray([trace.accuracy for trace in first], dtype=float)
    second_accuracy = np.asarray([trace.accuracy for trace in second], dtype=float)
    auc_delta = np.mean(first_accuracy, axis=1) - np.mean(second_accuracy, axis=1)
    ci = _bootstrap_mean_ci(
        auc_delta,
        replicates=config.bootstrap_replicates,
        seed=int.from_bytes(f"{config.seed}:{label}".encode("utf-8"), "little", signed=False) % (2**63 - 1),
    )
    return {
        "accuracy_auc_delta_mean": float(np.mean(auc_delta)),
        "accuracy_auc_delta_ci95_descriptive": [ci[0], ci[1]],
        "final_accuracy_delta_mean": float(np.mean(first_accuracy[:, -1] - second_accuracy[:, -1])),
        "round_accuracy_delta_mean": [float(item) for item in np.mean(first_accuracy - second_accuracy, axis=0)],
        "wins_ties_losses": [
            int(np.count_nonzero(auc_delta > 0.0)),
            int(np.count_nonzero(auc_delta == 0.0)),
            int(np.count_nonzero(auc_delta < 0.0)),
        ],
    }


def _serialize_history(history: History, matrix: FrozenMatrix) -> list[dict[str, Any]]:
    return [{"trait_id": matrix.traits[action], "answer": answer} for action, answer in history]


def _serialize_trace(trace: PolicyTrace) -> dict[str, Any]:
    return {
        "arm": trace.arm,
        "target_index": trace.target_index,
        "target_name": trace.target_name,
        "logical_candidate_calls": trace.logical_candidate_calls,
        "candidate_cache_hits": trace.candidate_cache_hits,
        "steps": [asdict(step) for step in trace.steps],
    }


def run_pilot(matrix: FrozenMatrix, provider: LLMCandidateProvider, config: PilotConfig) -> dict[str, Any]:
    config.validate()
    if config.candidate_width > len(matrix.traits) - config.num_rounds + 1:
        raise ValueError("candidate_width leaves too few legal actions for the configured number of rounds")
    rng = np.random.default_rng(config.seed)
    targets = [int(item) for item in rng.integers(0, len(matrix.names), size=config.num_trials)]
    traces: dict[str, list[PolicyTrace]] = {"d1_shared": [], "d2": [], "d1_matched_width": []}
    root_cells_shared = True
    width_allocation_matches = True

    for trial_index, target in enumerate(targets):
        for arm in ("d1_shared", "d2", "d1_matched_width"):
            traces[arm].append(
                run_policy(matrix, provider, config, arm=arm, trial_index=trial_index, target=target)
            )
        root_pools = [traces[arm][-1].steps[0].base_candidate_pool for arm in traces]
        root_cells_shared = root_cells_shared and len(set(root_pools)) == 1
        width_trace = traces["d1_matched_width"][-1]
        width_allocation_matches = width_allocation_matches and all(
            step.candidate_call_budget == step.virtual_depth_two_call_budget for step in width_trace.steps
        )

    d2_vs_d1 = _paired_summary(traces["d2"], traces["d1_shared"], config=config, label="d2-d1")
    d2_vs_width = _paired_summary(
        traces["d2"], traces["d1_matched_width"], config=config, label="d2-width"
    )
    mechanics = {
        "no_invalid_llm_candidate_responses": not provider.invalid_responses,
        "all_initial_candidate_cells_shared": root_cells_shared,
        "width_call_allocation_matches_virtual_depth_two": width_allocation_matches,
        "physical_candidate_requests": len(provider.physical_requests),
        "logical_candidate_requests": int(sum(trace.logical_candidate_calls for arm in traces.values() for trace in arm)),
        "cache_hits": int(sum(trace.candidate_cache_hits for arm in traces.values() for trace in arm)),
    }
    directional_gain = (
        d2_vs_d1["accuracy_auc_delta_mean"] > 0.0
        and d2_vs_width["accuracy_auc_delta_mean"] > 0.0
    )
    promotable = all(
        (
            mechanics["no_invalid_llm_candidate_responses"],
            mechanics["all_initial_candidate_cells_shared"],
            mechanics["width_call_allocation_matches_virtual_depth_two"],
            directional_gain,
        )
    )
    return {
        "schema_version": 1,
        "exploratory_only": True,
        "data": {
            "name": "UCI Zoo frozen attribute matrix",
            "source_sha256": matrix.source_sha256,
            "num_entities": len(matrix.names),
            "num_traits": len(matrix.traits),
            "traits": list(matrix.traits),
        },
        "pilot_config": asdict(config),
        "targets": [{"index": target, "name": matrix.names[target]} for target in targets],
        "summary": {arm: _policy_summary(items) for arm, items in traces.items()},
        "paired": {"d2_minus_d1_shared": d2_vs_d1, "d2_minus_d1_matched_width": d2_vs_width},
        "mechanics": mechanics,
        "promotion": {
            "directional_gain_against_both_controls": directional_gain,
            "promotable": promotable,
            "criterion": "positive paired accuracy-AUC for d2 against both controls plus all mechanics checks",
        },
        "candidate_requests": provider.physical_requests,
        "invalid_candidate_responses": provider.invalid_responses,
        "traces": {arm: [_serialize_trace(trace) for trace in items] for arm, items in traces.items()},
    }


def render_report(summary: dict[str, Any]) -> str:
    paired = summary["paired"]
    mechanics = summary["mechanics"]
    lines = [
        "# UCI Zoo LLM Candidate-Proposal Pilot",
        "",
        "**Exploratory only.** The frozen matrix supplies every answer, likelihood, posterior update, EIG score, and MAP decode. The LLM only proposes legal trait IDs.",
        "",
        "| Arm | Accuracy AUC | Final MAP accuracy | Final entropy | Mean unique pool | Logical calls / decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("d1_shared", "d2", "d1_matched_width"):
        row = summary["summary"][arm]
        lines.append(
            f"| {arm} | {row['accuracy_auc_mean']:.4f} | {row['final_accuracy_mean']:.4f} | "
            f"{row['final_entropy_mean']:.4f} | {row['mean_unique_pool_size']:.2f} | "
            f"{row['mean_logical_candidate_calls_per_decision']:.2f} |"
        )
    lines.extend(["", "## Paired Accuracy-AUC", "", "| Comparison | Mean delta | Descriptive 95% bootstrap CI | W / T / L |", "| --- | ---: | --- | --- |"])
    for label, row in paired.items():
        ci = row["accuracy_auc_delta_ci95_descriptive"]
        wins = row["wins_ties_losses"]
        lines.append(
            f"| {label} | {row['accuracy_auc_delta_mean']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{wins[0]} / {wins[1]} / {wins[2]} |"
        )
    lines.extend(
        [
            "",
            "## Mechanics",
            "",
            f"- No invalid LLM candidate responses: `{mechanics['no_invalid_llm_candidate_responses']}`.",
            f"- Initial candidate cells shared across all arms: `{mechanics['all_initial_candidate_cells_shared']}`.",
            f"- Width allocation equals its virtual depth-two allocation at every decision: `{mechanics['width_call_allocation_matches_virtual_depth_two']}`.",
            f"- Candidate requests: `{mechanics['physical_candidate_requests']}` physical / `{mechanics['logical_candidate_requests']}` logical; cache hits `{mechanics['cache_hits']}`.",
            "",
            "## Decision",
            "",
            f"- Directional d2 gain versus both controls: `{summary['promotion']['directional_gain_against_both_controls']}`.",
            f"- **Promotable to one preregistered confirmatory run: `{summary['promotion']['promotable']}`.**",
            "- This is an eight-task exploratory screen. The intervals are descriptive and are not confirmatory evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = getattr(model, "usage_snapshot", None)
    return snapshot() if callable(snapshot) else {"backend": "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config_nonmyopic_ucizoo_llm_pilot_openrouter.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/nonmyopic/uci_zoo.data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/nonmyopic/ucizoo_llm_pilot"))
    parser.add_argument("--run-id", default="nonmyopic-ucizoo-llm-pilot-20260714")
    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--num-rounds", type=int, default=6)
    parser.add_argument("--candidate-width", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1304)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--candidate-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="run deterministic mechanics without OpenRouter")
    args = parser.parse_args()

    pilot_config = PilotConfig(
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        candidate_width=args.candidate_width,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        temperature=args.temperature,
        candidate_retries=args.candidate_retries,
    )
    pilot_config.validate()
    matrix = load_frozen_matrix(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        model: ChatModel = DeterministicCandidateModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        if not runtime_config.model_pairs:
            raise ValueError("pilot config requires a questioner model pair")
        model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = LLMCandidateProvider(model, matrix, pilot_config)
    try:
        summary = run_pilot(matrix, provider, pilot_config)
    except CandidateProposalError as exc:
        failure = {
            "schema_version": 1,
            "exploratory_only": True,
            "status": "failed_closed",
            "error": str(exc),
            "pilot_config": asdict(pilot_config),
            "candidate_requests": provider.physical_requests,
            "invalid_candidate_responses": provider.invalid_responses,
            "usage": _usage_snapshot(model),
        }
        (args.output_dir / "PILOT_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    summary["usage"] = _usage_snapshot(model)
    summary["run_id"] = args.run_id
    summary["dry_run"] = args.dry_run
    (args.output_dir / "PILOT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "PILOT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary["promotion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
