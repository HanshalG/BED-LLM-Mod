"""Bounded LLM candidate-proposal pilot on the exact Rock Diagnosis task.

The LLM only proposes a small cell of legal action identifiers. The environment,
hidden target, observations, posterior updates, EIG values, decode, and every policy
choice are exact. The three arms differ only in acquisition scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Literal, Protocol

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from environments.rock_diagnosis.core import EPSILON
from helpers import Config, load_config
from model_factory import build_model_adapter


History = tuple[tuple[str, str | None], ...]
ArmName = Literal["d1_shared", "d2", "d1_call_matched_width"]


class CandidateProposalError(RuntimeError):
    """The candidate model could not produce a complete legal cell."""


class ChatModel(Protocol):
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]: ...


@dataclass(frozen=True)
class PilotConfig:
    map_name: str = "3-6"
    num_trials: int = 8
    num_rounds: int = 8
    candidate_width: int = 3
    seed: int = 2304
    bootstrap_replicates: int = 10_000
    temperature: float = 0.0
    candidate_retries: int = 1
    half_efficiency_distance: float = math.log(2.0)
    exploratory_only: bool = True

    def validate(self) -> None:
        paper_map = get_paper_map(self.map_name)
        if self.exploratory_only and not 1 <= self.num_trials <= 10:
            raise ValueError("num_trials must be in [1, 10] for an exploration pilot")
        if not self.exploratory_only and self.num_trials < 12:
            raise ValueError("a confirmatory run requires at least 12 paired trajectories")
        if self.num_rounds < 2:
            raise ValueError("num_rounds must be at least two")
        if self.candidate_width <= 0:
            raise ValueError("candidate_width must be positive")
        min_legal_actions = 2 + len(paper_map.rock_positions)
        if self.candidate_width > min_legal_actions:
            raise ValueError("candidate_width exceeds the smallest legal action set on this map")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if self.candidate_retries < 0:
            raise ValueError("candidate_retries must be non-negative")
        if self.half_efficiency_distance <= 0.0:
            raise ValueError("half_efficiency_distance must be positive")


@dataclass(frozen=True)
class CandidatePool:
    trial_index: int
    position: tuple[int, int]
    history: History
    label: str
    action_ids: tuple[str, ...]
    raw_response: str
    cache_hit: bool


@dataclass
class ArmCounters:
    logical_candidate_calls: int = 0
    cache_hits: int = 0


@dataclass(frozen=True)
class Selection:
    action: str
    candidate_pool: tuple[str, ...]
    base_candidate_pool: tuple[str, ...]
    immediate_scores: dict[str, float]
    total_scores: dict[str, float]
    candidate_call_budget: int
    virtual_future_cells: int


@dataclass(frozen=True)
class StepTrace:
    action: str
    position_before: tuple[int, int]
    observation: str | None
    selected_eig: float
    selection_score: float
    entropy: float
    map_correct: float
    truth_log_probability: float
    candidate_pool: tuple[str, ...]
    base_candidate_pool: tuple[str, ...]
    candidate_call_budget: int
    virtual_future_cells: int


@dataclass(frozen=True)
class PolicyTrace:
    arm: ArmName
    trial_index: int
    truth_index: int
    steps: tuple[StepTrace, ...]
    logical_candidate_calls: int
    candidate_cache_hits: int

    @property
    def final_entropy(self) -> float:
        return self.steps[-1].entropy

    @property
    def final_map_accuracy(self) -> float:
        return self.steps[-1].map_correct

    @property
    def final_truth_log_probability(self) -> float:
        return self.steps[-1].truth_log_probability


def _stable_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def _uniform(*parts: Any) -> float:
    return _stable_seed(*parts) / 2**64


def _normalize_json_response(response: str) -> str:
    normalized = response.strip()
    fence_start = "```json\n"
    fence_end = "\n```"
    if normalized.startswith(fence_start):
        if not normalized.endswith(fence_end):
            raise CandidateProposalError("response has an incomplete JSON fence")
        normalized = normalized[len(fence_start) : -len(fence_end)]
    return normalized


def parse_candidate_action_ids(
    response: str, *, allowed_actions: tuple[str, ...], expected_count: int
) -> tuple[str, ...]:
    """Strictly parse one complete legal candidate cell without substitutions."""

    try:
        payload = json.loads(_normalize_json_response(response))
    except json.JSONDecodeError as exc:
        raise CandidateProposalError("response is not a JSON object") from exc
    if not isinstance(payload, dict) or set(payload) != {"action_ids"}:
        raise CandidateProposalError("response must be exactly {'action_ids': [...]}" )
    action_ids = payload["action_ids"]
    if not isinstance(action_ids, list) or not all(isinstance(item, str) for item in action_ids):
        raise CandidateProposalError("action_ids must be a JSON list of strings")
    if len(action_ids) != expected_count:
        raise CandidateProposalError(f"expected exactly {expected_count} action IDs")
    if len(set(action_ids)) != len(action_ids):
        raise CandidateProposalError("action IDs must be distinct")
    invalid = [action for action in action_ids if action not in allowed_actions]
    if invalid:
        raise CandidateProposalError(f"action IDs are not legal at this position: {invalid}")
    return tuple(action_ids)


def _history_text(history: History) -> str:
    if not history:
        return "No actions have been taken."
    return "\n".join(
        f"- {action}: {'no observation' if observation is None else observation}"
        for action, observation in history
    )


def _state_label(state: tuple[str, ...]) -> str:
    return "".join("G" if str(value).lower() == "good" else "B" for value in state)


class LLMCandidateProvider:
    """Caches legal proposal cells by exact simulator state and purpose label."""

    def __init__(self, model: ChatModel, environment: RockDiagnosisModel, config: PilotConfig) -> None:
        self.model = model
        self.environment = environment
        self.config = config
        self._cache: dict[tuple[int, tuple[int, int], History, str, tuple[str, ...]], CandidatePool] = {}
        self.physical_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []

    def _messages(
        self,
        *,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
        label: str,
        avoid_actions: tuple[str, ...],
    ) -> list[dict[str, str]]:
        legal_actions = self.environment.legal_actions(position)
        ordered = np.argsort(-belief, kind="stable")
        belief_lines = [
            f"- rock types {_state_label(self.environment.hidden_states[int(index)])}: {belief[int(index)]:.6f}"
            for index in ordered
        ]
        system = (
            "You propose a candidate cell for an exact Rock Diagnosis information task. "
            "Do not solve the task or explain your reasoning. Return exactly one JSON object and no prose: "
            '{"action_ids":["move-EAST","check-2","check-1"]}.'
        )
        lines = [
            f"Candidate cell: {label}.",
            f"Grid side length: {self.environment.map_spec.grid_size}.",
            f"Rock coordinates by ID: {list(enumerate(self.environment.map_spec.rock_positions))}.",
            f"Current rover position: {position}.",
            f"Choose exactly {self.config.candidate_width} distinct legal action IDs.",
            "A separate exact program scores EIG and selects one proposed action.",
            "Exact posterior over the full latent rock-type vector:",
            *belief_lines,
            "Action and observation history:",
            _history_text(history),
            "Legal action IDs:",
            ", ".join(legal_actions),
        ]
        if avoid_actions:
            lines.extend(
                [
                    "For this call-matched width expansion, avoid these IDs whenever possible:",
                    ", ".join(avoid_actions),
                ]
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]

    def propose(
        self,
        *,
        trial_index: int,
        position: tuple[int, int],
        belief: np.ndarray,
        history: History,
        label: str,
        avoid_actions: tuple[str, ...] = (),
    ) -> CandidatePool:
        key = (trial_index, position, history, label, avoid_actions)
        cached = self._cache.get(key)
        if cached is not None:
            return replace(cached, cache_hit=True)
        legal_actions = self.environment.legal_actions(position)
        messages = self._messages(
            position=position,
            belief=belief,
            history=history,
            label=label,
            avoid_actions=avoid_actions,
        )
        last_error: CandidateProposalError | None = None
        for attempt in range(self.config.candidate_retries + 1):
            responses = self.model.chat_complete(messages, self.config.temperature, num_responses=1)
            if len(responses) != 1:
                raise CandidateProposalError("candidate model did not return exactly one response")
            response = responses[0]
            try:
                action_ids = parse_candidate_action_ids(
                    response, allowed_actions=legal_actions, expected_count=self.config.candidate_width
                )
            except CandidateProposalError as exc:
                last_error = exc
                self.invalid_responses.append(
                    {
                        "trial_index": trial_index,
                        "position": list(position),
                        "history": _serialize_history(history),
                        "label": label,
                        "attempt": attempt,
                        "error": str(exc),
                        "raw_response": response,
                    }
                )
                if attempt < self.config.candidate_retries:
                    # At temperature zero, replaying an unchanged invalid request simply
                    # recreates the same completion. The repair remains bounded and
                    # transparent: show the rejected cell and restate the legal set.
                    messages = [
                        *messages,
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": (
                                f"The previous response is invalid: {exc}. Do not repeat it. "
                                f"Return exactly one JSON object with {self.config.candidate_width} distinct IDs "
                                f"chosen only from this legal list: {', '.join(legal_actions)}."
                            ),
                        },
                    ]
                continue
            pool = CandidatePool(
                trial_index=trial_index,
                position=position,
                history=history,
                label=label,
                action_ids=action_ids,
                raw_response=response,
                cache_hit=False,
            )
            self._cache[key] = pool
            self.physical_requests.append(
                {
                    "trial_index": trial_index,
                    "position": list(position),
                    "history": _serialize_history(history),
                    "label": label,
                    "attempt": attempt,
                    "action_ids": list(action_ids),
                    "raw_response": response,
                    "legal_action_ids": list(legal_actions),
                }
            )
            return pool
        raise CandidateProposalError(
            f"candidate cell trial={trial_index}, position={position}, label={label} failed after "
            f"{self.config.candidate_retries + 1} attempt(s): {last_error}"
        )


class DeterministicCandidateModel:
    """Offline legal proposer used only for no-spend mechanics tests."""

    def __init__(self, candidate_width: int) -> None:
        self.candidate_width = candidate_width

    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("dry-run model supports one response")
        content = messages[-1]["content"]
        marker = "Legal action IDs:\n"
        legal = content.split(marker, maxsplit=1)[1].split("\n", maxsplit=1)[0].split(", ")
        avoid: list[str] = []
        avoid_marker = "For this call-matched width expansion, avoid these IDs whenever possible:\n"
        if avoid_marker in content:
            avoid = content.split(avoid_marker, maxsplit=1)[1].split("\n", maxsplit=1)[0].split(", ")
        preferred = ["move-EAST", "check-2", "check-1", "check-0", "move-NORTH", "move-SOUTH", "move-WEST"]
        chosen = [action for action in preferred if action in legal and action not in avoid]
        chosen.extend(action for action in legal if action not in chosen and action not in avoid)
        if len(chosen) < self.candidate_width:
            chosen.extend(action for action in legal if action not in chosen)
        return [json.dumps({"action_ids": chosen[: self.candidate_width]})]

    def usage_snapshot(self) -> dict[str, Any]:
        return {"backend": "dry_run", "requests": 0, "cost_usd": 0.0}


def _request_pool(
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    *,
    trial_index: int,
    position: tuple[int, int],
    belief: np.ndarray,
    history: History,
    label: str,
    avoid_actions: tuple[str, ...] = (),
) -> CandidatePool:
    counters.logical_candidate_calls += 1
    pool = provider.propose(
        trial_index=trial_index,
        position=position,
        belief=belief,
        history=history,
        label=label,
        avoid_actions=avoid_actions,
    )
    if pool.cache_hit:
        counters.cache_hits += 1
    return pool


def _choose(actions: tuple[str, ...], scores: dict[str, float]) -> str:
    return max(actions, key=lambda action: (scores[action], -actions.index(action)))


def _future_cells(
    environment: RockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    base_pool: tuple[str, ...],
) -> tuple[tuple[str, str | None], ...]:
    return tuple(
        (action, outcome)
        for action in base_pool
        for outcome in environment.outcomes(action)
        if environment.outcome_probability(position, belief, action, outcome) > EPSILON
    )


def _one_step_selection(
    environment: RockDiagnosisModel,
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    *,
    trial_index: int,
    position: tuple[int, int],
    belief: np.ndarray,
    history: History,
) -> Selection:
    before = counters.logical_candidate_calls
    base = _request_pool(
        provider, counters, trial_index=trial_index, position=position, belief=belief, history=history, label="base"
    )
    scores = {action: environment.expected_information_gain(position, belief, action) for action in base.action_ids}
    return Selection(
        action=_choose(base.action_ids, scores),
        candidate_pool=base.action_ids,
        base_candidate_pool=base.action_ids,
        immediate_scores=scores,
        total_scores=scores,
        candidate_call_budget=counters.logical_candidate_calls - before,
        virtual_future_cells=0,
    )


def _depth_two_selection(
    environment: RockDiagnosisModel,
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    *,
    trial_index: int,
    position: tuple[int, int],
    belief: np.ndarray,
    history: History,
    remaining_rounds: int,
) -> Selection:
    before = counters.logical_candidate_calls
    base = _request_pool(
        provider, counters, trial_index=trial_index, position=position, belief=belief, history=history, label="base"
    )
    immediate = {action: environment.expected_information_gain(position, belief, action) for action in base.action_ids}
    cells = _future_cells(environment, position=position, belief=belief, base_pool=base.action_ids) if remaining_rounds > 1 else ()
    scores: dict[str, float] = {}
    for action in base.action_ids:
        continuation = 0.0
        next_position = environment.next_position(position, action)
        for outcome in environment.outcomes(action):
            probability = environment.outcome_probability(position, belief, action, outcome)
            if probability <= EPSILON or remaining_rounds == 1:
                continue
            posterior = environment.posterior(position, belief, action, outcome)
            future = _request_pool(
                provider,
                counters,
                trial_index=trial_index,
                position=next_position,
                belief=posterior,
                history=history + ((action, outcome),),
                label=f"future:{action}:{outcome}",
            )
            continuation += probability * max(
                environment.expected_information_gain(next_position, posterior, future_action)
                for future_action in future.action_ids
            )
        scores[action] = immediate[action] + continuation
    budget = counters.logical_candidate_calls - before
    expected_budget = 1 + len(cells)
    if budget != expected_budget:
        raise AssertionError("depth-two candidate-call accounting drifted")
    return Selection(
        action=_choose(base.action_ids, scores),
        candidate_pool=base.action_ids,
        base_candidate_pool=base.action_ids,
        immediate_scores=immediate,
        total_scores=scores,
        candidate_call_budget=budget,
        virtual_future_cells=len(cells),
    )


def _call_matched_width_selection(
    environment: RockDiagnosisModel,
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    *,
    trial_index: int,
    position: tuple[int, int],
    belief: np.ndarray,
    history: History,
    remaining_rounds: int,
) -> Selection:
    before = counters.logical_candidate_calls
    base = _request_pool(
        provider, counters, trial_index=trial_index, position=position, belief=belief, history=history, label="base"
    )
    cells = _future_cells(environment, position=position, belief=belief, base_pool=base.action_ids) if remaining_rounds > 1 else ()
    candidate_pool = list(base.action_ids)
    for cell_index, _cell in enumerate(cells):
        extra = _request_pool(
            provider,
            counters,
            trial_index=trial_index,
            position=position,
            belief=belief,
            history=history,
            label=f"current-width:{cell_index}",
            avoid_actions=tuple(candidate_pool),
        )
        candidate_pool.extend(action for action in extra.action_ids if action not in candidate_pool)
    pool = tuple(candidate_pool)
    scores = {action: environment.expected_information_gain(position, belief, action) for action in pool}
    budget = counters.logical_candidate_calls - before
    expected_budget = 1 + len(cells)
    if budget != expected_budget:
        raise AssertionError("call-matched width candidate-call accounting drifted")
    return Selection(
        action=_choose(pool, scores),
        candidate_pool=pool,
        base_candidate_pool=base.action_ids,
        immediate_scores=scores,
        total_scores=scores,
        candidate_call_budget=budget,
        virtual_future_cells=len(cells),
    )


def run_policy(
    environment: RockDiagnosisModel,
    provider: LLMCandidateProvider,
    config: PilotConfig,
    *,
    arm: ArmName,
    trial_index: int,
    truth_index: int,
) -> PolicyTrace:
    belief = environment.initial_belief.copy()
    position = environment.map_spec.start_position
    history: History = ()
    counters = ArmCounters()
    check_counts: dict[tuple[tuple[int, int], int], int] = {}
    steps: list[StepTrace] = []
    for round_index in range(config.num_rounds):
        remaining_rounds = config.num_rounds - round_index
        if arm == "d1_shared":
            selection = _one_step_selection(
                environment, provider, counters, trial_index=trial_index, position=position, belief=belief, history=history
            )
        elif arm == "d2":
            selection = _depth_two_selection(
                environment, provider, counters, trial_index=trial_index, position=position, belief=belief,
                history=history, remaining_rounds=remaining_rounds,
            )
        elif arm == "d1_call_matched_width":
            selection = _call_matched_width_selection(
                environment, provider, counters, trial_index=trial_index, position=position, belief=belief,
                history=history, remaining_rounds=remaining_rounds,
            )
        else:
            raise ValueError(f"unknown arm: {arm}")
        if selection.action not in environment.legal_actions(position):
            raise AssertionError("selected action is not legal")
        check_id = environment.check_id(selection.action)
        if check_id is None:
            observation: str | None = None
        else:
            key = (position, check_id)
            repeat_index = check_counts.get(key, 0)
            check_counts[key] = repeat_index + 1
            probability_good = float(environment.likelihood_vector(position, selection.action, "good")[truth_index])
            observation = (
                "good"
                if _uniform(config.seed, "rock-diagnosis-observation", trial_index, position, check_id, repeat_index)
                < probability_good
                else "bad"
            )
        selected_eig = environment.expected_information_gain(position, belief, selection.action)
        belief = environment.posterior(position, belief, selection.action, observation)
        steps.append(
            StepTrace(
                action=selection.action,
                position_before=position,
                observation=observation,
                selected_eig=selected_eig,
                selection_score=selection.total_scores[selection.action],
                entropy=environment.entropy(belief),
                map_correct=float(environment.decode_map_index(belief) == truth_index),
                truth_log_probability=float(math.log(max(float(belief[truth_index]), np.finfo(float).tiny))),
                candidate_pool=selection.candidate_pool,
                base_candidate_pool=selection.base_candidate_pool,
                candidate_call_budget=selection.candidate_call_budget,
                virtual_future_cells=selection.virtual_future_cells,
            )
        )
        history = history + ((selection.action, observation),)
        position = environment.next_position(position, selection.action)
    return PolicyTrace(
        arm=arm,
        trial_index=trial_index,
        truth_index=truth_index,
        steps=tuple(steps),
        logical_candidate_calls=counters.logical_candidate_calls,
        candidate_cache_hits=counters.cache_hits,
    )


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 512):
        batch = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        samples[start : start + batch] = np.mean(values[indices], axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _summary(traces: list[PolicyTrace]) -> dict[str, Any]:
    entropy = np.asarray([[step.entropy for step in trace.steps] for trace in traces], dtype=float)
    accuracy = np.asarray([[step.map_correct for step in trace.steps] for trace in traces], dtype=float)
    truth_log = np.asarray([[step.truth_log_probability for step in trace.steps] for trace in traces], dtype=float)
    pools = np.asarray([[len(step.candidate_pool) for step in trace.steps] for trace in traces], dtype=float)
    calls = np.asarray([[step.candidate_call_budget for step in trace.steps] for trace in traces], dtype=float)
    selected_eig = np.asarray([[step.selected_eig for step in trace.steps] for trace in traces], dtype=float)
    initial_moves = np.asarray([trace.steps[0].action.startswith("move-") for trace in traces], dtype=float)
    return {
        "entropy_auc_mean": float(np.mean(entropy)),
        "final_entropy_mean": float(np.mean(entropy[:, -1])),
        "final_map_accuracy_mean": float(np.mean(accuracy[:, -1])),
        "final_truth_log_probability_mean": float(np.mean(truth_log[:, -1])),
        "round_entropy_mean": [float(value) for value in np.mean(entropy, axis=0)],
        "round_map_accuracy_mean": [float(value) for value in np.mean(accuracy, axis=0)],
        "mean_unique_pool_size": float(np.mean(pools)),
        "mean_logical_candidate_calls_per_decision": float(np.mean(calls)),
        "mean_selected_eig": float(np.mean(selected_eig)),
        "initial_move_rate": float(np.mean(initial_moves)),
        "logical_candidate_calls": int(sum(trace.logical_candidate_calls for trace in traces)),
    }


def _paired_summary(
    first: list[PolicyTrace], second: list[PolicyTrace], *, config: PilotConfig, label: str
) -> dict[str, Any]:
    first_entropy = np.asarray([trace.final_entropy for trace in first], dtype=float)
    second_entropy = np.asarray([trace.final_entropy for trace in second], dtype=float)
    entropy_reduction = second_entropy - first_entropy
    first_accuracy = np.asarray([trace.final_map_accuracy for trace in first], dtype=float)
    second_accuracy = np.asarray([trace.final_map_accuracy for trace in second], dtype=float)
    first_truth_log = np.asarray([trace.final_truth_log_probability for trace in first], dtype=float)
    second_truth_log = np.asarray([trace.final_truth_log_probability for trace in second], dtype=float)
    first_auc = np.asarray([np.mean([step.entropy for step in trace.steps]) for trace in first], dtype=float)
    second_auc = np.asarray([np.mean([step.entropy for step in trace.steps]) for trace in second], dtype=float)
    ci = _bootstrap_mean_ci(
        entropy_reduction,
        seed=_stable_seed(config.seed, "rock-diagnosis-pilot-bootstrap", label),
        replicates=config.bootstrap_replicates,
    )
    return {
        "final_entropy_reduction_mean": float(np.mean(entropy_reduction)),
        "final_entropy_reduction_ci95_descriptive": [ci[0], ci[1]],
        "entropy_auc_reduction_mean": float(np.mean(second_auc - first_auc)),
        "final_map_accuracy_delta_mean": float(np.mean(first_accuracy - second_accuracy)),
        "final_truth_log_probability_delta_mean": float(np.mean(first_truth_log - second_truth_log)),
        "wins_ties_losses": [
            int(np.count_nonzero(entropy_reduction > 0.0)),
            int(np.count_nonzero(entropy_reduction == 0.0)),
            int(np.count_nonzero(entropy_reduction < 0.0)),
        ],
    }


def _serialize_history(history: History) -> list[dict[str, Any]]:
    return [{"action": action, "observation": observation} for action, observation in history]


def _serialize_trace(trace: PolicyTrace) -> dict[str, Any]:
    return {
        "arm": trace.arm,
        "trial_index": trace.trial_index,
        "truth_index": trace.truth_index,
        "logical_candidate_calls": trace.logical_candidate_calls,
        "candidate_cache_hits": trace.candidate_cache_hits,
        "steps": [asdict(step) for step in trace.steps],
    }


def run_pilot(provider: LLMCandidateProvider, config: PilotConfig) -> dict[str, Any]:
    config.validate()
    environment = provider.environment
    traces: dict[ArmName, list[PolicyTrace]] = {
        "d1_shared": [], "d2": [], "d1_call_matched_width": [],
    }
    roots_shared = True
    width_matches = True
    all_actions_legal = True
    truth_indices: list[int] = []
    for trial_index in range(config.num_trials):
        truth_rng = np.random.default_rng(_stable_seed(config.seed, "rock-diagnosis-truth", trial_index))
        truth_index = int(truth_rng.integers(len(environment.hidden_states)))
        truth_indices.append(truth_index)
        for arm in traces:
            traces[arm].append(
                run_policy(environment, provider, config, arm=arm, trial_index=trial_index, truth_index=truth_index)
            )
        roots = [traces[arm][-1].steps[0].base_candidate_pool for arm in traces]
        roots_shared = roots_shared and len(set(roots)) == 1
        width_trace = traces["d1_call_matched_width"][-1]
        width_matches = width_matches and all(
            step.candidate_call_budget == 1 + step.virtual_future_cells for step in width_trace.steps
        )
        all_actions_legal = all_actions_legal and all(
            step.action in environment.legal_actions(step.position_before)
            for arm in traces.values()
            for step in arm[-1].steps
        )
    d2_vs_d1 = _paired_summary(traces["d2"], traces["d1_shared"], config=config, label="d2-d1")
    d2_vs_width = _paired_summary(
        traces["d2"], traces["d1_call_matched_width"], config=config, label="d2-width"
    )
    mechanics = {
        "terminal_candidate_cell_failures": 0,
        "all_selected_actions_legal": all_actions_legal,
        "all_initial_candidate_cells_shared": roots_shared,
        "width_call_allocation_matches_virtual_depth_two": width_matches,
        "raw_rejected_candidate_attempts": len(provider.invalid_responses),
        "physical_candidate_requests": len(provider.physical_requests),
        "logical_candidate_requests": int(sum(trace.logical_candidate_calls for arm in traces.values() for trace in arm)),
        "cache_hits": int(sum(trace.candidate_cache_hits for arm in traces.values() for trace in arm)),
    }
    directional_gain = (
        d2_vs_d1["final_entropy_reduction_mean"] > 0.0
        and d2_vs_width["final_entropy_reduction_mean"] > 0.0
    )
    mechanics_pass = all(
        (
            mechanics["all_selected_actions_legal"],
            mechanics["all_initial_candidate_cells_shared"],
            mechanics["width_call_allocation_matches_virtual_depth_two"],
        )
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "exploratory_only": config.exploratory_only,
        "environment": {
            "name": "Rock Diagnosis Figure 4 3-6",
            "map_name": config.map_name,
            "grid_size": environment.map_spec.grid_size,
            "rock_positions": [list(position) for position in environment.map_spec.rock_positions],
            "start_position": list(environment.map_spec.start_position),
            "num_hidden_states": len(environment.hidden_states),
            "half_efficiency_distance": config.half_efficiency_distance,
        },
        "pilot_config": asdict(config),
        "truth_indices": truth_indices,
        "summary": {arm: _summary(items) for arm, items in traces.items()},
        "paired": {"d2_minus_d1_shared": d2_vs_d1, "d2_minus_d1_call_matched_width": d2_vs_width},
        "mechanics": mechanics,
        "candidate_requests": provider.physical_requests,
        "invalid_candidate_responses": provider.invalid_responses,
        "traces": {arm: [_serialize_trace(trace) for trace in items] for arm, items in traces.items()},
    }
    if config.exploratory_only:
        summary["promotion"] = {
            "directional_final_entropy_gain_against_both_controls": directional_gain,
            "promotable": mechanics_pass and directional_gain,
            "criterion": "positive paired final-entropy reduction for d2 against both controls plus mechanics checks",
        }
    else:
        ci_excludes_zero = all(
            comparison["final_entropy_reduction_ci95_descriptive"][0] > 0.0
            for comparison in (d2_vs_d1, d2_vs_width)
        )
        summary["confirmation"] = {
            "final_entropy_ci95_excludes_zero_against_both_controls": ci_excludes_zero,
            "confirmed": mechanics_pass and ci_excludes_zero,
            "criterion": "positive lower bound of both paired 95% final-entropy bootstrap intervals plus mechanics checks",
        }
    return summary


def render_report(summary: dict[str, Any]) -> str:
    exploratory = bool(summary["exploratory_only"])
    ci_label = "Descriptive 95% bootstrap CI" if exploratory else "95% bootstrap CI"
    lines = [
        "# Rock Diagnosis LLM Candidate-Proposal Pilot", "",
        (
            "**Exploratory only.**" if exploratory else "**Pre-registered confirmatory run.**"
        ) + " The LLM proposes only legal action IDs. Rock dynamics, observations, exact posterior updates, EIG values, action selection, and full-vector MAP decoding are programmatic.", "",
        "| Arm | Entropy AUC | Final entropy | Final MAP accuracy | Final truth log p | Mean pool | Logical calls / decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("d1_shared", "d2", "d1_call_matched_width"):
        row = summary["summary"][arm]
        lines.append(
            f"| {arm} | {row['entropy_auc_mean']:.4f} | {row['final_entropy_mean']:.4f} | "
            f"{row['final_map_accuracy_mean']:.4f} | {row['final_truth_log_probability_mean']:.4f} | "
            f"{row['mean_unique_pool_size']:.2f} | {row['mean_logical_candidate_calls_per_decision']:.2f} |"
        )
    lines.extend(["", "## Paired Final Entropy", "", "Positive values favor depth two.", "",
                  f"| Comparison | Mean entropy reduction | {ci_label} | W / T / L |",
                  "| --- | ---: | --- | --- |"])
    for label, row in summary["paired"].items():
        ci = row["final_entropy_reduction_ci95_descriptive"]
        wins = row["wins_ties_losses"]
        lines.append(
            f"| {label} | {row['final_entropy_reduction_mean']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{wins[0]} / {wins[1]} / {wins[2]} |"
        )
    mechanics = summary["mechanics"]
    lines.extend([
        "", "## Mechanics", "",
        f"- Terminal candidate-cell failures: `{mechanics['terminal_candidate_cell_failures']}`.",
        f"- Raw rejected candidate attempts: `{mechanics['raw_rejected_candidate_attempts']}`.",
        f"- All selected actions legal: `{mechanics['all_selected_actions_legal']}`.",
        f"- Initial candidate cells shared: `{mechanics['all_initial_candidate_cells_shared']}`.",
        f"- Width call allocation matches virtual d2: `{mechanics['width_call_allocation_matches_virtual_depth_two']}`.",
        f"- Candidate calls: `{mechanics['physical_candidate_requests']}` physical / "
        f"`{mechanics['logical_candidate_requests']}` logical; cache hits `{mechanics['cache_hits']}`.",
        "", "## Decision", "",
    ])
    if exploratory:
        lines.extend([
            f"- Directional d2 final-entropy gain versus both controls: `{summary['promotion']['directional_final_entropy_gain_against_both_controls']}`.",
            f"- **Promotable to one preregistered confirmatory run: `{summary['promotion']['promotable']}`.**",
            "- This is an eight-task exploratory screen. Its intervals are descriptive, not confirmatory evidence.", "",
        ])
    else:
        lines.extend([
            f"- Paired final-entropy 95% intervals exclude zero against both controls: `{summary['confirmation']['final_entropy_ci95_excludes_zero_against_both_controls']}`.",
            f"- **Confirmed under the preregistered criterion: `{summary['confirmation']['confirmed']}`.**", "",
        ])
    return "\n".join(lines)


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = getattr(model, "usage_snapshot", None)
    return snapshot() if callable(snapshot) else {"backend": "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config_nonmyopic_rock_diagnosis_pilot_openrouter.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/nonmyopic/rock_diagnosis_llm_pilot"))
    parser.add_argument("--run-id", default="nonmyopic-rock-diagnosis-llm-pilot-20260715")
    parser.add_argument("--map-name", default="3-6")
    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--candidate-width", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2304)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--candidate-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="use a deterministic no-spend proposer")
    parser.add_argument("--confirmatory", action="store_true", help="enable the preregistered powered run mode")
    args = parser.parse_args()

    config = PilotConfig(
        map_name=args.map_name, num_trials=args.num_trials, num_rounds=args.num_rounds,
        candidate_width=args.candidate_width, seed=args.seed, bootstrap_replicates=args.bootstrap_replicates,
        temperature=args.temperature, candidate_retries=args.candidate_retries,
        exploratory_only=not args.confirmatory,
    )
    config.validate()
    environment = RockDiagnosisModel(
        get_paper_map(config.map_name), half_efficiency_distance=config.half_efficiency_distance
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        model: ChatModel = DeterministicCandidateModel(config.candidate_width)
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        if not runtime_config.model_pairs:
            raise ValueError("pilot config requires a questioner model pair")
        model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = LLMCandidateProvider(model, environment, config)
    try:
        summary = run_pilot(provider, config)
    except CandidateProposalError as exc:
        failure = {
            "schema_version": 1, "exploratory_only": config.exploratory_only, "status": "failed_closed", "error": str(exc),
            "pilot_config": asdict(config), "candidate_requests": provider.physical_requests,
            "invalid_candidate_responses": provider.invalid_responses, "usage": _usage_snapshot(model),
        }
        (args.output_dir / "PILOT_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    summary["usage"] = _usage_snapshot(model)
    summary["run_id"] = args.run_id
    summary["dry_run"] = args.dry_run
    (args.output_dir / "PILOT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "PILOT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary.get("promotion", summary.get("confirmation")), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
