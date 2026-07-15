"""Bounded LLM candidate-proposal pilot for exact dynamic location BED.

The LLM proposes only legal local grid-action IDs. The signal model, likelihood,
posterior, EIG score, action feasibility, and posterior-mean decode are exact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.constrained_oracle_check import (
    OracleConfig,
    _entropy,
    _sample_source_configs,
    _signal,
    _update_belief,
)


QUADRATURE_ZS = np.asarray([-1.0, 0.0, 1.0], dtype=float)
QUADRATURE_WEIGHTS = np.asarray([1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0], dtype=float)


class CandidateProposalError(RuntimeError):
    """A candidate cell could not be made legal within its fixed retry budget."""


class ChatModel(Protocol):
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]: ...


@dataclass(frozen=True)
class PilotConfig:
    num_trials: int = 6
    num_rounds: int = 6
    candidate_width: int = 2
    planning_support_size: int = 2
    num_particles: int = 40
    grid_size: int = 11
    arena: float = 2.5
    max_step_radius: float = 0.5
    source_radius: float = 2.2
    noise_sd: float = 0.15
    signal_lengthscale: float = 0.5
    signal_amplitude: float = 8.0
    seed: int = 1304
    bootstrap_replicates: int = 10_000
    temperature: float = 0.7
    candidate_retries: int = 1

    def validate(self) -> None:
        if not 1 <= self.num_trials <= 10:
            raise ValueError("num_trials must be in [1, 10] for exploration")
        if self.num_rounds < 2:
            raise ValueError("num_rounds must include the fixed origin plus one selected action")
        if self.candidate_width <= 0:
            raise ValueError("candidate_width must be positive")
        if self.planning_support_size <= 0:
            raise ValueError("planning_support_size must be positive")
        if self.num_particles <= 0 or self.grid_size < 3 or self.grid_size % 2 == 0:
            raise ValueError("num_particles must be positive and grid_size an odd integer at least 3")
        if self.max_step_radius <= 0.0 or self.noise_sd <= 0.0:
            raise ValueError("max_step_radius and noise_sd must be positive")
        if self.bootstrap_replicates <= 0 or self.candidate_retries < 0:
            raise ValueError("bootstrap_replicates must be positive and candidate_retries non-negative")


@dataclass(frozen=True)
class CandidatePool:
    trial_index: int
    state_key: str
    label: str
    action_ids: tuple[str, ...]
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
    action_id: str
    action: tuple[float, float]
    selected_eig: float
    selection_score: float
    rmse: float
    entropy: float
    candidate_pool: tuple[str, ...]
    base_candidate_pool: tuple[str, ...]
    candidate_call_budget: int
    virtual_depth_two_call_budget: int
    fixed_origin: bool


@dataclass(frozen=True)
class PolicyTrace:
    arm: str
    trial_index: int
    hidden_source: tuple[float, float]
    steps: tuple[StepTrace, ...]
    logical_candidate_calls: int
    candidate_cache_hits: int

    @property
    def rmse(self) -> tuple[float, ...]:
        return tuple(step.rmse for step in self.steps)

    @property
    def entropy(self) -> tuple[float, ...]:
        return tuple(step.entropy for step in self.steps)


def _oracle_config(config: PilotConfig) -> OracleConfig:
    return OracleConfig(
        num_trials=config.num_trials,
        num_rounds=config.num_rounds,
        num_particles=config.num_particles,
        grid_size=config.grid_size,
        arena=config.arena,
        max_step_radius=config.max_step_radius,
        noise_sd=config.noise_sd,
        planner_depth=2,
        planning_support_size=config.planning_support_size,
        source_prior="branch_decoy",
        source_radius=config.source_radius,
        seed=config.seed,
        num_sources=1,
        fixed_first_query=True,
        signal_model="local_bump",
        signal_lengthscale=config.signal_lengthscale,
        signal_amplitude=config.signal_amplitude,
    )


def _grid(config: PilotConfig) -> np.ndarray:
    axis = np.linspace(-config.arena, config.arena, config.grid_size)
    return np.asarray([(float(x), float(y)) for x in axis for y in axis], dtype=float)


def _origin_index(grid: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(grid, axis=1)))


def _action_id(index: int) -> str:
    return f"q_{index}"


def _feasible_indices(grid: np.ndarray, previous: int | None, max_step_radius: float) -> tuple[int, ...]:
    if previous is None:
        return tuple(range(len(grid)))
    distances = np.linalg.norm(grid - grid[previous][None, :], axis=1)
    indices = np.flatnonzero(distances <= max_step_radius + 1e-12)
    if len(indices):
        return tuple(int(index) for index in indices)
    return (int(np.argmin(distances)),)


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
    response: str,
    *,
    allowed_actions: tuple[int, ...],
    expected_count: int,
) -> tuple[int, ...]:
    try:
        payload = json.loads(_normalize_json_response(response))
    except json.JSONDecodeError as exc:
        raise CandidateProposalError("response is not a JSON object") from exc
    if not isinstance(payload, dict) or set(payload) != {"action_ids"}:
        raise CandidateProposalError("response must be exactly {'action_ids': [...]}")
    ids = payload["action_ids"]
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise CandidateProposalError("action_ids must be a JSON list of strings")
    if len(ids) != expected_count:
        raise CandidateProposalError(f"expected exactly {expected_count} action IDs")
    if len(set(ids)) != len(ids):
        raise CandidateProposalError("action IDs must be distinct")
    lookup = {_action_id(action): action for action in allowed_actions}
    unknown = [item for item in ids if item not in lookup]
    if unknown:
        raise CandidateProposalError(f"unknown or infeasible action IDs: {unknown}")
    return tuple(lookup[item] for item in ids)


def _state_key(probabilities: np.ndarray, previous_action: int | None) -> str:
    rounded = np.round(probabilities, decimals=7).astype(np.float64).tobytes()
    encoded = (str(previous_action) + ":").encode("ascii") + rounded
    return hashlib.sha256(encoded).hexdigest()[:20]


class LLMCandidateProvider:
    """Strict candidate cells cached by exact trial state and request label."""

    def __init__(self, model: ChatModel, grid: np.ndarray, config: PilotConfig) -> None:
        self.model = model
        self.grid = grid
        self.config = config
        self._cache: dict[tuple[int, str, str, tuple[int, ...]], CandidatePool] = {}
        self.physical_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []

    def _messages(
        self,
        *,
        probabilities: np.ndarray,
        particles: np.ndarray,
        previous_action: int | None,
        legal_actions: tuple[int, ...],
        label: str,
        avoid_actions: tuple[int, ...],
    ) -> list[dict[str, str]]:
        top = np.argsort(probabilities)[-min(4, len(probabilities)) :][::-1]
        belief_lines = [
            f"- ({particles[index, 0, 0]:+.2f}, {particles[index, 0, 1]:+.2f}), p={probabilities[index]:.3f}"
            for index in top
        ]
        location = "origin before the fixed first query" if previous_action is None else (
            f"({self.grid[previous_action, 0]:+.2f}, {self.grid[previous_action, 1]:+.2f})"
        )
        legal_lines = [
            f"{_action_id(action)}=({self.grid[action, 0]:+.2f}, {self.grid[action, 1]:+.2f})"
            for action in legal_actions
        ]
        system = (
            "You propose legal next sensor locations for a bounded location-finding experiment. "
            "Return exactly one JSON object and no prose: {\"action_ids\":[\"q_1\",\"q_2\"]}."
        )
        lines = [
            f"Candidate cell: {label}.",
            f"Current sensor location: {location}.",
            f"Choose exactly {self.config.candidate_width} distinct legal action IDs.",
            "A separate exact program scores information gain and chooses one candidate.",
            "Top exact posterior source hypotheses:",
            *belief_lines,
            "Legal next action IDs:",
            "; ".join(legal_lines),
        ]
        if avoid_actions:
            lines.extend(
                [
                    "For this width-expansion cell, avoid re-proposing these IDs whenever possible:",
                    ", ".join(_action_id(action) for action in avoid_actions),
                ]
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]

    def propose(
        self,
        *,
        trial_index: int,
        probabilities: np.ndarray,
        particles: np.ndarray,
        previous_action: int | None,
        legal_actions: tuple[int, ...],
        label: str,
        avoid_actions: tuple[int, ...] = (),
    ) -> CandidatePool:
        if len(legal_actions) < self.config.candidate_width:
            raise CandidateProposalError("fewer legal actions than required candidate width")
        key_state = _state_key(probabilities, previous_action)
        key = (trial_index, key_state, label, avoid_actions)
        cached = self._cache.get(key)
        if cached is not None:
            return replace(cached, cache_hit=True)
        messages = self._messages(
            probabilities=probabilities,
            particles=particles,
            previous_action=previous_action,
            legal_actions=legal_actions,
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
                actions = parse_candidate_action_ids(
                    response, allowed_actions=legal_actions, expected_count=self.config.candidate_width
                )
            except CandidateProposalError as exc:
                last_error = exc
                self.invalid_responses.append(
                    {
                        "trial_index": trial_index,
                        "state_key": key_state,
                        "label": label,
                        "attempt": attempt,
                        "error": str(exc),
                        "raw_response": response,
                    }
                )
                continue
            pool = CandidatePool(
                trial_index=trial_index,
                state_key=key_state,
                label=label,
                action_ids=tuple(_action_id(action) for action in actions),
                actions=actions,
                raw_response=response,
                cache_hit=False,
            )
            self._cache[key] = pool
            self.physical_requests.append(
                {
                    "trial_index": trial_index,
                    "state_key": key_state,
                    "label": label,
                    "attempt": attempt,
                    "action_ids": list(pool.action_ids),
                    "raw_response": response,
                    "legal_action_count": len(legal_actions),
                }
            )
            return pool
        raise CandidateProposalError(
            f"candidate cell trial={trial_index}, state={key_state}, label={label} failed after "
            f"{self.config.candidate_retries + 1} attempts: {last_error}"
        )


class DeterministicCandidateModel:
    """Offline legal-action model used only to exercise mechanics without spending."""

    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("dry-run model accepts one response")
        content = messages[-1]["content"]
        legal = content.split("Legal next action IDs:\n", maxsplit=1)[1].split("\n", maxsplit=1)[0]
        ids = [item.split("=", maxsplit=1)[0] for item in legal.split("; ")]
        avoid: list[str] = []
        marker = "For this width-expansion cell, avoid re-proposing these IDs whenever possible:\n"
        if marker in content:
            avoid = content.split(marker, maxsplit=1)[1].split("\n", maxsplit=1)[0].split(", ")
        chosen = [item for item in ids if item not in avoid][:2]
        if len(chosen) < 2:
            chosen.extend(item for item in ids if item not in chosen)
        return [json.dumps({"action_ids": chosen[:2]})]

    def usage_snapshot(self) -> dict[str, Any]:
        return {"backend": "dry_run", "requests": 0, "cost_usd": 0.0}


class PromptRandomLegalCandidateModel:
    """Deterministic prompt-conditioned random legal proposer for zero-cost controls."""

    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        if num_responses != 1:
            raise ValueError("random-legal model accepts one response")
        content = messages[-1]["content"]
        legal = content.split("Legal next action IDs:\n", maxsplit=1)[1].split("\n", maxsplit=1)[0]
        ids = [item.split("=", maxsplit=1)[0] for item in legal.split("; ")]
        avoid: list[str] = []
        marker = "For this width-expansion cell, avoid re-proposing these IDs whenever possible:\n"
        if marker in content:
            avoid = content.split(marker, maxsplit=1)[1].split("\n", maxsplit=1)[0].split(", ")
        eligible = [item for item in ids if item not in avoid]
        if len(eligible) < 2:
            eligible = ids
        seed = int.from_bytes(hashlib.sha256(content.encode("utf-8")).digest()[:8], "little")
        selected = np.random.default_rng(seed).choice(eligible, size=2, replace=False).tolist()
        return [json.dumps({"action_ids": selected})]

    def usage_snapshot(self) -> dict[str, Any]:
        return {"backend": "random_legal_gate", "requests": 0, "cost_usd": 0.0}


def _request_pool(
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    *,
    trial_index: int,
    probabilities: np.ndarray,
    particles: np.ndarray,
    previous_action: int | None,
    grid: np.ndarray,
    config: PilotConfig,
    label: str,
    avoid_actions: tuple[int, ...] = (),
) -> CandidatePool:
    counters.logical_candidate_calls += 1
    pool = provider.propose(
        trial_index=trial_index,
        probabilities=probabilities,
        particles=particles,
        previous_action=previous_action,
        legal_actions=_feasible_indices(grid, previous_action, config.max_step_radius),
        label=label,
        avoid_actions=avoid_actions,
    )
    if pool.cache_hit:
        counters.cache_hits += 1
    return pool


def _planning_indices(probabilities: np.ndarray, limit: int) -> np.ndarray:
    support = np.flatnonzero(probabilities > 1e-6)
    if len(support) <= limit:
        return support
    ranked = support[np.argsort(probabilities[support])[-limit:]]
    return np.sort(ranked)


def _expected_eig(
    particles: np.ndarray, probabilities: np.ndarray, action: np.ndarray, oracle: OracleConfig, config: PilotConfig
) -> float:
    del config
    current_entropy = _entropy(probabilities)
    expected = 0.0
    for index in np.flatnonzero(probabilities > 1e-12):
        probability = float(probabilities[index])
        if probability <= 0.0:
            continue
        mean = float(_signal(particles[index : index + 1], action, oracle)[0])
        for z, weight in zip(QUADRATURE_ZS, QUADRATURE_WEIGHTS):
            value = mean * math.exp(oracle.noise_sd * float(z))
            next_probabilities = _update_belief(particles, probabilities, action, value, oracle.noise_sd, oracle)
            expected += probability * float(weight) * (current_entropy - _entropy(next_probabilities))
    return expected


def _choose(pool: tuple[int, ...], scores: dict[int, float]) -> int:
    return max(pool, key=lambda action: (scores[action], -action))


def _one_step_selection(
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    particles: np.ndarray,
    probabilities: np.ndarray,
    previous_action: int | None,
    grid: np.ndarray,
    oracle: OracleConfig,
    config: PilotConfig,
    trial_index: int,
) -> Selection:
    before = counters.logical_candidate_calls
    base = _request_pool(
        provider, counters, trial_index=trial_index, probabilities=probabilities, particles=particles,
        previous_action=previous_action, grid=grid, config=config, label="root"
    )
    scores = {action: _expected_eig(particles, probabilities, grid[action], oracle, config) for action in base.actions}
    return Selection(
        action=_choose(base.actions, scores), candidate_pool=base.actions, base_candidate_pool=base.actions,
        immediate_scores=scores, total_scores=scores,
        candidate_call_budget=counters.logical_candidate_calls - before, virtual_depth_two_call_budget=1,
    )


def _depth_two_selection(
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    particles: np.ndarray,
    probabilities: np.ndarray,
    previous_action: int | None,
    grid: np.ndarray,
    oracle: OracleConfig,
    config: PilotConfig,
    trial_index: int,
    remaining_rounds: int,
) -> Selection:
    before = counters.logical_candidate_calls
    base = _request_pool(
        provider, counters, trial_index=trial_index, probabilities=probabilities, particles=particles,
        previous_action=previous_action, grid=grid, config=config, label="root"
    )
    immediate = {action: _expected_eig(particles, probabilities, grid[action], oracle, config) for action in base.actions}
    if remaining_rounds == 1:
        return Selection(
            action=_choose(base.actions, immediate), candidate_pool=base.actions, base_candidate_pool=base.actions,
            immediate_scores=immediate, total_scores=immediate,
            candidate_call_budget=counters.logical_candidate_calls - before, virtual_depth_two_call_budget=1,
        )

    support_indices = _planning_indices(probabilities, config.planning_support_size)
    scores: dict[int, float] = {}
    for root_action in base.actions:
        continuation = 0.0
        for index in support_indices:
            probability = float(probabilities[index])
            if probability <= 0.0:
                continue
            branch_mean = float(_signal(particles[index : index + 1], grid[root_action], oracle)[0])
            for noise_index, (z, weight) in enumerate(zip(QUADRATURE_ZS, QUADRATURE_WEIGHTS)):
                value = branch_mean * math.exp(oracle.noise_sd * float(z))
                next_probabilities = _update_belief(
                    particles, probabilities, grid[root_action], value, oracle.noise_sd, oracle
                )
                future = _request_pool(
                    provider,
                    counters,
                    trial_index=trial_index,
                    probabilities=next_probabilities,
                    particles=particles,
                    previous_action=root_action,
                    grid=grid,
                    config=config,
                    label=f"future:{root_action}:{int(index)}:{noise_index}",
                )
                future_scores = [
                    _expected_eig(particles, next_probabilities, grid[action], oracle, config)
                    for action in future.actions
                ]
                continuation += probability * float(weight) * max(future_scores)
        scores[root_action] = immediate[root_action] + continuation
    budget = counters.logical_candidate_calls - before
    expected_budget = 1 + len(base.actions) * len(support_indices) * len(QUADRATURE_ZS)
    if budget != expected_budget:
        raise AssertionError("depth-two candidate allocation drifted")
    return Selection(
        action=_choose(base.actions, scores), candidate_pool=base.actions, base_candidate_pool=base.actions,
        immediate_scores=immediate, total_scores=scores,
        candidate_call_budget=budget, virtual_depth_two_call_budget=expected_budget,
    )


def _matched_width_selection(
    provider: LLMCandidateProvider,
    counters: ArmCounters,
    particles: np.ndarray,
    probabilities: np.ndarray,
    previous_action: int | None,
    grid: np.ndarray,
    oracle: OracleConfig,
    config: PilotConfig,
    trial_index: int,
    remaining_rounds: int,
) -> Selection:
    before = counters.logical_candidate_calls
    base = _request_pool(
        provider, counters, trial_index=trial_index, probabilities=probabilities, particles=particles,
        previous_action=previous_action, grid=grid, config=config, label="root"
    )
    virtual_budget = 1
    if remaining_rounds > 1:
        virtual_budget += len(base.actions) * len(_planning_indices(probabilities, config.planning_support_size)) * len(QUADRATURE_ZS)
    pools = [base.actions]
    seen = set(base.actions)
    for sample_index in range(1, virtual_budget):
        extra = _request_pool(
            provider, counters, trial_index=trial_index, probabilities=probabilities, particles=particles,
            previous_action=previous_action, grid=grid, config=config, label=f"width:{sample_index}",
            avoid_actions=tuple(sorted(seen)),
        )
        pools.append(extra.actions)
        seen.update(extra.actions)
    candidates = tuple(dict.fromkeys(action for pool in pools for action in pool))
    scores = {action: _expected_eig(particles, probabilities, grid[action], oracle, config) for action in candidates}
    budget = counters.logical_candidate_calls - before
    if budget != virtual_budget:
        raise AssertionError("matched-width candidate allocation drifted")
    return Selection(
        action=_choose(candidates, scores), candidate_pool=candidates, base_candidate_pool=base.actions,
        immediate_scores=scores, total_scores=scores,
        candidate_call_budget=budget, virtual_depth_two_call_budget=virtual_budget,
    )


def _decode_rmse(particles: np.ndarray, probabilities: np.ndarray, hidden_state: np.ndarray) -> float:
    estimate = np.sum(particles[:, 0, :] * probabilities[:, None], axis=0)
    return float(np.linalg.norm(estimate - hidden_state[0]))


def run_policy(
    provider: LLMCandidateProvider,
    config: PilotConfig,
    *,
    arm: str,
    trial_index: int,
    hidden_state: np.ndarray,
    initial_particles: np.ndarray,
    noise_zs: np.ndarray,
) -> PolicyTrace:
    if arm not in {"d1_shared", "d2", "d1_matched_width"}:
        raise ValueError(f"unknown arm: {arm}")
    oracle = _oracle_config(config)
    grid = _grid(config)
    particles = np.concatenate([initial_particles, hidden_state[None, :, :]], axis=0)
    probabilities = np.full(len(particles), 1.0 / len(particles), dtype=float)
    previous_action: int | None = None
    counters = ArmCounters()
    steps: list[StepTrace] = []
    for round_index in range(config.num_rounds):
        if round_index == 0:
            action = _origin_index(grid)
            selection = Selection(
                action=action, candidate_pool=(), base_candidate_pool=(), immediate_scores={action: 0.0},
                total_scores={action: 0.0}, candidate_call_budget=0, virtual_depth_two_call_budget=0,
            )
            fixed_origin = True
        else:
            remaining_rounds = config.num_rounds - round_index
            fixed_origin = False
            if arm == "d1_shared":
                selection = _one_step_selection(
                    provider, counters, particles, probabilities, previous_action, grid, oracle, config, trial_index
                )
            elif arm == "d2":
                selection = _depth_two_selection(
                    provider, counters, particles, probabilities, previous_action, grid, oracle, config,
                    trial_index, remaining_rounds
                )
            else:
                selection = _matched_width_selection(
                    provider, counters, particles, probabilities, previous_action, grid, oracle, config,
                    trial_index, remaining_rounds
                )
            action = selection.action
        legal = _feasible_indices(grid, previous_action, config.max_step_radius)
        if action not in legal:
            raise AssertionError("selected an infeasible dynamic action")
        mean = float(_signal(hidden_state[None, :, :], grid[action], oracle)[0])
        value = mean * math.exp(oracle.noise_sd * float(noise_zs[round_index]))
        probabilities = _update_belief(particles, probabilities, grid[action], value, oracle.noise_sd, oracle)
        steps.append(
            StepTrace(
                action_id=_action_id(action), action=(float(grid[action, 0]), float(grid[action, 1])),
                selected_eig=selection.immediate_scores[action], selection_score=selection.total_scores[action],
                rmse=_decode_rmse(particles, probabilities, hidden_state), entropy=_entropy(probabilities),
                candidate_pool=tuple(_action_id(item) for item in selection.candidate_pool),
                base_candidate_pool=tuple(_action_id(item) for item in selection.base_candidate_pool),
                candidate_call_budget=selection.candidate_call_budget,
                virtual_depth_two_call_budget=selection.virtual_depth_two_call_budget,
                fixed_origin=fixed_origin,
            )
        )
        previous_action = action
    return PolicyTrace(
        arm=arm, trial_index=trial_index,
        hidden_source=(float(hidden_state[0, 0]), float(hidden_state[0, 1])),
        steps=tuple(steps), logical_candidate_calls=counters.logical_candidate_calls, candidate_cache_hits=counters.cache_hits,
    )


def _bootstrap_mean_ci(values: np.ndarray, *, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 512):
        batch = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        samples[start : start + batch] = np.mean(values[indices], axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _summary(traces: list[PolicyTrace]) -> dict[str, Any]:
    rmse = np.asarray([trace.rmse for trace in traces], dtype=float)
    entropy = np.asarray([trace.entropy for trace in traces], dtype=float)
    pools = np.asarray([[len(step.candidate_pool) for step in trace.steps[1:]] for trace in traces], dtype=float)
    calls = np.asarray([[step.candidate_call_budget for step in trace.steps[1:]] for trace in traces], dtype=float)
    return {
        "rmse_auc_mean": float(np.mean(rmse)),
        "final_rmse_mean": float(np.mean(rmse[:, -1])),
        "final_entropy_mean": float(np.mean(entropy[:, -1])),
        "round_rmse_mean": [float(item) for item in np.mean(rmse, axis=0)],
        "round_entropy_mean": [float(item) for item in np.mean(entropy, axis=0)],
        "mean_unique_pool_size": float(np.mean(pools)),
        "mean_logical_candidate_calls_per_decision": float(np.mean(calls)),
        "logical_candidate_calls": int(sum(trace.logical_candidate_calls for trace in traces)),
    }


def _paired(
    first: list[PolicyTrace], second: list[PolicyTrace], *, config: PilotConfig, label: str
) -> dict[str, Any]:
    first_rmse = np.asarray([trace.rmse for trace in first], dtype=float)
    second_rmse = np.asarray([trace.rmse for trace in second], dtype=float)
    auc_delta = np.mean(first_rmse, axis=1) - np.mean(second_rmse, axis=1)
    seed = int.from_bytes(f"{config.seed}:{label}".encode("utf-8"), "little", signed=False) % (2**63 - 1)
    ci = _bootstrap_mean_ci(auc_delta, replicates=config.bootstrap_replicates, seed=seed)
    return {
        "rmse_auc_delta_mean": float(np.mean(auc_delta)),
        "rmse_auc_delta_ci95_descriptive": [ci[0], ci[1]],
        "final_rmse_delta_mean": float(np.mean(first_rmse[:, -1] - second_rmse[:, -1])),
        "round_rmse_delta_mean": [float(item) for item in np.mean(first_rmse - second_rmse, axis=0)],
        "wins_ties_losses": [
            int(np.count_nonzero(auc_delta < 0.0)),
            int(np.count_nonzero(auc_delta == 0.0)),
            int(np.count_nonzero(auc_delta > 0.0)),
        ],
    }


def _serialize_trace(trace: PolicyTrace) -> dict[str, Any]:
    return {
        "arm": trace.arm, "trial_index": trace.trial_index, "hidden_source": list(trace.hidden_source),
        "logical_candidate_calls": trace.logical_candidate_calls, "candidate_cache_hits": trace.candidate_cache_hits,
        "steps": [asdict(step) for step in trace.steps],
    }


def run_pilot(provider: LLMCandidateProvider, config: PilotConfig) -> dict[str, Any]:
    config.validate()
    oracle = _oracle_config(config)
    rng = np.random.default_rng(config.seed)
    traces: dict[str, list[PolicyTrace]] = {"d1_shared": [], "d2": [], "d1_matched_width": []}
    roots_shared = True
    width_matches = True
    for trial_index in range(config.num_trials):
        hidden = _sample_source_configs(rng, 1, 1, prior="branch_decoy", radius=config.source_radius)[0]
        initial = _sample_source_configs(rng, config.num_particles, 1, prior="branch_decoy", radius=config.source_radius)
        noise_zs = rng.normal(size=config.num_rounds)
        for arm in ("d1_shared", "d2", "d1_matched_width"):
            traces[arm].append(
                run_policy(provider, config, arm=arm, trial_index=trial_index, hidden_state=hidden, initial_particles=initial, noise_zs=noise_zs)
            )
        root_pools = [traces[arm][-1].steps[1].base_candidate_pool for arm in traces]
        roots_shared = roots_shared and len(set(root_pools)) == 1
        width_matches = width_matches and all(
            step.candidate_call_budget == step.virtual_depth_two_call_budget
            for step in traces["d1_matched_width"][-1].steps[1:]
        )
    d2_vs_d1 = _paired(traces["d2"], traces["d1_shared"], config=config, label="d2-d1")
    d2_vs_width = _paired(traces["d2"], traces["d1_matched_width"], config=config, label="d2-width")
    mechanics = {
        "terminal_candidate_cell_failures": 0,
        "all_selected_actions_legal": True,
        "all_initial_candidate_cells_shared": roots_shared,
        "width_call_allocation_matches_virtual_depth_two": width_matches,
        "raw_rejected_candidate_attempts": len(provider.invalid_responses),
        "physical_candidate_requests": len(provider.physical_requests),
        "logical_candidate_requests": int(sum(trace.logical_candidate_calls for arm in traces.values() for trace in arm)),
        "cache_hits": int(sum(trace.candidate_cache_hits for arm in traces.values() for trace in arm)),
    }
    directional = d2_vs_d1["rmse_auc_delta_mean"] < 0.0 and d2_vs_width["rmse_auc_delta_mean"] < 0.0
    promotable = all(
        (
            mechanics["all_selected_actions_legal"], mechanics["all_initial_candidate_cells_shared"],
            mechanics["width_call_allocation_matches_virtual_depth_two"], directional,
        )
    )
    return {
        "schema_version": 1, "exploratory_only": True, "pilot_config": asdict(config),
        "oracle_geometry": asdict(oracle), "summary": {arm: _summary(items) for arm, items in traces.items()},
        "paired": {"d2_minus_d1_shared": d2_vs_d1, "d2_minus_d1_matched_width": d2_vs_width},
        "mechanics": mechanics,
        "promotion": {
            "directional_rmse_auc_gain_against_both_controls": directional, "promotable": promotable,
            "criterion": "lower paired RMSE AUC for d2 versus both controls plus terminal mechanics",
        },
        "candidate_requests": provider.physical_requests, "invalid_candidate_responses": provider.invalid_responses,
        "traces": {arm: [_serialize_trace(trace) for trace in items] for arm, items in traces.items()},
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Dynamic Location LLM Candidate-Proposal Pilot", "",
        "**Exploratory only.** The LLM proposes legal local grid IDs; all inference, EIG, feasibility, and posterior-mean decoding are exact.", "",
        "| Arm | RMSE AUC | Final RMSE | Final entropy | Mean unique pool | Logical calls / selected decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("d1_shared", "d2", "d1_matched_width"):
        row = summary["summary"][arm]
        lines.append(
            f"| {arm} | {row['rmse_auc_mean']:.4f} | {row['final_rmse_mean']:.4f} | "
            f"{row['final_entropy_mean']:.4f} | {row['mean_unique_pool_size']:.2f} | "
            f"{row['mean_logical_candidate_calls_per_decision']:.2f} |"
        )
    lines.extend(["", "## Paired RMSE AUC", "", "| Comparison | Mean delta (d2 - control) | Descriptive 95% bootstrap CI | W / T / L |", "| --- | ---: | --- | --- |"])
    for label, row in summary["paired"].items():
        ci = row["rmse_auc_delta_ci95_descriptive"]
        wins = row["wins_ties_losses"]
        lines.append(f"| {label} | {row['rmse_auc_delta_mean']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {wins[0]} / {wins[1]} / {wins[2]} |")
    mechanics = summary["mechanics"]
    lines.extend([
        "", "## Mechanics", "",
        f"- Terminal candidate-cell failures: `{mechanics['terminal_candidate_cell_failures']}`.",
        f"- All selected actions legal: `{mechanics['all_selected_actions_legal']}`.",
        f"- Initial candidate cells shared: `{mechanics['all_initial_candidate_cells_shared']}`.",
        f"- Width call allocation matches virtual d2: `{mechanics['width_call_allocation_matches_virtual_depth_two']}`.",
        f"- Raw rejected attempts: `{mechanics['raw_rejected_candidate_attempts']}`; candidate requests `{mechanics['physical_candidate_requests']}` physical / `{mechanics['logical_candidate_requests']}` logical.",
        "", "## Decision", "",
        f"- Directional d2 RMSE-AUC gain versus both controls: `{summary['promotion']['directional_rmse_auc_gain_against_both_controls']}`.",
        f"- **Promotable: `{summary['promotion']['promotable']}`.**",
        "- This is a six-trial exploratory screen; its intervals are descriptive only.", "",
    ])
    return "\n".join(lines)


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = getattr(model, "usage_snapshot", None)
    return snapshot() if callable(snapshot) else {"backend": "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config_nonmyopic_dynamic_location_pilot_openrouter.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/nonmyopic/dynamic_location_llm_pilot"))
    parser.add_argument("--run-id", default="nonmyopic-dynamic-location-pilot-20260715")
    parser.add_argument("--num-trials", type=int, default=6)
    parser.add_argument("--num-rounds", type=int, default=6)
    parser.add_argument("--candidate-width", type=int, default=2)
    parser.add_argument("--planning-support-size", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=1304)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--random-legal", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.random_legal:
        parser.error("--dry-run and --random-legal are mutually exclusive")
    config = PilotConfig(
        num_trials=args.num_trials, num_rounds=args.num_rounds, candidate_width=args.candidate_width,
        planning_support_size=args.planning_support_size, grid_size=args.grid_size, seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    config.validate()
    grid = _grid(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        model: ChatModel = DeterministicCandidateModel()
    elif args.random_legal:
        model = PromptRandomLegalCandidateModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = LLMCandidateProvider(model, grid, config)
    try:
        summary = run_pilot(provider, config)
    except CandidateProposalError as exc:
        failure = {
            "schema_version": 1, "exploratory_only": True, "status": "failed_closed", "error": str(exc),
            "pilot_config": asdict(config), "candidate_requests": provider.physical_requests,
            "invalid_candidate_responses": provider.invalid_responses, "usage": _usage_snapshot(model),
        }
        (args.output_dir / "PILOT_FAILURE.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    summary["usage"] = _usage_snapshot(model)
    summary["run_id"] = args.run_id
    summary["dry_run"] = args.dry_run
    (args.output_dir / "PILOT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "PILOT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary["promotion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
