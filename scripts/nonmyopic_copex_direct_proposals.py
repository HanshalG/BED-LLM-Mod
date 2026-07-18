"""Paired depth-by-proposal experiment on the continuous COPEx task.

The LLM has one narrow job: propose a small set of legal *next sensor moves*.
All likelihood evaluation, posterior updates, counterfactual observations, and
depth-two selection are programmatic.  This keeps proposal quality distinct
from the non-myopic objective while preserving the finite-support COPEx model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
from typing import Any, Literal, Protocol

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.location_finding.continuous_strategy import (
    copex_signal,
    particle_entropy,
    update_copex_belief,
)
from helpers import Config, load_config
from model_factory import build_model_adapter


ArmName = Literal["llm_d1", "llm_d2", "llm_width", "grid_d1", "grid_d2"]
ARMS: tuple[ArmName, ...] = ("llm_d1", "llm_d2", "llm_width", "grid_d1", "grid_d2")


class DirectProposalError(RuntimeError):
    """A direct action-proposal cell failed its bounded repair policy."""


class ChatModel(Protocol):
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]: ...


@dataclass(frozen=True)
class DirectProposalConfig:
    num_trials: int = 8
    num_rounds: int = 8
    num_particles: int = 48
    candidate_width: int = 3
    outer_rollouts: int = 4
    child_rollouts: int = 8
    max_step: float = 0.1
    noise_sd: float = 0.5
    grid_resolution: int = 12
    seed: int = 46_021
    bootstrap_replicates: int = 10_000
    temperature: float = 0.0
    validation_retries: int = 1
    trial_concurrency: int = 16

    def validate(self) -> None:
        counts = (
            self.num_trials,
            self.num_rounds,
            self.num_particles,
            self.candidate_width,
            self.outer_rollouts,
            self.child_rollouts,
            self.grid_resolution,
            self.bootstrap_replicates,
            self.trial_concurrency,
        )
        if min(counts) <= 0:
            raise ValueError("all direct-proposal counts must be positive")
        if self.num_rounds < 2 or self.candidate_width < 2:
            raise ValueError("the factorial requires at least two rounds and two candidates")
        if self.grid_resolution < self.candidate_width:
            raise ValueError("grid_resolution must be at least candidate_width")
        if not 0.0 < self.max_step <= 1.0 or self.noise_sd <= 0.0:
            raise ValueError("max_step and noise_sd must be positive")
        if self.validation_retries != 1:
            raise ValueError("the registered interface permits one validation retry")


@dataclass(frozen=True)
class ProposalCell:
    actions: tuple[tuple[float, float], ...]
    raw_response: str
    cache_hit: bool


@dataclass
class PolicyState:
    probabilities: np.ndarray
    position: np.ndarray
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Selection:
    action: tuple[float, float]
    score: float
    candidate_actions: tuple[tuple[float, float], ...]
    candidate_scores: tuple[float, ...]
    logical_llm_calls: int
    virtual_depth_two_calls: int


def _stable_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=list).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _state_key(position: np.ndarray, probabilities: np.ndarray) -> str:
    payload = np.round(np.concatenate([position, probabilities]), 8).astype(np.float64).tobytes()
    return hashlib.sha256(payload).hexdigest()[:24]


def _canonical_action(action: np.ndarray | tuple[float, float]) -> tuple[float, float]:
    vector = np.asarray(action, dtype=float)
    return (float(vector[0]), float(vector[1]))


def _action_key(action: np.ndarray | tuple[float, float]) -> tuple[float, float]:
    """Stable deduplication key without perturbing the executed action."""
    vector = np.asarray(action, dtype=float)
    return (round(float(vector[0]), 10), round(float(vector[1]), 10))


def _parse_moves(
    response: str,
    *,
    expected_count: int,
    position: np.ndarray,
    max_step: float,
) -> tuple[tuple[float, float], ...]:
    normalized = response.strip()
    if normalized.startswith("```json\n"):
        if not normalized.endswith("\n```"):
            raise DirectProposalError("response has an incomplete JSON fence")
        normalized = normalized[len("```json\n") : -len("\n```")]
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise DirectProposalError("response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"moves"}:
        raise DirectProposalError("response must contain exactly the moves key")
    moves = payload["moves"]
    if not isinstance(moves, list) or len(moves) != expected_count:
        raise DirectProposalError(f"expected exactly {expected_count} moves")

    actions: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    position = np.asarray(position, dtype=float)
    for index, item in enumerate(moves):
        if not isinstance(item, dict) or set(item) != {"dx", "dy"}:
            raise DirectProposalError(f"move {index} requires exactly dx and dy")
        dx, dy = item["dx"], item["dy"]
        if not isinstance(dx, (int, float)) or not isinstance(dy, (int, float)):
            raise DirectProposalError(f"move {index} coordinates must be numeric")
        delta = np.asarray([float(dx), float(dy)], dtype=float)
        if not np.all(np.isfinite(delta)):
            raise DirectProposalError(f"move {index} coordinates must be finite")
        if float(np.max(np.abs(delta))) > max_step + 1e-12 or float(np.max(np.abs(delta))) <= 1e-12:
            raise DirectProposalError(f"move {index} must be nonzero with L-infinity step <= {max_step}")
        action = position + delta
        if not bool(np.all(action >= -1e-12) and np.all(action <= 1.0 + 1e-12)):
            raise DirectProposalError(f"move {index} exits the [0,1]^2 box")
        canonical = _action_key(action)
        if canonical in seen:
            raise DirectProposalError("moves must induce distinct next locations")
        seen.add(canonical)
        actions.append(_canonical_action(action))
    return tuple(actions)


class DirectProposalProvider:
    """Strict, state-cached LLM proposal cells for legal continuous moves."""

    def __init__(self, model: ChatModel, config: DirectProposalConfig) -> None:
        self.model = model
        self.config = config
        self._cache: dict[tuple[int, str, str], ProposalCell] = {}
        self._lock = threading.Lock()
        self.accepted_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []
        self.logical_calls = 0
        self.cache_hits = 0

    @staticmethod
    def _belief_lines(particles: np.ndarray, probabilities: np.ndarray) -> list[str]:
        ranking = np.argsort(-probabilities, kind="stable")[:4]
        return [
            f"rank {rank}: ({particles[index, 0]:.4f}, {particles[index, 1]:.4f}), p={probabilities[index]:.5f}"
            for rank, index in enumerate(ranking)
        ]

    def _messages(
        self,
        *,
        position: np.ndarray,
        particles: np.ndarray,
        probabilities: np.ndarray,
        avoid_actions: tuple[tuple[float, float], ...],
    ) -> list[dict[str, str]]:
        x_lo, y_lo = -float(position[0]), -float(position[1])
        x_hi, y_hi = 1.0 - float(position[0]), 1.0 - float(position[1])
        lines = [
            "Task: propose legal next sensor moves for an exact Bayesian location experiment.",
            "A source lies in [0,1]^2. A query at x observes log(0.1 + (1e-4 + ||x-theta||^2)^-1) plus Gaussian noise sd 0.5.",
            f"Current sensor location: ({position[0]:.5f}, {position[1]:.5f}).",
            f"Return exactly {self.config.candidate_width} distinct moves as JSON only:",
            '{"moves":[{"dx":0.04,"dy":-0.10},{"dx":-0.10,"dy":0.02},...]}.',
            f"Every move must be nonzero, satisfy max(abs(dx), abs(dy)) <= {self.config.max_step}, and end inside [0,1]^2.",
            f"Therefore dx must be in [{x_lo:.5f}, {x_hi:.5f}] and dy in [{y_lo:.5f}, {y_hi:.5f}].",
            "A separate exact program scores these moves. Propose geometrically diverse locations that distinguish the leading posterior hypotheses.",
            "Leading exact posterior particles:",
            *self._belief_lines(particles, probabilities),
        ]
        if avoid_actions:
            formatted = "; ".join(f"({x:.4f},{y:.4f})" for x, y in avoid_actions[-24:])
            lines.extend(["For this extra current-state proposal cell, avoid these already proposed endpoints whenever possible:", formatted])
        return [
            {
                "role": "system",
                "content": "Return one valid JSON move cell and no prose. Do not estimate information scores.",
            },
            {"role": "user", "content": "\n".join(lines)},
        ]

    def propose(
        self,
        *,
        trial_index: int,
        position: np.ndarray,
        particles: np.ndarray,
        probabilities: np.ndarray,
        label: str,
        avoid_actions: tuple[tuple[float, float], ...] = (),
    ) -> ProposalCell:
        state_key = _state_key(position, probabilities)
        key = (trial_index, state_key, label)
        with self._lock:
            self.logical_calls += 1
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
        if cached is not None:
            return ProposalCell(cached.actions, cached.raw_response, True)

        messages = self._messages(
            position=position, particles=particles, probabilities=probabilities, avoid_actions=avoid_actions
        )
        last_error: Exception | None = None
        for attempt in range(self.config.validation_retries + 1):
            responses = self.model.chat_complete(messages, self.config.temperature, num_responses=1)
            if len(responses) != 1:
                raise DirectProposalError("model did not return exactly one response")
            response = responses[0]
            try:
                actions = _parse_moves(
                    response,
                    expected_count=self.config.candidate_width,
                    position=position,
                    max_step=self.config.max_step,
                )
            except DirectProposalError as exc:
                last_error = exc
                with self._lock:
                    self.invalid_responses.append(
                        {
                            "trial_index": trial_index,
                            "state_key": state_key,
                            "label": label,
                            "attempt": attempt,
                            "error": str(exc),
                            "raw_response": response,
                        }
                    )
                if attempt < self.config.validation_retries:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": f"Invalid: {exc}. Return the complete corrected JSON cell only."},
                    ]
                continue
            with self._lock:
                self.accepted_requests.append(
                    {
                        "trial_index": trial_index,
                        "state_key": state_key,
                        "label": label,
                        "attempt": attempt,
                        "raw_response": response,
                    }
                )
                cell = ProposalCell(actions, response, False)
                self._cache[key] = cell
            return cell
        raise DirectProposalError(f"cell failed after two attempts: {last_error}")


class DeterministicDirectProposalModel:
    """No-spend proposal emitter used only for mechanics and test coverage."""

    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature, num_responses
        content = messages[-1]["content"]
        marker = "Return exactly "
        count = int(content.split(marker, 1)[1].split(" distinct moves", 1)[0])
        position_line = next(line for line in content.splitlines() if line.startswith("Current sensor location:"))
        coordinates = position_line.split("(", 1)[1].split(")", 1)[0].split(",")
        position = np.asarray([float(coordinates[0]), float(coordinates[1])])
        actions: list[tuple[float, float]] = []
        for angle in np.linspace(0.0, 2.0 * math.pi, num=16, endpoint=False):
            delta = np.asarray([0.1 * math.cos(angle), 0.1 * math.sin(angle)])
            candidate = position + delta
            if np.all(candidate >= 0.0) and np.all(candidate <= 1.0):
                actions.append((float(delta[0]), float(delta[1])))
            if len(actions) == count:
                break
        if len(actions) != count:
            raise AssertionError("dry model could not find enough legal moves")
        return [json.dumps({"moves": [{"dx": x, "dy": y} for x, y in actions]})]

    def usage_snapshot(self) -> dict[str, Any]:
        return {"backend": "dry_run", "requests": 0, "cost_usd": 0.0}


def _draws(
    config: DirectProposalConfig,
    *,
    trial_index: int,
    round_index: int,
    state: PolicyState,
    label: str,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(
        _stable_seed(config.seed, trial_index, round_index, _state_key(state.position, state.probabilities), label)
    )
    return rng.random(count), rng.normal(size=count)


def _immediate_eig(
    action: tuple[float, float],
    *,
    position: np.ndarray,
    particles: np.ndarray,
    probabilities: np.ndarray,
    uniforms: np.ndarray,
    noise_zs: np.ndarray,
    config: DirectProposalConfig,
) -> float:
    del position
    if len(uniforms) != len(noise_zs):
        raise ValueError("uniforms and noise_zs must share a length")
    start_entropy = particle_entropy(probabilities)
    cumulative = np.cumsum(probabilities)
    truth_indices = np.minimum(np.searchsorted(cumulative, uniforms, side="right"), len(probabilities) - 1)
    drops = []
    query = np.asarray(action, dtype=float)
    for truth_index, noise_z in zip(truth_indices, noise_zs, strict=True):
        observation = copex_signal(particles[int(truth_index)], query) + config.noise_sd * float(noise_z)
        posterior = update_copex_belief(particles, probabilities, query, observation, noise_sd=config.noise_sd)
        drops.append(start_entropy - particle_entropy(posterior))
    return float(np.mean(drops))


def _grid_actions(position: np.ndarray, config: DirectProposalConfig) -> tuple[tuple[float, float], ...]:
    actions: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for angle in np.linspace(0.0, 2.0 * math.pi, num=config.grid_resolution, endpoint=False):
        delta = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
        delta *= config.max_step / max(abs(float(delta[0])), abs(float(delta[1])))
        action = np.clip(position + delta, 0.0, 1.0)
        if float(np.max(np.abs(action - position))) <= 1e-12:
            continue
        key = _action_key(action)
        if key not in seen:
            seen.add(key)
            actions.append(_canonical_action(action))
        if len(actions) == config.candidate_width:
            break
    if len(actions) != config.candidate_width:
        raise AssertionError("grid could not provide the required number of legal actions")
    return tuple(actions)


def _choose(actions: tuple[tuple[float, float], ...], scores: tuple[float, ...], calls: int, virtual: int) -> Selection:
    index = max(range(len(scores)), key=lambda item: (scores[item], -item))
    return Selection(
        action=actions[index],
        score=scores[index],
        candidate_actions=actions,
        candidate_scores=scores,
        logical_llm_calls=calls,
        virtual_depth_two_calls=virtual,
    )


def _d1_selection(
    actions: tuple[tuple[float, float], ...],
    *,
    state: PolicyState,
    particles: np.ndarray,
    uniforms: np.ndarray,
    noise_zs: np.ndarray,
    config: DirectProposalConfig,
    calls: int,
    virtual: int,
) -> Selection:
    scores = tuple(
        _immediate_eig(
            action,
            position=state.position,
            particles=particles,
            probabilities=state.probabilities,
            uniforms=uniforms,
            noise_zs=noise_zs,
            config=config,
        )
        for action in actions
    )
    return _choose(actions, scores, calls, virtual)


def _d2_llm_selection(
    provider: DirectProposalProvider,
    *,
    state: PolicyState,
    particles: np.ndarray,
    config: DirectProposalConfig,
    trial_index: int,
    round_index: int,
) -> Selection:
    root = provider.propose(
        trial_index=trial_index,
        position=state.position,
        particles=particles,
        probabilities=state.probabilities,
        label="root",
    )
    uniforms, root_noise = _draws(
        config, trial_index=trial_index, round_index=round_index, state=state, label="outer", count=config.outer_rollouts
    )
    cumulative = np.cumsum(state.probabilities)
    truth_indices = np.minimum(np.searchsorted(cumulative, uniforms, side="right"), len(particles) - 1)
    start_entropy = particle_entropy(state.probabilities)
    scores: list[float] = []
    for root_index, action in enumerate(root.actions):
        total = 0.0
        query = np.asarray(action, dtype=float)
        for rollout_index, (truth_index, noise_z) in enumerate(zip(truth_indices, root_noise, strict=True)):
            observation = copex_signal(particles[int(truth_index)], query) + config.noise_sd * float(noise_z)
            branch_probabilities = update_copex_belief(
                particles, state.probabilities, query, observation, noise_sd=config.noise_sd
            )
            branch_state = PolicyState(branch_probabilities, query)
            child = provider.propose(
                trial_index=trial_index,
                position=query,
                particles=particles,
                probabilities=branch_probabilities,
                label=f"child:{round_index}:{root_index}:{rollout_index}",
            )
            child_uniforms, child_noise = _draws(
                config,
                trial_index=trial_index,
                round_index=round_index,
                state=branch_state,
                label=f"child-score:{root_index}:{rollout_index}",
                count=config.child_rollouts,
            )
            child_scores = tuple(
                _immediate_eig(
                    child_action,
                    position=query,
                    particles=particles,
                    probabilities=branch_probabilities,
                    uniforms=child_uniforms,
                    noise_zs=child_noise,
                    config=config,
                )
                for child_action in child.actions
            )
            total += start_entropy - particle_entropy(branch_probabilities) + max(child_scores)
        scores.append(total / config.outer_rollouts)
    virtual = 1 + config.candidate_width * config.outer_rollouts
    return _choose(root.actions, tuple(scores), virtual, virtual)


def _d2_grid_selection(
    *,
    state: PolicyState,
    particles: np.ndarray,
    config: DirectProposalConfig,
    trial_index: int,
    round_index: int,
) -> Selection:
    root_actions = _grid_actions(state.position, config)
    uniforms, root_noise = _draws(
        config, trial_index=trial_index, round_index=round_index, state=state, label="outer", count=config.outer_rollouts
    )
    cumulative = np.cumsum(state.probabilities)
    truth_indices = np.minimum(np.searchsorted(cumulative, uniforms, side="right"), len(particles) - 1)
    start_entropy = particle_entropy(state.probabilities)
    scores: list[float] = []
    for root_index, action in enumerate(root_actions):
        total = 0.0
        query = np.asarray(action, dtype=float)
        for rollout_index, (truth_index, noise_z) in enumerate(zip(truth_indices, root_noise, strict=True)):
            observation = copex_signal(particles[int(truth_index)], query) + config.noise_sd * float(noise_z)
            branch_probabilities = update_copex_belief(
                particles, state.probabilities, query, observation, noise_sd=config.noise_sd
            )
            branch_state = PolicyState(branch_probabilities, query)
            child_uniforms, child_noise = _draws(
                config,
                trial_index=trial_index,
                round_index=round_index,
                state=branch_state,
                label=f"grid-child-score:{root_index}:{rollout_index}",
                count=config.child_rollouts,
            )
            child_scores = tuple(
                _immediate_eig(
                    child_action,
                    position=query,
                    particles=particles,
                    probabilities=branch_probabilities,
                    uniforms=child_uniforms,
                    noise_zs=child_noise,
                    config=config,
                )
                for child_action in _grid_actions(query, config)
            )
            total += start_entropy - particle_entropy(branch_probabilities) + max(child_scores)
        scores.append(total / config.outer_rollouts)
    return _choose(root_actions, tuple(scores), 0, 0)


def _width_selection(
    provider: DirectProposalProvider,
    *,
    state: PolicyState,
    particles: np.ndarray,
    config: DirectProposalConfig,
    trial_index: int,
    round_index: int,
) -> Selection:
    virtual = 1 if round_index == config.num_rounds - 1 else 1 + config.candidate_width * config.outer_rollouts
    seen: list[tuple[float, float]] = []
    for cell_index in range(virtual):
        cell = provider.propose(
            trial_index=trial_index,
            position=state.position,
            particles=particles,
            probabilities=state.probabilities,
            label="root" if cell_index == 0 else f"width:{round_index}:{cell_index}",
            avoid_actions=tuple(seen),
        )
        seen.extend(action for action in cell.actions if action not in seen)
    actions = tuple(seen)
    uniforms, noise_zs = _draws(
        config, trial_index=trial_index, round_index=round_index, state=state, label="outer", count=config.outer_rollouts
    )
    return _d1_selection(
        actions,
        state=state,
        particles=particles,
        uniforms=uniforms,
        noise_zs=noise_zs,
        config=config,
        calls=virtual,
        virtual=virtual,
    )


def _select(
    arm: ArmName,
    provider: DirectProposalProvider,
    *,
    state: PolicyState,
    particles: np.ndarray,
    config: DirectProposalConfig,
    trial_index: int,
    round_index: int,
) -> Selection:
    if arm == "llm_d2":
        if round_index == config.num_rounds - 1:
            root = provider.propose(
                trial_index=trial_index,
                position=state.position,
                particles=particles,
                probabilities=state.probabilities,
                label="root",
            )
            uniforms, noise_zs = _draws(
                config, trial_index=trial_index, round_index=round_index, state=state,
                label="outer", count=config.outer_rollouts,
            )
            return _d1_selection(
                root.actions, state=state, particles=particles, uniforms=uniforms, noise_zs=noise_zs,
                config=config, calls=1, virtual=1,
            )
        return _d2_llm_selection(
            provider, state=state, particles=particles, config=config, trial_index=trial_index, round_index=round_index
        )
    if arm == "grid_d2":
        if round_index == config.num_rounds - 1:
            uniforms, noise_zs = _draws(
                config, trial_index=trial_index, round_index=round_index, state=state,
                label="outer", count=config.outer_rollouts,
            )
            return _d1_selection(
                _grid_actions(state.position, config), state=state, particles=particles,
                uniforms=uniforms, noise_zs=noise_zs, config=config, calls=0, virtual=0,
            )
        return _d2_grid_selection(
            state=state, particles=particles, config=config, trial_index=trial_index, round_index=round_index
        )
    if arm == "llm_width":
        return _width_selection(
            provider, state=state, particles=particles, config=config, trial_index=trial_index, round_index=round_index
        )
    uniforms, noise_zs = _draws(
        config, trial_index=trial_index, round_index=round_index, state=state, label="outer", count=config.outer_rollouts
    )
    if arm == "llm_d1":
        root = provider.propose(
            trial_index=trial_index,
            position=state.position,
            particles=particles,
            probabilities=state.probabilities,
            label="root",
        )
        return _d1_selection(
            root.actions, state=state, particles=particles, uniforms=uniforms, noise_zs=noise_zs,
            config=config, calls=1, virtual=1 + config.candidate_width * config.outer_rollouts,
        )
    return _d1_selection(
        _grid_actions(state.position, config), state=state, particles=particles, uniforms=uniforms, noise_zs=noise_zs,
        config=config, calls=0, virtual=0,
    )


def _run_trial(provider: DirectProposalProvider, config: DirectProposalConfig, trial_index: int) -> dict[str, Any]:
    rng = np.random.default_rng(_stable_seed(config.seed, "trial", trial_index))
    truth = rng.uniform(0.0, 1.0, size=2)
    particles = np.concatenate([rng.uniform(0.0, 1.0, size=(config.num_particles, 2)), truth[None, :]], axis=0)
    initial = rng.uniform(0.0, 1.0, size=2)
    states = {arm: PolicyState(np.full(len(particles), 1.0 / len(particles)), initial.copy()) for arm in ARMS}
    initial_root_shared = True
    width_calls_match = True
    legal = True
    for round_index in range(config.num_rounds):
        selections = {
            arm: _select(
                arm, provider, state=states[arm], particles=particles, config=config,
                trial_index=trial_index, round_index=round_index,
            )
            for arm in ARMS
        }
        if round_index == 0:
            initial_root_shared = selections["llm_d1"].candidate_actions == selections["llm_d2"].candidate_actions
        width_calls_match = width_calls_match and (
            selections["llm_width"].logical_llm_calls == selections["llm_d2"].virtual_depth_two_calls
        )
        actual_z = float(np.random.default_rng(_stable_seed(config.seed, trial_index, round_index, "actual-z")).normal())
        for arm, selection in selections.items():
            state = states[arm]
            action = np.asarray(selection.action, dtype=float)
            legal = legal and bool(np.all(action >= 0.0) and np.all(action <= 1.0))
            legal = legal and float(np.max(np.abs(action - state.position))) <= config.max_step + 1e-12
            observation = copex_signal(truth, action) + config.noise_sd * actual_z
            state.probabilities = update_copex_belief(
                particles, state.probabilities, action, observation, noise_sd=config.noise_sd
            )
            state.position = action
            estimate = np.sum(particles * state.probabilities[:, None], axis=0)
            state.steps.append(
                {
                    "round": round_index + 1,
                    "action": action.tolist(),
                    "observation": observation,
                    "entropy": particle_entropy(state.probabilities),
                    "truth_log_probability": float(math.log(max(float(state.probabilities[-1]), 1e-300))),
                    "rmse": float(np.linalg.norm(estimate - truth)),
                    "planning_score": selection.score,
                    "candidate_scores": list(selection.candidate_scores),
                    "candidate_actions": [list(item) for item in selection.candidate_actions],
                    "logical_llm_calls": selection.logical_llm_calls,
                    "virtual_depth_two_calls": selection.virtual_depth_two_calls,
                }
            )
    return {
        "trial_index": trial_index,
        "truth": truth.tolist(),
        "initial_position": initial.tolist(),
        "traces": {arm: states[arm].steps for arm in ARMS},
        "initial_root_cell_shared": initial_root_shared,
        "width_calls_match": width_calls_match,
        "all_actions_legal": legal,
    }


def _bootstrap_ci(values: np.ndarray, config: DirectProposalConfig, label: str) -> tuple[float, float]:
    rng = np.random.default_rng(_stable_seed(config.seed, "bootstrap", label))
    draws = rng.integers(0, len(values), size=(config.bootstrap_replicates, len(values)))
    means = np.mean(values[draws], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _arm_summary(trials: list[dict[str, Any]], arm: ArmName) -> dict[str, Any]:
    entropy = np.asarray([[step["entropy"] for step in trial["traces"][arm]] for trial in trials])
    truth_log = np.asarray([[step["truth_log_probability"] for step in trial["traces"][arm]] for trial in trials])
    rmse = np.asarray([[step["rmse"] for step in trial["traces"][arm]] for trial in trials])
    calls = np.asarray([[step["logical_llm_calls"] for step in trial["traces"][arm]] for trial in trials])
    return {
        "entropy_mean_trace": np.mean(entropy, axis=0).tolist(),
        "entropy_auc_mean": float(np.mean(np.mean(entropy, axis=1))),
        "final_entropy_mean": float(np.mean(entropy[:, -1])),
        "truth_log_probability_mean_trace": np.mean(truth_log, axis=0).tolist(),
        "truth_log_probability_auc_mean": float(np.mean(np.mean(truth_log, axis=1))),
        "final_truth_log_probability_mean": float(np.mean(truth_log[:, -1])),
        "rmse_mean_trace": np.mean(rmse, axis=0).tolist(),
        "final_rmse_mean": float(np.mean(rmse[:, -1])),
        "mean_logical_llm_calls": float(np.mean(calls)),
    }


def _paired_metric(
    trials: list[dict[str, Any]], *, first: ArmName, second: ArmName, key: str, config: DirectProposalConfig, label: str
) -> dict[str, Any]:
    first_values = np.asarray([
        np.mean([step[key] for step in trial["traces"][first]]) for trial in trials
    ])
    second_values = np.asarray([
        np.mean([step[key] for step in trial["traces"][second]]) for trial in trials
    ])
    gains = second_values - first_values if key == "entropy" else first_values - second_values
    ci = _bootstrap_ci(gains, config, label)
    return {
        "mean": float(np.mean(gains)),
        "ci95": list(ci),
        "wins_ties_losses": [
            int(np.count_nonzero(gains > 0.0)),
            int(np.count_nonzero(gains == 0.0)),
            int(np.count_nonzero(gains < 0.0)),
        ],
        "paired_values": gains.tolist(),
    }


def run_factorial(provider: DirectProposalProvider, config: DirectProposalConfig) -> dict[str, Any]:
    config.validate()
    workers = min(config.trial_concurrency, config.num_trials)
    if workers == 1:
        trials = [_run_trial(provider, config, index) for index in range(config.num_trials)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_trial, provider, config, index) for index in range(config.num_trials)]
            trials = [future.result() for future in futures]
    comparisons = {
        "llm_d2_minus_llm_d1": _paired_metric(trials, first="llm_d2", second="llm_d1", key="entropy", config=config, label="llm-d2-d1"),
        "llm_d2_minus_llm_width": _paired_metric(trials, first="llm_d2", second="llm_width", key="entropy", config=config, label="llm-d2-width"),
        "grid_d2_minus_grid_d1": _paired_metric(trials, first="grid_d2", second="grid_d1", key="entropy", config=config, label="grid-d2-d1"),
        "llm_d2_minus_grid_d2": _paired_metric(trials, first="llm_d2", second="grid_d2", key="entropy", config=config, label="llm-d2-grid-d2"),
        "llm_d2_minus_llm_d1_truth_log_probability": _paired_metric(trials, first="llm_d2", second="llm_d1", key="truth_log_probability", config=config, label="llm-d2-d1-truth"),
    }
    llm_depth = np.asarray(comparisons["llm_d2_minus_llm_d1"]["paired_values"])
    grid_depth = np.asarray(comparisons["grid_d2_minus_grid_d1"]["paired_values"])
    interaction = llm_depth - grid_depth
    comparisons["llm_depth_minus_grid_depth_interaction"] = {
        "mean": float(np.mean(interaction)),
        "ci95": list(_bootstrap_ci(interaction, config, "interaction")),
        "paired_values": interaction.tolist(),
    }
    mechanics = {
        "terminal_cell_failures": 0,
        "all_actions_legal": all(trial["all_actions_legal"] for trial in trials),
        "initial_root_cell_shared": all(trial["initial_root_cell_shared"] for trial in trials),
        "width_call_allocation_matches_virtual_depth_two": all(trial["width_calls_match"] for trial in trials),
        "physical_llm_requests": len(provider.accepted_requests) + len(provider.invalid_responses),
        "accepted_llm_cells": len(provider.accepted_requests),
        "raw_rejected_responses": len(provider.invalid_responses),
        "logical_llm_calls": provider.logical_calls,
        "cache_hits": provider.cache_hits,
        "inner_llm_calls_used_only_for_action_proposals": True,
    }
    return {
        "schema_version": 1,
        "stage": "COPEx_direct_proposal_factorial",
        "config": asdict(config),
        "summary": {arm: _arm_summary(trials, arm) for arm in ARMS},
        "comparisons": comparisons,
        "mechanics": mechanics,
        "requests": provider.accepted_requests,
        "invalid_responses": provider.invalid_responses,
        "trials": trials,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# COPEx Direct-Proposal Depth x Proposal Factorial",
        "",
        "The LLM proposes only legal next sensor locations. Likelihoods, finite-support posterior updates, counterfactual observations, and all scoring are programmatic.",
        "",
        "| Arm | Entropy AUC | Final entropy | Truth-log-posterior AUC | Final RMSE | Mean LLM proposal cells / decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = summary["summary"][arm]
        lines.append(
            f"| {arm} | {row['entropy_auc_mean']:.4f} | {row['final_entropy_mean']:.4f} | "
            f"{row['truth_log_probability_auc_mean']:.4f} | {row['final_rmse_mean']:.4f} | {row['mean_logical_llm_calls']:.1f} |"
        )
    lines.extend(["", "| Comparison (positive favors LLM d2 / depth) | Entropy-AUC gain | 95% paired bootstrap CI | W / T / L |", "| --- | ---: | --- | --- |"])
    for label, row in summary["comparisons"].items():
        if "wins_ties_losses" not in row:
            continue
        lo, hi = row["ci95"]
        wtl = row["wins_ties_losses"]
        lines.append(f"| {label} | {row['mean']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {wtl[0]} / {wtl[1]} / {wtl[2]} |")
    interaction = summary["comparisons"]["llm_depth_minus_grid_depth_interaction"]
    lines.extend([
        "",
        f"Depth-by-proposal interaction: `{interaction['mean']:+.4f}` nats entropy-AUC, 95% CI [{interaction['ci95'][0]:+.4f}, {interaction['ci95'][1]:+.4f}].",
        "",
        "## Mechanics",
        "",
        *(f"- {key}: `{value}`." for key, value in summary["mechanics"].items()),
        "",
    ])
    return "\n".join(lines)


def _usage(model: Any) -> dict[str, Any]:
    snapshot = getattr(model, "usage_snapshot", None)
    return snapshot() if callable(snapshot) else {"backend": "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config_nonmyopic_copex_direct_proposals_openrouter.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/nonmyopic/copex_direct_proposals"))
    parser.add_argument("--run-id", default="copex-direct-proposals")
    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--num-particles", type=int, default=48)
    parser.add_argument("--candidate-width", type=int, default=3)
    parser.add_argument("--outer-rollouts", type=int, default=4)
    parser.add_argument("--child-rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=46_021)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--trial-concurrency", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = DirectProposalConfig(
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        num_particles=args.num_particles,
        candidate_width=args.candidate_width,
        outer_rollouts=args.outer_rollouts,
        child_rollouts=args.child_rollouts,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        trial_concurrency=args.trial_concurrency,
    )
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        model: ChatModel = DeterministicDirectProposalModel()
    else:
        runtime: Config = load_config(args.config)
        runtime.run_id = args.run_id
        model = build_model_adapter(runtime.model_pairs[0].questioner, config=runtime)
    provider = DirectProposalProvider(model, config)
    try:
        summary = run_factorial(provider, config)
    except DirectProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "COPEx_direct_proposal_factorial",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(config),
            "requests": provider.accepted_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage(model),
        }
        (args.output_dir / "FACTORIAL_FAILURE.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise
    summary["usage"] = _usage(model)
    summary["run_id"] = args.run_id
    summary["dry_run"] = args.dry_run
    (args.output_dir / "FACTORIAL.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "FACTORIAL.md").write_text(render_report(summary))
    print(json.dumps({"comparisons": summary["comparisons"], "mechanics": summary["mechanics"]}, indent=2))


if __name__ == "__main__":
    main()
