"""LLM strategy-prior evaluation on the COPEx constrained location task."""

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
    ContinuousStrategy,
    ContinuousStrategyParseError,
    ContinuousStrategyScore,
    ContinuousStrategyStep,
    copex_signal,
    execute_continuous_step,
    parse_continuous_strategy_cell,
    particle_entropy,
    random_continuous_strategy,
    score_continuous_strategy,
    update_copex_belief,
)
from helpers import Config, load_config
from model_factory import build_model_adapter


ArmName = Literal["strategy_eig", "shared_d1", "width", "random_strategy", "grid_d2"]
ARMS: tuple[ArmName, ...] = (
    "strategy_eig",
    "shared_d1",
    "width",
    "random_strategy",
    "grid_d2",
)


class ContinuousProposalError(RuntimeError):
    """A complete strategy or width cell failed its bounded repair policy."""


class ChatModel(Protocol):
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]: ...


@dataclass(frozen=True)
class L3Config:
    num_trials: int = 20
    num_rounds: int = 30
    num_particles: int = 64
    num_strategies: int = 4
    planning_horizon: int = 4
    rollout_samples: int = 64
    max_step: float = 0.1
    noise_sd: float = 0.5
    grid_resolution: int = 8
    seed: int = 31_001
    bootstrap_replicates: int = 10_000
    temperature: float = 0.0
    validation_retries: int = 1
    trial_concurrency: int = 24

    def validate(self) -> None:
        counts = (
            self.num_trials,
            self.num_rounds,
            self.num_particles,
            self.num_strategies,
            self.planning_horizon,
            self.rollout_samples,
            self.grid_resolution,
            self.bootstrap_replicates,
            self.trial_concurrency,
        )
        if min(counts) <= 0:
            raise ValueError("all L3 counts must be positive")
        if self.num_strategies < 2 or self.planning_horizon < 2:
            raise ValueError("L3 requires at least two strategies and horizon at least two")
        if self.grid_resolution < 2:
            raise ValueError("grid_resolution must be at least two")
        if not 0.0 < self.max_step <= 1.0 or self.noise_sd <= 0.0:
            raise ValueError("max_step and noise_sd must be positive")
        if self.validation_retries != 1:
            raise ValueError("the registered interface permits one validation retry")

    @property
    def width_count(self) -> int:
        return self.num_strategies * self.planning_horizon

    @property
    def grid_sequence_budget(self) -> int:
        return max(1, self.width_count // 2)


@dataclass(frozen=True)
class StrategyCell:
    strategies: tuple[ContinuousStrategy, ...]
    raw_response: str
    cache_hit: bool


@dataclass(frozen=True)
class WidthCell:
    steps: tuple[ContinuousStrategyStep, ...]
    raw_response: str
    cache_hit: bool


@dataclass(frozen=True)
class Selection:
    action: tuple[float, float]
    planning_score: float
    candidate_scores: tuple[float, ...]
    candidates: tuple[str, ...]
    selected_strategy: str | None
    scorer_units: int
    logical_llm_calls: int


@dataclass
class PolicyState:
    probabilities: np.ndarray
    position: np.ndarray
    steps: list[dict[str, Any]] = field(default_factory=list)


def _stable_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=list).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _state_key(position: np.ndarray, probabilities: np.ndarray, horizon: int) -> str:
    payload = np.round(np.concatenate([position, probabilities]), 8).astype(np.float64).tobytes()
    return hashlib.sha256(str(horizon).encode() + payload).hexdigest()[:24]


def _vector_step(dx: float, dy: float) -> ContinuousStrategyStep:
    return ContinuousStrategyStep(kind="vector", dx=float(dx), dy=float(dy))


def _strategy_from_steps(name: str, steps: tuple[ContinuousStrategyStep, ...]) -> ContinuousStrategy:
    return ContinuousStrategy(name=name, description=name, steps=steps, canonical_json=name)


def _parse_width_cell(response: str, *, expected_count: int, max_step: float) -> tuple[ContinuousStrategyStep, ...]:
    normalized = response.strip()
    if normalized.startswith("```json\n"):
        if not normalized.endswith("\n```"):
            raise ContinuousProposalError("width response has an incomplete JSON fence")
        normalized = normalized[len("```json\n") : -len("\n```")]
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ContinuousProposalError("width response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"vectors"}:
        raise ContinuousProposalError("width response must contain exactly the vectors key")
    vectors = payload["vectors"]
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        raise ContinuousProposalError(f"expected exactly {expected_count} width vectors")
    steps: list[ContinuousStrategyStep] = []
    seen: set[tuple[float, float]] = set()
    for vector in vectors:
        if not isinstance(vector, dict) or set(vector) != {"dx", "dy"}:
            raise ContinuousProposalError("each width vector requires exactly dx and dy")
        dx, dy = vector["dx"], vector["dy"]
        if not isinstance(dx, (int, float)) or not isinstance(dy, (int, float)):
            raise ContinuousProposalError("width vector coordinates must be numeric")
        pair = (float(dx), float(dy))
        if max(abs(pair[0]), abs(pair[1])) > max_step + 1e-12 or max(abs(pair[0]), abs(pair[1])) <= 1e-12:
            raise ContinuousProposalError("width vectors must be nonzero and satisfy the max step")
        rounded = (round(pair[0], 8), round(pair[1], 8))
        if rounded in seen:
            raise ContinuousProposalError("width vectors must be distinct")
        seen.add(rounded)
        steps.append(_vector_step(*pair))
    return tuple(steps)


class ContinuousStrategyProvider:
    def __init__(self, model: ChatModel, config: L3Config) -> None:
        self.model = model
        self.config = config
        self._strategy_cache: dict[tuple[int, str], StrategyCell] = {}
        self._width_cache: dict[tuple[int, str], WidthCell] = {}
        self._lock = threading.Lock()
        self.accepted_requests: list[dict[str, Any]] = []
        self.invalid_responses: list[dict[str, Any]] = []
        self.logical_strategy_calls = 0
        self.logical_width_calls = 0
        self.cache_hits = 0

    def _belief_lines(self, particles: np.ndarray, probabilities: np.ndarray) -> list[str]:
        ranking = np.argsort(-probabilities, kind="stable")[:4]
        return [
            f"rank {rank}: ({particles[index, 0]:.4f}, {particles[index, 1]:.4f}), p={probabilities[index]:.5f}"
            for rank, index in enumerate(ranking)
        ]

    def _strategy_messages(
        self,
        position: np.ndarray,
        particles: np.ndarray,
        probabilities: np.ndarray,
        horizon: int,
    ) -> list[dict[str, str]]:
        system = (
            "Generate compact spatial plans for sequential Bayesian experimental design. Return one JSON "
            "object and no prose. A program executes every plan under the exact transition constraint and "
            "scores total information gain with the analytic likelihood; do not estimate scores."
        )
        schema = (
            '{"strategies":[{"name":"short NL name","description":"spatial rationale",'
            '"steps":[{"kind":"vector","dx":0.1,"dy":-0.05},'
            '{"kind":"toward_rank","rank":0},{"kind":"midpoint_ranks","ranks":[0,1]},'
            '{"kind":"toward_mean"}]}]}'
        )
        user = "\n".join(
            [
                "Task: one source is uniform on [0,1]^2. A query observes log(0.1 + "
                "(1e-4 + squared distance)^-1) plus Gaussian noise sd 0.5.",
                f"Current query position: ({position[0]:.4f}, {position[1]:.4f}).",
                f"Every next query must satisfy L-infinity step <= {self.config.max_step}; the executor "
                "projects macros to this constraint and [0,1]^2.",
                f"Return exactly {self.config.num_strategies} distinct strategies, each with exactly "
                f"{horizon} steps, using this schema:",
                schema,
                "Allowed step kinds only: vector (nonzero dx/dy with max absolute component <= 0.1), "
                "toward_rank (rank 0..3), midpoint_ranks (two distinct ranks 0..3), toward_mean.",
                "Rank macros are recomputed after each simulated observation, so plans can react to the "
                "updated posterior. Make plans geometrically diverse and genuinely multi-step.",
                "Leading exact posterior particles:",
                *self._belief_lines(particles, probabilities),
            ]
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _width_messages(
        self,
        position: np.ndarray,
        particles: np.ndarray,
        probabilities: np.ndarray,
        expected_count: int,
    ) -> list[dict[str, str]]:
        user = "\n".join(
            [
                f"Current query position: ({position[0]:.4f}, {position[1]:.4f}).",
                f"Return exactly {expected_count} distinct nonzero relative vectors as "
                '{"vectors":[{"dx":0.1,"dy":0.0},...]}.',
                "Each vector must have max(|dx|,|dy|) <= 0.1. Return JSON only. A program scores "
                "their exact-model immediate information gain.",
                "Leading posterior particles:",
                *self._belief_lines(particles, probabilities),
            ]
        )
        return [{"role": "system", "content": "Propose diverse one-step sensor moves."}, {"role": "user", "content": user}]

    def _complete(self, messages: list[dict[str, str]], context: dict[str, Any], parser: Any) -> tuple[Any, str]:
        last_error: Exception | None = None
        for attempt in range(self.config.validation_retries + 1):
            responses = self.model.chat_complete(messages, self.config.temperature, num_responses=1)
            if len(responses) != 1:
                raise ContinuousProposalError("model did not return exactly one response")
            response = responses[0]
            try:
                parsed = parser(response)
            except (ContinuousStrategyParseError, ContinuousProposalError) as exc:
                last_error = exc
                with self._lock:
                    self.invalid_responses.append({**context, "attempt": attempt, "error": str(exc), "raw_response": response})
                if attempt < self.config.validation_retries:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": f"Invalid: {exc}. Correct the whole JSON cell exactly."},
                    ]
                continue
            with self._lock:
                self.accepted_requests.append({**context, "attempt": attempt, "raw_response": response})
            return parsed, response
        raise ContinuousProposalError(f"cell failed after two attempts: {last_error}")

    def strategies(
        self,
        trial_index: int,
        position: np.ndarray,
        particles: np.ndarray,
        probabilities: np.ndarray,
        horizon: int,
    ) -> StrategyCell:
        state_key = _state_key(position, probabilities, horizon)
        key = (trial_index, state_key)
        with self._lock:
            self.logical_strategy_calls += 1
            cached = self._strategy_cache.get(key)
            if cached is not None:
                self.cache_hits += 1
        if cached is not None:
            return StrategyCell(cached.strategies, cached.raw_response, True)
        parsed, raw = self._complete(
            self._strategy_messages(position, particles, probabilities, horizon),
            {"type": "strategy", "trial_index": trial_index, "state_key": state_key, "horizon": horizon},
            lambda response: parse_continuous_strategy_cell(
                response,
                expected_count=self.config.num_strategies,
                horizon=horizon,
                max_step=self.config.max_step,
            ),
        )
        cell = StrategyCell(parsed, raw, False)
        with self._lock:
            self._strategy_cache[key] = cell
        return cell

    def width(
        self,
        trial_index: int,
        position: np.ndarray,
        particles: np.ndarray,
        probabilities: np.ndarray,
        expected_count: int,
    ) -> WidthCell:
        state_key = _state_key(position, probabilities, expected_count)
        key = (trial_index, state_key)
        with self._lock:
            self.logical_width_calls += 1
            cached = self._width_cache.get(key)
            if cached is not None:
                self.cache_hits += 1
        if cached is not None:
            return WidthCell(cached.steps, cached.raw_response, True)
        parsed, raw = self._complete(
            self._width_messages(position, particles, probabilities, expected_count),
            {"type": "width", "trial_index": trial_index, "state_key": state_key, "horizon": 1},
            lambda response: _parse_width_cell(
                response, expected_count=expected_count, max_step=self.config.max_step
            ),
        )
        cell = WidthCell(parsed, raw, False)
        with self._lock:
            self._width_cache[key] = cell
        return cell


class DeterministicContinuousModel:
    """Valid no-spend emitter for runner mechanics."""

    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature, num_responses
        content = messages[-1]["content"]
        if '"vectors"' in content:
            count = int(content.split("Return exactly ", 1)[1].split(" distinct", 1)[0])
            vectors = []
            for index in range(count):
                angle = 2.0 * math.pi * index / count
                scale = 0.1 / max(abs(math.cos(angle)), abs(math.sin(angle)))
                vectors.append({"dx": scale * math.cos(angle), "dy": scale * math.sin(angle)})
            return [json.dumps({"vectors": vectors})]
        count = int(content.split("Return exactly ", 1)[1].split(" distinct strategies", 1)[0])
        horizon = int(content.split("each with exactly ", 1)[1].split(" steps", 1)[0])
        kinds = ["toward_rank", "midpoint_ranks", "toward_mean", "vector"]
        strategies = []
        for index in range(count):
            steps = []
            for step_index in range(horizon):
                kind = kinds[(index + step_index) % len(kinds)]
                if kind == "toward_rank":
                    step = {"kind": kind, "rank": index % 4}
                elif kind == "midpoint_ranks":
                    step = {"kind": kind, "ranks": [index % 4, (index + 1) % 4]}
                elif kind == "vector":
                    angle = (index + step_index) * math.pi / 4.0
                    step = {"kind": kind, "dx": 0.1 * math.cos(angle), "dy": 0.1 * math.sin(angle)}
                else:
                    step = {"kind": kind}
                steps.append(step)
            strategies.append({"name": f"plan-{index}", "description": f"Diverse plan {index}.", "steps": steps})
        return [json.dumps({"strategies": strategies})]

    def usage_snapshot(self) -> dict[str, Any]:
        return {"backend": "dry_run", "requests": 0, "cost_usd": 0.0}


def _crn(config: L3Config, trial_index: int, round_index: int, state: PolicyState, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(
        _stable_seed(config.seed, trial_index, round_index, _state_key(state.position, state.probabilities, horizon), "score")
    )
    return rng.random(config.rollout_samples), rng.normal(size=(config.rollout_samples, horizon))


def _score_candidates(
    candidates: tuple[ContinuousStrategy, ...],
    state: PolicyState,
    particles: np.ndarray,
    uniforms: np.ndarray,
    noise: np.ndarray,
    config: L3Config,
) -> tuple[tuple[ContinuousStrategyScore, ...], int]:
    scores = tuple(
        score_continuous_strategy(
            strategy,
            position=state.position,
            particles=particles,
            probabilities=state.probabilities,
            truth_uniforms=uniforms,
            noise_zs=noise[:, : len(strategy.steps)],
            noise_sd=config.noise_sd,
            max_step=config.max_step,
        )
        for strategy in candidates
    )
    return scores, sum(score.scorer_units for score in scores)


def _choose(candidates: tuple[ContinuousStrategy, ...], scores: tuple[ContinuousStrategyScore, ...], calls: int) -> Selection:
    index = max(range(len(scores)), key=lambda item: (scores[item].total_eig, -item))
    return Selection(
        action=scores[index].root_action,
        planning_score=scores[index].total_eig,
        candidate_scores=tuple(score.total_eig for score in scores),
        candidates=tuple(candidate.canonical_json for candidate in candidates),
        selected_strategy=candidates[index].canonical_json,
        scorer_units=sum(score.scorer_units for score in scores),
        logical_llm_calls=calls,
    )


def _select(
    arm: ArmName,
    provider: ContinuousStrategyProvider,
    config: L3Config,
    state: PolicyState,
    particles: np.ndarray,
    trial_index: int,
    round_index: int,
    horizon: int,
) -> Selection:
    uniforms, noise = _crn(config, trial_index, round_index, state, horizon)
    if arm in {"strategy_eig", "shared_d1"}:
        cell = provider.strategies(trial_index, state.position, particles, state.probabilities, horizon)
        display_candidates = cell.strategies
        candidates = display_candidates
        if arm == "shared_d1":
            candidates = tuple(
                _strategy_from_steps(f"root:{item.name}", (item.steps[0],)) for item in candidates
            )
        scores, _units = _score_candidates(candidates, state, particles, uniforms, noise, config)
        selection = _choose(candidates, scores, 1)
        if arm == "shared_d1":
            selected_index = max(
                range(len(scores)), key=lambda item: (scores[item].total_eig, -item)
            )
            return Selection(
                action=selection.action,
                planning_score=selection.planning_score,
                candidate_scores=selection.candidate_scores,
                candidates=tuple(item.canonical_json for item in display_candidates),
                selected_strategy=display_candidates[selected_index].canonical_json,
                scorer_units=selection.scorer_units,
                logical_llm_calls=selection.logical_llm_calls,
            )
        return selection
    if arm == "random_strategy":
        rng = np.random.default_rng(_stable_seed(config.seed, trial_index, round_index, "random-strategies"))
        candidates = tuple(
            random_continuous_strategy(
                rng,
                horizon=horizon,
                max_step=config.max_step,
                index=index,
            )
            for index in range(config.num_strategies)
        )
        scores, _units = _score_candidates(candidates, state, particles, uniforms, noise, config)
        return _choose(candidates, scores, 0)
    if arm == "width":
        cell = provider.width(
            trial_index,
            state.position,
            particles,
            state.probabilities,
            config.num_strategies * horizon,
        )
        candidates = tuple(
            _strategy_from_steps(f"width:{index}", (step,)) for index, step in enumerate(cell.steps)
        )
        scores, _units = _score_candidates(candidates, state, particles, uniforms, noise, config)
        return _choose(candidates, scores, 1)

    directions = tuple(
        _vector_step(
            config.max_step * math.cos(2.0 * math.pi * index / config.grid_resolution)
            / max(abs(math.cos(2.0 * math.pi * index / config.grid_resolution)), abs(math.sin(2.0 * math.pi * index / config.grid_resolution))),
            config.max_step * math.sin(2.0 * math.pi * index / config.grid_resolution)
            / max(abs(math.cos(2.0 * math.pi * index / config.grid_resolution)), abs(math.sin(2.0 * math.pi * index / config.grid_resolution))),
        )
        for index in range(config.grid_resolution)
    )
    if horizon == 1:
        budget = min(config.num_strategies, len(directions))
        indices = np.linspace(0, len(directions) - 1, budget, dtype=int)
        candidates = tuple(
            _strategy_from_steps(f"grid-root:{index}", (directions[index],))
            for index in indices
        )
    else:
        all_pairs = [(first, second) for first in range(len(directions)) for second in range(len(directions))]
        target_units = config.num_strategies * horizon
        budget = min(max(1, target_units // 2), len(all_pairs))
        indices = np.linspace(0, len(all_pairs) - 1, budget, dtype=int)
        candidate_list = [
            _strategy_from_steps(
                f"grid-pair:{all_pairs[index][0]}:{all_pairs[index][1]}",
                (directions[all_pairs[index][0]], directions[all_pairs[index][1]]),
            )
            for index in indices
        ]
        if target_units % 2:
            candidate_list.append(_strategy_from_steps("grid-extra-root", (directions[0],)))
        candidates = tuple(candidate_list)
    scores, _units = _score_candidates(candidates, state, particles, uniforms, noise, config)
    return _choose(candidates, scores, 0)


def _run_trial(provider: ContinuousStrategyProvider, config: L3Config, trial_index: int) -> dict[str, Any]:
    rng = np.random.default_rng(_stable_seed(config.seed, "trial", trial_index))
    truth = rng.uniform(0.0, 1.0, size=2)
    particles = np.concatenate([rng.uniform(0.0, 1.0, size=(config.num_particles, 2)), truth[None, :]], axis=0)
    initial = rng.uniform(0.0, 1.0, size=2)
    states = {arm: PolicyState(np.full(len(particles), 1.0 / len(particles)), initial.copy()) for arm in ARMS}
    initial_cells_shared = True
    width_units_match = True
    grid_units_match = True
    legal = True
    for round_index in range(config.num_rounds):
        horizon = min(config.planning_horizon, config.num_rounds - round_index)
        selections = {
            arm: _select(arm, provider, config, states[arm], particles, trial_index, round_index, horizon)
            for arm in ARMS
        }
        if round_index == 0:
            initial_cells_shared = selections["strategy_eig"].candidates == selections["shared_d1"].candidates
        width_units_match = width_units_match and selections["width"].scorer_units == selections["strategy_eig"].scorer_units
        grid_units_match = grid_units_match and selections["grid_d2"].scorer_units == selections["strategy_eig"].scorer_units
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
                    "rmse": float(np.linalg.norm(estimate - truth)),
                    "planning_score": selection.planning_score,
                    "candidate_scores": list(selection.candidate_scores),
                    "candidates": list(selection.candidates),
                    "selected_strategy": selection.selected_strategy,
                    "scorer_units": selection.scorer_units,
                    "logical_llm_calls": selection.logical_llm_calls,
                }
            )
    return {
        "trial_index": trial_index,
        "truth": truth.tolist(),
        "initial_position": initial.tolist(),
        "traces": {arm: states[arm].steps for arm in ARMS},
        "initial_cells_shared": initial_cells_shared,
        "width_units_match": width_units_match,
        "grid_units_match": grid_units_match,
        "all_actions_legal": legal,
    }


def _bootstrap_ci(values: np.ndarray, config: L3Config, label: str) -> tuple[float, float]:
    rng = np.random.default_rng(_stable_seed(config.seed, "bootstrap", label))
    draws = rng.integers(0, len(values), size=(config.bootstrap_replicates, len(values)))
    means = np.mean(values[draws], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _arm_summary(trials: list[dict[str, Any]], arm: ArmName) -> dict[str, Any]:
    entropy = np.asarray([[step["entropy"] for step in trial["traces"][arm]] for trial in trials])
    rmse = np.asarray([[step["rmse"] for step in trial["traces"][arm]] for trial in trials])
    units = np.asarray([[step["scorer_units"] for step in trial["traces"][arm]] for trial in trials])
    return {
        "entropy_mean_trace": np.mean(entropy, axis=0).tolist(),
        "rmse_mean_trace": np.mean(rmse, axis=0).tolist(),
        "final_entropy_mean": float(np.mean(entropy[:, -1])),
        "final_rmse_mean": float(np.mean(rmse[:, -1])),
        "mean_scorer_units": float(np.mean(units)),
    }


def run_l3(provider: ContinuousStrategyProvider, config: L3Config) -> dict[str, Any]:
    config.validate()
    workers = min(config.trial_concurrency, config.num_trials)
    if workers == 1:
        trials = [_run_trial(provider, config, index) for index in range(config.num_trials)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_trial, provider, config, index) for index in range(config.num_trials)]
            trials = [future.result() for future in futures]
    paired: dict[str, Any] = {}
    strategy_final = np.asarray([trial["traces"]["strategy_eig"][-1]["entropy"] for trial in trials])
    for baseline in ("shared_d1", "width", "random_strategy", "grid_d2"):
        baseline_final = np.asarray([trial["traces"][baseline][-1]["entropy"] for trial in trials])
        gains = baseline_final - strategy_final
        ci = _bootstrap_ci(gains, config, baseline)
        paired[f"strategy_eig_minus_{baseline}"] = {
            "final_entropy_gain_mean": float(np.mean(gains)),
            "final_entropy_gain_ci95": list(ci),
            "wins_ties_losses": [
                int(np.count_nonzero(gains > 0)),
                int(np.count_nonzero(gains == 0)),
                int(np.count_nonzero(gains < 0)),
            ],
            "paired_values": gains.tolist(),
        }
    mechanics = {
        "terminal_cell_failures": 0,
        "all_actions_legal": all(trial["all_actions_legal"] for trial in trials),
        "initial_strategy_cells_shared_with_d1": all(trial["initial_cells_shared"] for trial in trials),
        "width_scorer_units_match_strategy_eig": all(trial["width_units_match"] for trial in trials),
        "grid_d2_scorer_units_match_strategy_eig": all(trial["grid_units_match"] for trial in trials),
        "rollout_scoring_llm_calls": 0,
        "physical_llm_requests": len(provider.accepted_requests) + len(provider.invalid_responses),
        "accepted_llm_cells": len(provider.accepted_requests),
        "raw_rejected_responses": len(provider.invalid_responses),
        "logical_strategy_calls": provider.logical_strategy_calls,
        "logical_width_calls": provider.logical_width_calls,
        "cache_hits": provider.cache_hits,
    }
    gate = all(row["final_entropy_gain_ci95"][0] > 0.0 for row in paired.values())
    return {
        "schema_version": 1,
        "stage": "L3",
        "task_provenance": "COPEx Location_budgeted equations; independent implementation",
        "config": asdict(config),
        "summary": {arm: _arm_summary(trials, arm) for arm in ARMS},
        "paired": paired,
        "mechanics": mechanics,
        "gate_passed": gate and all(
            mechanics[key]
            for key in (
                "all_actions_legal",
                "initial_strategy_cells_shared_with_d1",
                "width_scorer_units_match_strategy_eig",
                "grid_d2_scorer_units_match_strategy_eig",
            )
        ),
        "requests": provider.accepted_requests,
        "invalid_responses": provider.invalid_responses,
        "trials": trials,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# COPEx-Task Continuous Strategy-Prior L3",
        "",
        "The LLM generates compact continuous plans; the task equations, likelihood, posterior, rollout simulation, and scoring are programmatic.",
        "",
        "| Arm | Final entropy | Final RMSE | Mean scorer units / decision |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = summary["summary"][arm]
        lines.append(f"| {arm} | {row['final_entropy_mean']:.4f} | {row['final_rmse_mean']:.4f} | {row['mean_scorer_units']:.1f} |")
    lines.extend(["", "| Comparison | Entropy gain | 95% paired bootstrap CI | W / T / L |", "| --- | ---: | --- | --- |"])
    for label, row in summary["paired"].items():
        ci = row["final_entropy_gain_ci95"]
        wtl = row["wins_ties_losses"]
        lines.append(f"| {label} | {row['final_entropy_gain_mean']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {wtl[0]} / {wtl[1]} / {wtl[2]} |")
    lines.extend(["", "## Mechanics", ""])
    lines.extend(f"- {key}: `{value}`." for key, value in summary["mechanics"].items())
    lines.extend(["", f"**L3 gate passed: `{summary['gate_passed']}`.**", ""])
    return "\n".join(lines)


def _usage(model: Any) -> dict[str, Any]:
    snapshot = getattr(model, "usage_snapshot", None)
    return snapshot() if callable(snapshot) else {"backend": "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config_nonmyopic_copex_strategy_l3_openrouter.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/nonmyopic/copex_strategy_l3/20260716"))
    parser.add_argument("--run-id", default="nonmyopic-copex-strategy-l3-20260716")
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--num-rounds", type=int, default=30)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--num-strategies", type=int, default=4)
    parser.add_argument("--planning-horizon", type=int, default=4)
    parser.add_argument("--rollout-samples", type=int, default=64)
    parser.add_argument("--grid-resolution", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31_001)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--trial-concurrency", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = L3Config(
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        num_particles=args.num_particles,
        num_strategies=args.num_strategies,
        planning_horizon=args.planning_horizon,
        rollout_samples=args.rollout_samples,
        grid_resolution=args.grid_resolution,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        trial_concurrency=args.trial_concurrency,
    )
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        model: ChatModel = DeterministicContinuousModel()
    else:
        runtime: Config = load_config(args.config)
        runtime.run_id = args.run_id
        model = build_model_adapter(runtime.model_pairs[0].questioner, config=runtime)
    provider = ContinuousStrategyProvider(model, config)
    try:
        summary = run_l3(provider, config)
    except ContinuousProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "L3",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(config),
            "requests": provider.accepted_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage(model),
        }
        (args.output_dir / "L3_FAILURE.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise
    summary["usage"] = _usage(model)
    summary["run_id"] = args.run_id
    summary["dry_run"] = args.dry_run
    (args.output_dir / "L3.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "L3.md").write_text(render_report(summary))
    print(json.dumps({"gate_passed": summary["gate_passed"], "mechanics": summary["mechanics"]}, indent=2))


if __name__ == "__main__":
    main()
