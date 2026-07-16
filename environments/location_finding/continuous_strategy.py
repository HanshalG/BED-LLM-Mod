"""Compact continuous strategies for the COPEx location task specification."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Literal

import numpy as np


StepKind = Literal["vector", "toward_rank", "midpoint_ranks", "toward_mean"]


class ContinuousStrategyParseError(ValueError):
    """A continuous strategy cell does not satisfy the strict plan grammar."""


@dataclass(frozen=True)
class ContinuousStrategyStep:
    kind: StepKind
    dx: float | None = None
    dy: float | None = None
    rank: int | None = None
    ranks: tuple[int, int] | None = None


@dataclass(frozen=True)
class ContinuousStrategy:
    name: str
    description: str
    steps: tuple[ContinuousStrategyStep, ...]
    canonical_json: str


@dataclass(frozen=True)
class ContinuousStrategyScore:
    total_eig: float
    root_action: tuple[float, float]
    scorer_units: int


def copex_signal(source: np.ndarray, query: np.ndarray) -> float:
    distance_sq = float(np.sum((np.asarray(source, dtype=float) - query) ** 2))
    return math.log(0.1 + 1.0 / (1e-4 + distance_sq))


def copex_signal_many(particles: np.ndarray, query: np.ndarray) -> np.ndarray:
    distance_sq = np.sum((np.asarray(particles, dtype=float) - query[None, :]) ** 2, axis=1)
    return np.log(0.1 + 1.0 / (1e-4 + distance_sq))


def particle_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def update_copex_belief(
    particles: np.ndarray,
    probabilities: np.ndarray,
    query: np.ndarray,
    observation: float,
    *,
    noise_sd: float = 0.5,
) -> np.ndarray:
    means = copex_signal_many(particles, query)
    log_weights = np.log(np.maximum(probabilities, 1e-300))
    log_weights += -0.5 * ((float(observation) - means) / noise_sd) ** 2
    log_weights -= math.log(noise_sd) + 0.5 * math.log(2.0 * math.pi)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    return weights / float(np.sum(weights))


def _parse_step(item: Any, *, max_step: float, max_rank: int) -> ContinuousStrategyStep:
    if not isinstance(item, dict) or not isinstance(item.get("kind"), str):
        raise ContinuousStrategyParseError("every step must be an object with a string kind")
    kind = item["kind"]
    if kind == "vector":
        if set(item) != {"kind", "dx", "dy"}:
            raise ContinuousStrategyParseError("vector steps require exactly kind, dx, and dy")
        dx, dy = item["dx"], item["dy"]
        if not isinstance(dx, (int, float)) or not isinstance(dy, (int, float)):
            raise ContinuousStrategyParseError("vector dx and dy must be numeric")
        dx, dy = float(dx), float(dy)
        if max(abs(dx), abs(dy)) > max_step + 1e-12 or max(abs(dx), abs(dy)) <= 1e-12:
            raise ContinuousStrategyParseError(
                f"vector steps must be nonzero with max(|dx|, |dy|) <= {max_step}"
            )
        return ContinuousStrategyStep(kind="vector", dx=dx, dy=dy)
    if kind == "toward_rank":
        if set(item) != {"kind", "rank"} or not isinstance(item["rank"], int):
            raise ContinuousStrategyParseError("toward_rank requires exactly an integer rank")
        if not 0 <= item["rank"] <= max_rank:
            raise ContinuousStrategyParseError(f"rank must be in [0, {max_rank}]")
        return ContinuousStrategyStep(kind="toward_rank", rank=int(item["rank"]))
    if kind == "midpoint_ranks":
        if set(item) != {"kind", "ranks"}:
            raise ContinuousStrategyParseError("midpoint_ranks requires exactly kind and ranks")
        ranks = item["ranks"]
        if (
            not isinstance(ranks, list)
            or len(ranks) != 2
            or not all(isinstance(rank, int) for rank in ranks)
            or ranks[0] == ranks[1]
            or not all(0 <= rank <= max_rank for rank in ranks)
        ):
            raise ContinuousStrategyParseError(
                f"ranks must be two distinct integers in [0, {max_rank}]"
            )
        return ContinuousStrategyStep(kind="midpoint_ranks", ranks=(ranks[0], ranks[1]))
    if kind == "toward_mean":
        if set(item) != {"kind"}:
            raise ContinuousStrategyParseError("toward_mean requires only the kind field")
        return ContinuousStrategyStep(kind="toward_mean")
    raise ContinuousStrategyParseError(f"unknown step kind: {kind!r}")


def parse_continuous_strategy_cell(
    response: str,
    *,
    expected_count: int,
    horizon: int,
    max_step: float = 0.1,
    max_rank: int = 3,
) -> tuple[ContinuousStrategy, ...]:
    normalized = response.strip()
    if normalized.startswith("```json\n"):
        if not normalized.endswith("\n```"):
            raise ContinuousStrategyParseError("response has an incomplete JSON fence")
        normalized = normalized[len("```json\n") : -len("\n```")]
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ContinuousStrategyParseError("response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"strategies"}:
        raise ContinuousStrategyParseError("response must contain exactly the strategies key")
    items = payload["strategies"]
    if not isinstance(items, list) or len(items) != expected_count:
        raise ContinuousStrategyParseError(f"expected exactly {expected_count} strategies")
    strategies: list[ContinuousStrategy] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {"name", "description", "steps"}:
            raise ContinuousStrategyParseError(
                f"strategy {index} requires exactly name, description, and steps"
            )
        if not isinstance(item["name"], str) or not item["name"].strip():
            raise ContinuousStrategyParseError(f"strategy {index} needs a nonempty name")
        if not isinstance(item["description"], str) or not item["description"].strip():
            raise ContinuousStrategyParseError(f"strategy {index} needs a nonempty description")
        if not isinstance(item["steps"], list) or len(item["steps"]) != horizon:
            raise ContinuousStrategyParseError(
                f"strategy {index} must contain exactly {horizon} steps"
            )
        canonical = json.dumps(item, sort_keys=True, separators=(",", ":"))
        if canonical in seen:
            raise ContinuousStrategyParseError("strategies must be distinct")
        seen.add(canonical)
        strategies.append(
            ContinuousStrategy(
                name=item["name"].strip(),
                description=item["description"].strip(),
                steps=tuple(
                    _parse_step(step, max_step=max_step, max_rank=max_rank)
                    for step in item["steps"]
                ),
                canonical_json=canonical,
            )
        )
    return tuple(strategies)


def execute_continuous_step(
    step: ContinuousStrategyStep,
    *,
    position: np.ndarray,
    particles: np.ndarray,
    probabilities: np.ndarray,
    max_step: float = 0.1,
) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    if step.kind == "vector":
        target = position + np.asarray([step.dx, step.dy], dtype=float)
    else:
        ranking = np.argsort(-probabilities, kind="stable")
        if step.kind == "toward_rank":
            target = particles[ranking[min(int(step.rank or 0), len(ranking) - 1)]]
        elif step.kind == "midpoint_ranks":
            assert step.ranks is not None
            first = particles[ranking[min(step.ranks[0], len(ranking) - 1)]]
            second = particles[ranking[min(step.ranks[1], len(ranking) - 1)]]
            target = 0.5 * (first + second)
        elif step.kind == "toward_mean":
            target = np.sum(particles * probabilities[:, None], axis=0)
        else:  # pragma: no cover - dataclass construction is parser-controlled
            raise ValueError(f"unknown step kind: {step.kind}")
    delta = np.clip(np.asarray(target, dtype=float) - position, -max_step, max_step)
    return np.clip(position + delta, 0.0, 1.0)


def score_continuous_strategy(
    strategy: ContinuousStrategy,
    *,
    position: np.ndarray,
    particles: np.ndarray,
    probabilities: np.ndarray,
    truth_uniforms: np.ndarray,
    noise_zs: np.ndarray,
    noise_sd: float = 0.5,
    max_step: float = 0.1,
) -> ContinuousStrategyScore:
    if noise_zs.ndim != 2 or noise_zs.shape[1] != len(strategy.steps):
        raise ValueError("noise_zs must have shape (rollouts, strategy horizon)")
    if len(truth_uniforms) != len(noise_zs):
        raise ValueError("truth_uniforms and noise_zs need the same rollout count")
    cumulative = np.cumsum(probabilities)
    truth_indices = np.minimum(
        np.searchsorted(cumulative, truth_uniforms, side="right"), len(probabilities) - 1
    )
    start_entropy = particle_entropy(probabilities)
    drops = np.empty(len(noise_zs), dtype=float)
    root_action: np.ndarray | None = None
    for rollout_index, (truth_index, rollout_zs) in enumerate(zip(truth_indices, noise_zs)):
        branch_probabilities = probabilities.copy()
        branch_position = np.asarray(position, dtype=float).copy()
        for step_index, (step, noise_z) in enumerate(zip(strategy.steps, rollout_zs)):
            branch_position = execute_continuous_step(
                step,
                position=branch_position,
                particles=particles,
                probabilities=branch_probabilities,
                max_step=max_step,
            )
            if root_action is None and step_index == 0:
                root_action = branch_position.copy()
            observation = copex_signal(particles[int(truth_index)], branch_position)
            observation += noise_sd * float(noise_z)
            branch_probabilities = update_copex_belief(
                particles,
                branch_probabilities,
                branch_position,
                observation,
                noise_sd=noise_sd,
            )
        drops[rollout_index] = start_entropy - particle_entropy(branch_probabilities)
    assert root_action is not None
    return ContinuousStrategyScore(
        total_eig=float(np.mean(drops)),
        root_action=(float(root_action[0]), float(root_action[1])),
        scorer_units=int(len(noise_zs) * len(strategy.steps)),
    )


def random_continuous_strategy(
    rng: np.random.Generator,
    *,
    horizon: int,
    max_step: float = 0.1,
    max_rank: int = 3,
    index: int = 0,
) -> ContinuousStrategy:
    steps: list[dict[str, Any]] = []
    for _ in range(horizon):
        kind = str(rng.choice(["vector", "toward_rank", "midpoint_ranks", "toward_mean"]))
        if kind == "vector":
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            scale = max_step / max(abs(math.cos(angle)), abs(math.sin(angle)))
            steps.append({"kind": kind, "dx": scale * math.cos(angle), "dy": scale * math.sin(angle)})
        elif kind == "toward_rank":
            steps.append({"kind": kind, "rank": int(rng.integers(max_rank + 1))})
        elif kind == "midpoint_ranks":
            ranks = rng.choice(max_rank + 1, size=2, replace=False)
            steps.append({"kind": kind, "ranks": [int(ranks[0]), int(ranks[1])]})
        else:
            steps.append({"kind": kind})
    item = {
        "name": f"random-plan-{index}",
        "description": "Grammar-matched random continuous strategy.",
        "steps": steps,
    }
    canonical = json.dumps(item, sort_keys=True, separators=(",", ":"))
    return ContinuousStrategy(
        name=item["name"],
        description=item["description"],
        steps=tuple(_parse_step(step, max_step=max_step, max_rank=max_rank) for step in steps),
        canonical_json=canonical,
    )
