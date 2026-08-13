#!/usr/bin/env python3
"""Pure mechanics for atomic-particle Number Game depth-three BED."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Iterable, Mapping, Sequence

from scripts.number_game_generator_aware_bed import (
    DOMAIN,
    RuleHypothesis,
    binary_entropy,
    compile_expression,
)


INTERFACE_VERSION = "number-game-atomic-particle-depth3-1"
TREE_SEEDS = tuple(range(202608143000, 202608143004))
MODEL_SEED_START = 202608136000
MODEL_REQUESTS = 6400
MODEL_SEEDS = tuple(range(MODEL_SEED_START, MODEL_SEED_START + MODEL_REQUESTS))
INITIAL_SLOTS = 64
GENERATED_SLOTS = 32
BELIEF_WIDTH = 64
ROOT_COUNT = 4
DIVERSITY_CUES = (
    "divisibility",
    "digit structure",
    "bounded interval",
    "prime or square structure",
    "affine transform",
    "modular structure",
    "boolean composition",
    "sequence structure",
)


@dataclass(frozen=True)
class AtomicResponse:
    name: str
    expression: str
    hypothesis: RuleHypothesis | None
    rejection: str | None


@dataclass(frozen=True)
class RootPlan:
    root: int
    score: float
    second_queries: Mapping[bool, int]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "number_game_atomic_hypothesis",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "expression"],
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 100},
                    "expression": {"type": "string", "minLength": 1, "maxLength": 240},
                },
            },
        },
    }


def particle_messages(
    observations: Sequence[tuple[int, bool]],
    *,
    slot: int,
) -> list[dict[str, str]]:
    history = [{"number": number, "answer": "YES" if answer else "NO"} for number, answer in observations]
    cue = DIVERSITY_CUES[slot % len(DIVERSITY_CUES)]
    system = (
        "Sample one plausible human-interpretable hypothesis for the classic Number Game. "
        "Return only the required JSON object. The expression is a single Python boolean expression over "
        "integer n in 0..100. Allowed syntax is and/or/not, bounded integer arithmetic and comparisons, and "
        "only divisible(n,k), is_square(n), is_power_of_two(n), is_prime(n), digit_sum(n), and "
        "ends_with(n,digit). Do not use containers, indexing, attributes, imports, lambdas, comprehensions, "
        "explicit member lists, exception lists, or constants. The rule must obey every observation exactly."
    )
    user = canonical_json({
        "observations": history,
        "diversity_cue": cue,
        "instruction": "Propose one coherent general rule; the cue is a soft diversity direction, not a required rule family.",
    })
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_atomic(raw: str, observations: Sequence[tuple[int, bool]]) -> AtomicResponse:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return AtomicResponse("", "", None, "json")
    if not isinstance(value, dict) or set(value) != {"name", "expression"}:
        return AtomicResponse("", "", None, "shape")
    name = " ".join(str(value["name"]).strip().split())
    expression = " ".join(str(value["expression"]).strip().split())
    if not name or len(name) > 100:
        return AtomicResponse(name, expression, None, "name")
    try:
        extension = compile_expression(expression)
    except (SyntaxError, ValueError):
        return AtomicResponse(name, expression, None, "expression")
    if any(extension[number] is not answer for number, answer in observations):
        return AtomicResponse(name, expression, None, "history")
    hypothesis = RuleHypothesis(name=name, expression=expression, extension=extension)
    return AtomicResponse(name, expression, hypothesis, None)


def parse_group(raw_values: Sequence[str], observations: Sequence[tuple[int, bool]]) -> tuple[list[RuleHypothesis], dict[str, Any]]:
    parsed = [parse_atomic(raw, observations) for raw in raw_values]
    valid = [row.hypothesis for row in parsed if row.hypothesis is not None]
    rejections: dict[str, int] = {}
    for row in parsed:
        if row.rejection is not None:
            rejections[row.rejection] = rejections.get(row.rejection, 0) + 1
    return valid, {
        "slots": len(raw_values),
        "valid_particles": len(valid),
        "unique_extensions": len({row.extension for row in valid}),
        "rejections": rejections,
    }


def systematic_resample(particles: Sequence[RuleHypothesis], size: int, seed: int) -> list[RuleHypothesis]:
    if not particles or size <= 0:
        raise ValueError("cannot resample an empty or nonpositive particle set")
    rng = random.Random(seed)
    offset = rng.random() / size
    return [particles[min(int((offset + index / size) * len(particles)), len(particles) - 1)] for index in range(size)]


def consistent(particles: Sequence[RuleHypothesis], observations: Sequence[tuple[int, bool]]) -> list[RuleHypothesis]:
    return [row for row in particles if all(row.extension[number] is answer for number, answer in observations)]


def refresh_belief(
    parent: Sequence[RuleHypothesis],
    generated: Sequence[RuleHypothesis],
    observations: Sequence[tuple[int, bool]],
    *,
    seed: int,
    width: int = BELIEF_WIDTH,
) -> list[RuleHypothesis]:
    if width % 2:
        raise ValueError("refreshed belief width must be even")
    retained = consistent(parent, observations)
    generated_valid = consistent(generated, observations)
    half = width // 2
    return [
        *systematic_resample(retained, half, seed),
        *systematic_resample(generated_valid, half, seed ^ 0x5DEECE66D),
    ]


def answer_probability(particles: Sequence[RuleHypothesis], query: int) -> float:
    if not particles:
        raise ValueError("particle belief is empty")
    return sum(row.extension[query] for row in particles) / len(particles)


def query_eig(particles: Sequence[RuleHypothesis], query: int) -> float:
    return binary_entropy(answer_probability(particles, query))


def best_query(particles: Sequence[RuleHypothesis], excluded: Iterable[int] = ()) -> tuple[int, float]:
    blocked = set(excluded)
    candidates = [(query_eig(particles, query), query) for query in DOMAIN if query not in blocked]
    value, query = max(candidates, key=lambda item: (item[0], -item[1]))
    return query, value


def filter_resample(
    particles: Sequence[RuleHypothesis],
    observations: Sequence[tuple[int, bool]],
    *,
    seed: int,
    width: int | None = None,
) -> list[RuleHypothesis]:
    kept = consistent(particles, observations)
    return systematic_resample(kept, width or len(particles), seed)


def fixed_depth3_score(particles: Sequence[RuleHypothesis], root: int, *, seed: int) -> RootPlan:
    immediate = query_eig(particles, root)
    future = 0.0
    second_queries: dict[bool, int] = {}
    for first_answer in (False, True):
        probability = answer_probability(particles, root)
        probability = probability if first_answer else 1.0 - probability
        if probability == 0:
            continue
        first = filter_resample(particles, ((root, first_answer),), seed=seed ^ (root * 17 + int(first_answer)))
        second, second_eig = best_query(first, (root,))
        second_queries[first_answer] = second
        third_expected = 0.0
        for second_answer in (False, True):
            branch_probability = answer_probability(first, second)
            branch_probability = branch_probability if second_answer else 1.0 - branch_probability
            if branch_probability == 0:
                continue
            second_belief = filter_resample(first, ((second, second_answer),), seed=seed ^ (root * 101 + second * 7 + int(second_answer)))
            third_expected += branch_probability * best_query(second_belief, (root, second))[1]
        future += probability * (second_eig + third_expected)
    return RootPlan(root=root, score=immediate + future, second_queries=second_queries)


def candidate_roots(particles: Sequence[RuleHypothesis], *, seed: int) -> tuple[int, ...]:
    immediate = {query: query_eig(particles, query) for query in DOMAIN}
    fixed = {query: fixed_depth3_score(particles, query, seed=seed).score for query in DOMAIN}
    chosen = [max(DOMAIN, key=lambda query: (immediate[query], -query))]
    fixed_best = max(DOMAIN, key=lambda query: (fixed[query], -query))
    if fixed_best not in chosen:
        chosen.append(fixed_best)
    signatures = {tuple(row.extension[query] for row in particles) for query in chosen}
    for query in sorted(DOMAIN, key=lambda value: (-immediate[value], value)):
        signature = tuple(row.extension[query] for row in particles)
        if query not in chosen and signature not in signatures:
            chosen.append(query)
            signatures.add(signature)
        if len(chosen) == ROOT_COUNT:
            break
    if len(chosen) != ROOT_COUNT:
        raise ValueError("could not construct four answer-distinct roots")
    return tuple(chosen)


def generated_depth3_score(
    initial: Sequence[RuleHypothesis],
    root: int,
    first_generated: Mapping[bool, Sequence[RuleHypothesis]],
    second_generated: Mapping[tuple[bool, bool], Sequence[RuleHypothesis]],
    *,
    seed: int,
    width: int = BELIEF_WIDTH,
    fixed_second_queries: Mapping[bool, int] | None = None,
) -> RootPlan:
    immediate = query_eig(initial, root)
    future = 0.0
    second_queries: dict[bool, int] = {}
    for first_answer in (False, True):
        probability = answer_probability(initial, root)
        probability = probability if first_answer else 1.0 - probability
        if probability == 0:
            continue
        history1 = ((root, first_answer),)
        first = refresh_belief(initial, first_generated[first_answer], history1, seed=seed ^ (root * 17 + int(first_answer)), width=width)
        second = fixed_second_queries[first_answer] if fixed_second_queries is not None else best_query(first, (root,))[0]
        if second == root:
            raise ValueError("fixed second query repeats the root")
        second_eig = query_eig(first, second)
        second_queries[first_answer] = second
        third_expected = 0.0
        for second_answer in (False, True):
            branch_probability = answer_probability(first, second)
            branch_probability = branch_probability if second_answer else 1.0 - branch_probability
            if branch_probability == 0:
                continue
            history2 = (*history1, (second, second_answer))
            second_belief = refresh_belief(first, second_generated[(first_answer, second_answer)], history2, seed=seed ^ (root * 101 + second * 7 + int(second_answer)), width=width)
            third_expected += branch_probability * best_query(second_belief, (root, second))[1]
        future += probability * (second_eig + third_expected)
    return RootPlan(root=root, score=immediate + future, second_queries=second_queries)


def choose_plan(plans: Mapping[int, RootPlan]) -> int:
    return max(plans, key=lambda root: (plans[root].score, -root))


def predictive_probabilities(particles: Sequence[RuleHypothesis]) -> tuple[float, ...]:
    return tuple(answer_probability(particles, query) for query in DOMAIN)


def posterior_predictive_brier(
    particles: Sequence[RuleHypothesis],
    truth: RuleHypothesis,
    excluded: Iterable[int],
) -> float:
    blocked = set(excluded)
    probabilities = predictive_probabilities(particles)
    values = [(probabilities[query] - float(truth.extension[query])) ** 2 for query in DOMAIN if query not in blocked]
    return sum(values) / len(values)


def extension_hash(hypothesis: RuleHypothesis) -> str:
    return hashlib.sha256(bytes(hypothesis.extension)).hexdigest()


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal nontrivial length")

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        out = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + end - 1) / 2.0
            for position in range(start, end):
                out[order[position]] = rank
            start = end
        return out

    x, y = ranks(left), ranks(right)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0
