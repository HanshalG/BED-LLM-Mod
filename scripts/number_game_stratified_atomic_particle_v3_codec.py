#!/usr/bin/env python3
"""Pure signature strata for the frozen atomic-particle V3 cohort."""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from scripts.number_game_atomic_particle_codec import (
    AtomicResponse,
    BELIEF_WIDTH,
    GENERATED_SLOTS,
    INITIAL_SLOTS,
    ROOT_COUNT,
    RootPlan,
    answer_probability,
    best_query,
    candidate_roots,
    canonical_json,
    choose_plan,
    consistent,
    extension_hash,
    filter_resample,
    fixed_depth3_score,
    generated_depth3_score,
    parse_atomic,
    posterior_predictive_brier,
    predictive_probabilities,
    query_eig,
    refresh_belief,
    response_format,
    spearman,
    systematic_resample,
)
from scripts.number_game_generator_aware_bed import RuleHypothesis


INTERFACE_VERSION = "number-game-stratified-atomic-particle-depth3-3"
TREE_SEEDS = tuple(range(202608187000, 202608187004))
MODEL_SEED_START = 202608180000
MODEL_REQUESTS = 6400
MODEL_SEEDS = tuple(range(MODEL_SEED_START, MODEL_SEED_START + MODEL_REQUESTS))
ANCHOR_ORDER = (
    7, 12, 25, 42, 81, 5, 16, 33, 64, 90,
    2, 9, 20, 50, 75, 3, 14, 27, 60, 99,
    1, 6, 18, 40, 70, 4, 11, 30, 55, 88,
)
SIGNATURES = tuple(
    tuple(bool(index & (1 << (4 - bit))) for bit in range(5))
    for index in range(32)
)


def anchors_for(protected: Iterable[int]) -> tuple[int, ...]:
    blocked = set(protected)
    anchors = tuple(number for number in ANCHOR_ORDER if number not in blocked)[:5]
    if len(anchors) != 5 or set(anchors) & blocked:
        raise ValueError("could not construct five protected anchors")
    return anchors


def signature_for_slot(slot: int, count: int) -> tuple[bool, ...]:
    if count <= 0 or count % len(SIGNATURES) != 0 or not 0 <= slot < count:
        raise ValueError("stratified group size or slot is invalid")
    repeats = count // len(SIGNATURES)
    return SIGNATURES[slot // repeats]


def signature_text(signature: Sequence[bool]) -> str:
    if len(signature) != 5:
        raise ValueError("signature must have five bits")
    return "".join("1" if value else "0" for value in signature)


def particle_signature(hypothesis: RuleHypothesis, anchors: Sequence[int]) -> tuple[bool, ...]:
    if len(anchors) != 5:
        raise ValueError("particle anchors must have length five")
    return tuple(hypothesis.extension[number] for number in anchors)


def particle_messages(
    observations: Sequence[tuple[int, bool]],
    *,
    anchors: Sequence[int],
    signature: Sequence[bool],
) -> list[dict[str, str]]:
    if len(anchors) != 5 or len(set(anchors)) != 5 or len(signature) != 5:
        raise ValueError("stratum requires five distinct anchors and five bits")
    history = [{"number": number, "answer": "YES" if answer else "NO"} for number, answer in observations]
    target = [
        {"number": number, "required_membership": "YES" if answer else "NO"}
        for number, answer in zip(anchors, signature, strict=True)
    ]
    system = (
        "Sample one plausible human-interpretable hypothesis for the classic Number Game. "
        "Return only the required JSON object. The expression is a single Python boolean expression over "
        "integer n in 0..100. Allowed syntax is and/or/not, bounded integer arithmetic and comparisons, and "
        "only divisible(n,k), is_square(n), is_power_of_two(n), is_prime(n), digit_sum(n), and "
        "ends_with(n,digit). Do not use containers, indexing, attributes, imports, lambdas, comprehensions, "
        "explicit member lists, exception lists, or constants. The rule must obey every observation and every "
        "required anchor membership exactly. Find a coherent general rule rather than listing the anchors."
    )
    user = canonical_json({
        "observations": history,
        "target_stratum": target,
        "instruction": "Propose one coherent general rule satisfying the exact target stratum.",
    })
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_stratified(
    raw: str,
    observations: Sequence[tuple[int, bool]],
    anchors: Sequence[int],
    signature: Sequence[bool],
) -> AtomicResponse:
    parsed = parse_atomic(raw, observations)
    if parsed.hypothesis is None:
        return parsed
    if particle_signature(parsed.hypothesis, anchors) != tuple(signature):
        return AtomicResponse(parsed.name, parsed.expression, None, "signature")
    return parsed


def request_identity(
    observations: Sequence[tuple[int, bool]],
    *,
    protected: Iterable[int],
    slot: int,
    count: int,
) -> tuple[tuple[int, ...], tuple[bool, ...], list[dict[str, str]]]:
    anchors = anchors_for(protected)
    signature = signature_for_slot(slot, count)
    return anchors, signature, particle_messages(observations, anchors=anchors, signature=signature)
