#!/usr/bin/env python3
"""Smoke-test a blinded final-readiness scorer over regenerated Zendo beliefs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.zendo_path_dependent_belief_gate import (
    MODEL_ID,
    OBSERVATION_ACCURACY,
    PARTICLE_COUNT,
    RULE_ORDER,
    SOURCE_COMMIT,
    _argmax,
    _canonical_json,
    _sha256,
    _signature_count,
    audit_bank,
    behavior_agreements,
    branch_key,
    eig,
    evaluate_rule,
    initial_messages,
    parse_hypotheses,
    posterior_weights,
    random_scene,
    raw_official_scene,
    refresh_messages,
    spearman,
    truth_function,
    update_weights,
    validate_rule,
    verify_source,
    weighted_agreement,
)


INTERFACE_VERSION = "zendo-final-readiness-belief-1"
TASK_NAME = "phi"
SELECTION_SEED = 24369
POOL_SIZE = 256
ROOT_COUNT = 4
EXPECTED_REQUESTS = 10
MAX_NEW_TOKENS = 8192
COST_CAP_USD = 0.75
MIN_BRANCH_PROBABILITY = 0.10
ROOT_EIG_FRACTIONS = (1.0, 0.75, 0.50, 0.25)
MIN_IMMEDIATE_SACRIFICE = 0.01
MIN_ENDPOINT_RANGE = 0.10
MIN_MODEL_MINUS_MYOPIC = 0.05
MIN_MODEL_MINUS_FIXED = 0.03
MIN_SCORE_ENDPOINT_SPEARMAN = 0.30
AUDIT_SCENE_COUNT = 512


class GateExecutionError(RuntimeError):
    """A failed-closed gate with usage accounting attached."""

    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def deterministic_scene_pool(
    initial_scene: dict[str, Any],
    *,
    selection_seed: int = SELECTION_SEED,
) -> list[dict[str, Any]]:
    """Create a target-blind, deterministic bank of distinct legal scenes."""
    rng = random.Random(selection_seed)
    initial_key = _canonical_json(initial_scene)
    scenes: dict[str, dict[str, Any]] = {}
    while len(scenes) < POOL_SIZE:
        scene = random_scene(rng)
        key = _canonical_json(scene)
        if key != initial_key:
            scenes.setdefault(key, scene)
    return list(scenes.values())


def deterministic_random_audit_bank(
    *,
    task_index: int,
    selection_seed: int,
) -> list[dict[str, Any]]:
    """Create a hidden-rule-independent audit bank within the six-block DSL."""
    rng = random.Random(selection_seed + 1009 * task_index)
    scenes: dict[str, dict[str, Any]] = {}
    while len(scenes) < AUDIT_SCENE_COUNT:
        scene = random_scene(rng)
        scenes.setdefault(_canonical_json(scene), scene)
    return list(scenes.values())


def parse_particle_population(
    text: str,
    *,
    allow_duplicate_asts: bool,
) -> list[dict[str, Any]]:
    """Parse either a unique support or a particle multiset."""
    if not allow_duplicate_asts:
        return parse_hypotheses(text)
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or set(payload) != {"hypotheses"}:
        raise ValueError("particle response has unexpected fields")
    raw = payload["hypotheses"]
    if not isinstance(raw, list) or len(raw) != PARTICLE_COUNT:
        raise ValueError(f"expected exactly {PARTICLE_COUNT} particles")
    particles = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or set(item) != {
            "id",
            "rule_text",
            "rule",
        }:
            raise ValueError("particle has wrong fields")
        expected_id = f"H{index:02d}"
        if item["id"] != expected_id:
            raise ValueError(f"expected particle id {expected_id}")
        rule_text = item["rule_text"]
        if not isinstance(rule_text, str) or not rule_text.strip():
            raise ValueError("rule_text must be nonempty")
        particles.append(
            {
                "id": expected_id,
                "rule_text": " ".join(rule_text.split()),
                "rule": validate_rule(item["rule"]),
            }
        )
    return particles


def _prediction_signature(
    hypotheses: Sequence[dict[str, Any]],
    scene: dict[str, Any],
) -> tuple[bool, ...]:
    return tuple(
        evaluate_rule(hypothesis["rule"], scene) for hypothesis in hypotheses
    )


def _hamming(left: Sequence[bool], right: Sequence[bool]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def select_root_scenes(
    hypotheses: Sequence[dict[str, Any]],
    initial_weights: Sequence[float],
    pool: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select four target-blind roots spanning the support's EIG range."""
    by_signature: dict[tuple[bool, ...], dict[str, Any]] = {}
    for pool_index, scene in enumerate(pool):
        signature = _prediction_signature(hypotheses, scene)
        _, probability_yes = update_weights(
            hypotheses, initial_weights, scene, True
        )
        row = {
            "pool_index": pool_index,
            "scene": scene,
            "signature": signature,
            "probability_yes": probability_yes,
            "eig": eig(hypotheses, initial_weights, scene),
        }
        incumbent = by_signature.get(signature)
        if incumbent is None or (row["eig"], -pool_index) > (
            incumbent["eig"],
            -incumbent["pool_index"],
        ):
            by_signature[signature] = row
    informative = [
        row
        for row in by_signature.values()
        if min(row["probability_yes"], 1.0 - row["probability_yes"])
        >= MIN_BRANCH_PROBABILITY
    ]
    if len(informative) < ROOT_COUNT:
        raise ValueError(
            f"only {len(informative)} informative prediction signatures"
        )
    maximum_eig = max(row["eig"] for row in informative)
    selected: list[dict[str, Any]] = []
    for fraction in ROOT_EIG_FRACTIONS:
        target = maximum_eig * fraction

        def key(row: dict[str, Any]) -> tuple[float, int, float, int]:
            novelty = (
                min(
                    _hamming(row["signature"], other["signature"])
                    for other in selected
                )
                if selected
                else len(row["signature"])
            )
            return (
                -abs(row["eig"] - target),
                novelty,
                row["eig"],
                -row["pool_index"],
            )

        selected.append(
            max(
                (
                    row
                    for row in informative
                    if row["pool_index"]
                    not in {other["pool_index"] for other in selected}
                ),
                key=key,
            )
        )
    public_rows = [
        {
            "root_index": index,
            "pool_index": row["pool_index"],
            "initial_eig": row["eig"],
            "initial_probability_yes": row["probability_yes"],
            "prediction_signature": [
                int(value) for value in row["signature"]
            ],
        }
        for index, row in enumerate(selected, start=1)
    ]
    return [row["scene"] for row in selected], public_rows


def best_continuation(
    hypotheses: Sequence[dict[str, Any]],
    weights: Sequence[float],
    pool: Sequence[dict[str, Any]],
    *,
    excluded_scene: dict[str, Any],
) -> tuple[int, dict[str, Any], float]:
    excluded_key = _canonical_json(excluded_scene)
    candidates = [
        (index, scene, eig(hypotheses, weights, scene))
        for index, scene in enumerate(pool)
        if _canonical_json(scene) != excluded_key
    ]
    return max(candidates, key=lambda row: (row[2], -row[0]))


def fixed_support_depth_two_scores(
    hypotheses: Sequence[dict[str, Any]],
    initial_weights: Sequence[float],
    roots: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
) -> list[float]:
    scores = []
    for root in roots:
        score = eig(hypotheses, initial_weights, root)
        for outcome in (False, True):
            branch_weights, probability = update_weights(
                hypotheses, initial_weights, root, outcome
            )
            _, _, continuation_eig = best_continuation(
                hypotheses,
                branch_weights,
                pool,
                excluded_scene=root,
            )
            score += probability * continuation_eig
        scores.append(score)
    return scores


def build_pathways(
    *,
    initial_scene: dict[str, Any],
    initial_hypotheses: Sequence[dict[str, Any]],
    initial_weights: Sequence[float],
    roots: Sequence[dict[str, Any]],
    branches: dict[str, list[dict[str, Any]]],
    pool: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    pathways = []
    for root_index, root in enumerate(roots):
        branch_packets = []
        for outcome in (False, True):
            key = branch_key(root_index, outcome)
            support = branches[key]
            history = [(initial_scene, True), (root, outcome)]
            weights = posterior_weights(support, history)
            pool_index, continuation, continuation_eig = best_continuation(
                support,
                weights,
                pool,
                excluded_scene=root,
            )
            _, probability = update_weights(
                initial_hypotheses,
                initial_weights,
                root,
                outcome,
            )
            branch_packets.append(
                {
                    "outcome": outcome,
                    "predicted_probability": probability,
                    "refreshed_belief": [
                        {
                            "rule_text": hypothesis["rule_text"],
                            "probability": weight,
                        }
                        for hypothesis, weight in zip(
                            support, weights, strict=True
                        )
                    ],
                    "continuation_pool_index": pool_index,
                    "continuation_scene": continuation,
                    "continuation_expected_information_nats": continuation_eig,
                }
            )
        pathways.append(
            {
                "label": chr(ord("A") + root_index),
                "root_scene": root,
                "branches": branch_packets,
            }
        )
    return pathways


def scorer_messages(
    pathways: Sequence[dict[str, Any]],
    *,
    particle_semantics: str = "unique_support",
) -> list[dict[str, str]]:
    multiplicity_note = (
        "Repeated semantic rules are repeated particles and their probabilities "
        "carry multiplicity."
        if particle_semantics == "multiset"
        else "Each displayed semantic rule is a distinct support element."
    )
    return [
        {
            "role": "system",
            "content": (
                "You assess complete two-experiment paths in a Zendo "
                "concept-learning game. Return one exact JSON object only, with "
                "no markdown, comments, reasoning, or extra fields."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    "A hidden executable rule classifies block scenes as good or bad.",
                    "For each root A-D you see both possible outcomes, their predicted",
                    "probabilities, the freshly regenerated semantic belief after that",
                    "outcome, and its best exact information-gain continuation scene.",
                    "Score each root from 0 to 100 for expected final readiness: after",
                    "the outcome-conditioned continuation, how likely is the agent to",
                    "have a behaviorally correct executable rule? Judge semantic",
                    "coverage, branch robustness, and whether the continuation resolves",
                    "the remaining ambiguity. You are not shown the hidden rule or any",
                    "immediate root information-gain score. Do not favor label order.",
                    multiplicity_note,
                    'Return exactly {"root_scores":[A,B,C,D]} using JSON integers.',
                    "PATHWAYS=" + _canonical_json(pathways),
                ]
            ),
        },
    ]


def parse_scorer(text: str) -> list[int]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or set(payload) != {"root_scores"}:
        raise ValueError("scorer response has unexpected fields")
    scores = payload["root_scores"]
    if (
        not isinstance(scores, list)
        or len(scores) != ROOT_COUNT
        or any(
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 100
            for score in scores
        )
    ):
        raise ValueError("root_scores must be four integers in [0,100]")
    return list(scores)


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(snapshot["adapter_cost_usd"]),
        "generator": snapshot,
    }


def _build_model(config: Config) -> Any:
    if len(config.model_pairs) != 1:
        raise ValueError("Zendo final-readiness gate requires one model pair")
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("Zendo final-readiness config selects the wrong model")
    return build_model_adapter(spec, config)


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _branch_signature(
    hypotheses: Sequence[dict[str, Any]],
    audit: Sequence[dict[str, Any]],
) -> tuple[tuple[bool, ...], ...]:
    return tuple(
        sorted(
            tuple(
                evaluate_rule(hypothesis["rule"], scene)
                for scene in audit
            )
            for hypothesis in hypotheses
        )
    )


def analyze_after_freeze(
    *,
    official_case: dict[str, Any],
    initial_scene: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    roots: list[dict[str, Any]],
    root_selection: list[dict[str, Any]],
    branches: dict[str, list[dict[str, Any]]],
    pathways: list[dict[str, Any]],
    scorer_scores: list[int],
    pool: list[dict[str, Any]],
    usage: dict[str, Any],
    task_name: str = TASK_NAME,
    interface_version: str = INTERFACE_VERSION,
    selection_seed: int = SELECTION_SEED,
    particle_semantics: str = "unique_support",
    audit_mode: str = "random_plus_official",
) -> dict[str, Any]:
    """Reveal the hidden rule only after every model output is frozen."""
    task_index = RULE_ORDER.index(task_name)
    if audit_mode == "random_plus_official":
        audit = audit_bank(task_index, official_case)
    elif audit_mode == "random_only":
        audit = deterministic_random_audit_bank(
            task_index=task_index,
            selection_seed=selection_seed,
        )
    else:
        raise ValueError(f"unknown audit mode {audit_mode!r}")
    truth = truth_function(task_name)
    initial_history = [(initial_scene, True)]
    initial_weights = posterior_weights(hypotheses, initial_history)
    immediate_scores = [
        eig(hypotheses, initial_weights, root) for root in roots
    ]
    fixed_scores = fixed_support_depth_two_scores(
        hypotheses, initial_weights, roots, pool
    )
    root_rows = []
    for root_index, root in enumerate(roots):
        actual_root_outcome = truth(root)
        actual_packet = next(
            packet
            for packet in pathways[root_index]["branches"]
            if packet["outcome"] == actual_root_outcome
        )
        continuation = actual_packet["continuation_scene"]
        actual_continuation_outcome = truth(continuation)
        support = branches[branch_key(root_index, actual_root_outcome)]
        final_weights = posterior_weights(
            support,
            [
                (initial_scene, True),
                (root, actual_root_outcome),
                (continuation, actual_continuation_outcome),
            ],
        )
        agreements = behavior_agreements(support, audit, truth)
        root_rows.append(
            {
                "root_index": root_index + 1,
                "root_label": chr(ord("A") + root_index),
                "actual_root_outcome": actual_root_outcome,
                "actual_continuation_outcome": actual_continuation_outcome,
                "immediate_eig": immediate_scores[root_index],
                "fixed_support_depth_two_score": fixed_scores[root_index],
                "model_final_readiness_score": scorer_scores[root_index],
                "realized_weighted_truth_agreement": weighted_agreement(
                    agreements, final_weights
                ),
                "realized_maximum_truth_agreement": max(agreements),
                "truth_recovered_at_0_95": max(agreements) >= 0.95,
                "continuation_pool_index": actual_packet[
                    "continuation_pool_index"
                ],
                "continuation_expected_information_nats": actual_packet[
                    "continuation_expected_information_nats"
                ],
            }
        )
    endpoints = [
        row["realized_weighted_truth_agreement"] for row in root_rows
    ]
    myopic_choice = _argmax(immediate_scores)
    fixed_choice = _argmax(fixed_scores)
    model_choice = _argmax(scorer_scores)
    support_signatures = {
        _branch_signature(branches[key], audit) for key in sorted(branches)
    }
    population_unique_ast_counts = {
        "initial": len(
            {_canonical_json(hypothesis["rule"]) for hypothesis in hypotheses}
        ),
        **{
            key: len(
                {
                    _canonical_json(hypothesis["rule"])
                    for hypothesis in branches[key]
                }
            )
            for key in sorted(branches)
        },
    }
    generator = usage.get("generator", {})
    mechanics = {
        "exact_10_physical_requests": (
            usage.get("physical_requests") == EXPECTED_REQUESTS
        ),
        "exact_10_http_attempts": (
            generator.get("http_attempts") == EXPECTED_REQUESTS
        ),
        "zero_transport_retries": generator.get("retry_count") == 0,
        "zero_reasoning_tokens": usage.get("reasoning_tokens") == 0,
        "zero_forced_exits": generator.get("forced_exits") == 0,
        "all_responses_parsed_without_repair": True,
        "initial_support_has_six_signatures": (
            _signature_count(hypotheses, audit) >= 6
        ),
        "four_distinct_root_prediction_signatures": (
            len(
                {
                    tuple(row["prediction_signature"])
                    for row in root_selection
                }
            )
            == ROOT_COUNT
        ),
        "four_informative_roots": all(
            min(
                row["initial_probability_yes"],
                1.0 - row["initial_probability_yes"],
            )
            >= MIN_BRANCH_PROBABILITY
            for row in root_selection
        ),
        "at_least_four_distinct_refreshed_supports": (
            len(support_signatures) >= 4
        ),
        "every_population_has_at_least_eight_unique_asts": all(
            count >= 8 for count in population_unique_ast_counts.values()
        ),
        "all_continuations_have_positive_finite_eig": all(
            math.isfinite(packet["continuation_expected_information_nats"])
            and packet["continuation_expected_information_nats"] > 0.0
            for pathway in pathways
            for packet in pathway["branches"]
        ),
        "scorer_scores_vary": len(set(scorer_scores)) >= 2,
        "scorer_has_unique_maximum": (
            sum(score == max(scorer_scores) for score in scorer_scores) == 1
        ),
        "cost_at_most_0_75": (
            usage.get("adapter_cost_usd", float("inf")) <= COST_CAP_USD
        ),
    }
    immediate_sacrifice = (
        immediate_scores[myopic_choice] - immediate_scores[model_choice]
    )
    model_minus_myopic = endpoints[model_choice] - endpoints[myopic_choice]
    model_minus_fixed = endpoints[model_choice] - endpoints[fixed_choice]
    science = {
        "model_root_differs_from_myopic": model_choice != myopic_choice,
        "immediate_sacrifice_at_least_0_01": (
            immediate_sacrifice >= MIN_IMMEDIATE_SACRIFICE
        ),
        "realized_endpoint_range_at_least_0_10": (
            max(endpoints) - min(endpoints) >= MIN_ENDPOINT_RANGE
        ),
        "model_minus_myopic_at_least_0_05": (
            model_minus_myopic >= MIN_MODEL_MINUS_MYOPIC
        ),
        "model_minus_fixed_at_least_0_03": (
            model_minus_fixed >= MIN_MODEL_MINUS_FIXED
        ),
        "score_endpoint_spearman_at_least_0_30": (
            spearman(scorer_scores, endpoints)
            >= MIN_SCORE_ENDPOINT_SPEARMAN
        ),
    }
    gates = {**mechanics, **science}
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": interface_version,
            "task_name": task_name,
            "model": MODEL_ID,
            "selection_seed": selection_seed,
            "source_commit": SOURCE_COMMIT,
            "pool_size": POOL_SIZE,
            "root_count": ROOT_COUNT,
            "root_eig_fractions": list(ROOT_EIG_FRACTIONS),
            "expected_physical_requests": EXPECTED_REQUESTS,
            "observation_accuracy": OBSERVATION_ACCURACY,
            "truth_hidden_until_all_branches_and_scores_frozen": True,
            "reasoning_requested": False,
            "repairs_or_scientific_retries": 0,
            "particle_semantics": particle_semantics,
            "audit_mode": audit_mode,
            "official_rules_are_development_only": True,
            "raw_responses_private_and_untracked": True,
        },
        "summary": {
            "mechanics_gates": mechanics,
            "science_gates": science,
            "gates": gates,
            "myopic_selected_root": myopic_choice + 1,
            "fixed_support_depth_two_selected_root": fixed_choice + 1,
            "model_selected_root": model_choice + 1,
            "immediate_sacrifice_nats": immediate_sacrifice,
            "realized_endpoint_range": max(endpoints) - min(endpoints),
            "model_minus_myopic_realized_weighted_truth_agreement": (
                model_minus_myopic
            ),
            "model_minus_fixed_realized_weighted_truth_agreement": (
                model_minus_fixed
            ),
            "score_realized_endpoint_spearman": spearman(
                scorer_scores, endpoints
            ),
            "initial_support_signature_count": _signature_count(
                hypotheses, audit
            ),
            "distinct_refreshed_support_count": len(support_signatures),
            "population_unique_ast_counts": population_unique_ast_counts,
        },
        "root_selection": root_selection,
        "root_rows": root_rows,
        "usage": usage,
        "audit": {
            "scene_count": len(audit),
            "sha256": hashlib.sha256(
                _canonical_json(audit).encode("utf-8")
            ).hexdigest(),
        },
    }


def run_gate(
    config: Config,
    *,
    source_dir: Path,
    raw_checkpoint_path: Path,
    model_adapter: Any | None = None,
    task_name: str = TASK_NAME,
    interface_version: str = INTERFACE_VERSION,
    selection_seed: int = SELECTION_SEED,
    allow_duplicate_particles: bool = False,
    audit_mode: str = "random_plus_official",
) -> dict[str, Any]:
    cases_path = verify_source(source_dir)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    official_case = dict(zip(RULE_ORDER, cases, strict=True))[task_name]
    initial_scene = raw_official_scene(official_case["t"][0])
    pool = deterministic_scene_pool(
        initial_scene, selection_seed=selection_seed
    )
    model = model_adapter if model_adapter is not None else _build_model(config)
    particle_semantics = (
        "multiset" if allow_duplicate_particles else "unique_support"
    )
    raw: dict[str, Any] = {
        "interface_version": interface_version,
        "task_name": task_name,
        "initial_hypotheses": None,
        "branch_refreshes": {},
        "scorer": None,
    }
    try:
        initial_response = model.chat_complete_messages_batched(
            [initial_messages(initial_scene)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=MAX_NEW_TOKENS,
        )[0]
        raw["initial_hypotheses"] = initial_response
        hypotheses = parse_particle_population(
            initial_response,
            allow_duplicate_asts=allow_duplicate_particles,
        )
        initial_weights = posterior_weights(
            hypotheses, [(initial_scene, True)]
        )
        roots, root_selection = select_root_scenes(
            hypotheses, initial_weights, pool
        )
        interventions = [
            (root_index, outcome)
            for root_index in range(ROOT_COUNT)
            for outcome in (False, True)
        ]
        refresh_responses = model.chat_complete_messages_batched(
            [
                refresh_messages(
                    initial_scene,
                    hypotheses,
                    roots[root_index],
                    outcome,
                )
                for root_index, outcome in interventions
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        raw["branch_refreshes"] = {
            branch_key(root_index, outcome): response
            for (root_index, outcome), response in zip(
                interventions, refresh_responses, strict=True
            )
        }
        branches = {
            key: parse_particle_population(
                response,
                allow_duplicate_asts=allow_duplicate_particles,
            )
            for key, response in raw["branch_refreshes"].items()
        }
        pathways = build_pathways(
            initial_scene=initial_scene,
            initial_hypotheses=hypotheses,
            initial_weights=initial_weights,
            roots=roots,
            branches=branches,
            pool=pool,
        )
        scorer_response = model.chat_complete_messages_batched(
            [
                scorer_messages(
                    pathways, particle_semantics=particle_semantics
                )
            ],
            temperature=0.0,
            block_size=1,
            max_new_tokens=MAX_NEW_TOKENS,
        )[0]
        raw["scorer"] = scorer_response
        scorer_scores = parse_scorer(scorer_response)
        _checkpoint(raw_checkpoint_path, raw)
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_checkpoint_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc
    payload = analyze_after_freeze(
        official_case=official_case,
        initial_scene=initial_scene,
        hypotheses=hypotheses,
        roots=roots,
        root_selection=root_selection,
        branches=branches,
        pathways=pathways,
        scorer_scores=scorer_scores,
        pool=pool,
        usage=usage,
        task_name=task_name,
        interface_version=interface_version,
        selection_seed=selection_seed,
        particle_semantics=particle_semantics,
        audit_mode=audit_mode,
    )
    payload["protocol"]["scene_pool_sha256"] = hashlib.sha256(
        _canonical_json(pool).encode("utf-8")
    ).hexdigest()
    payload["protocol"]["private_raw_sha256"] = _sha256(raw_checkpoint_path)
    return payload


def run_cli(
    *,
    interface_version: str = INTERFACE_VERSION,
    task_name: str = TASK_NAME,
    selection_seed: int = SELECTION_SEED,
    allow_duplicate_particles: bool = False,
    audit_mode: str = "random_plus_official",
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.25
    config.openrouter_run_budget_usd = COST_CAP_USD
    config.openrouter_concurrency = 8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_gate(
            config,
            source_dir=args.source_dir,
            raw_checkpoint_path=raw_path,
            task_name=task_name,
            interface_version=interface_version,
            selection_seed=selection_seed,
            allow_duplicate_particles=allow_duplicate_particles,
            audit_mode=audit_mode,
        )
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": interface_version,
            "task_name": task_name,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = _sha256(raw_path)
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path = args.output_dir / "SMOKE.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output_path),
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run_cli()
