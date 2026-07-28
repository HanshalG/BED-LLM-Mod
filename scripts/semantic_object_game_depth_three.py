#!/usr/bin/env python3
"""One-tree mechanics gate for semantic, LLM-native depth-three BED."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_oscillator_belief_smoke import (
    checkpoint,
    strict_json_object,
)
from scripts.number_game_predictive_risk_replication import (
    SeededStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "semantic-object-game-depth-three-mechanics-1"
PLANNING_MODEL_ID = "openai/gpt-5.4-mini"
TARGET_MODEL_ID = "google/gemini-2.5-flash"
TREE_SEED = 36000
VALIDATION_SEEDS = tuple(range(36100, 36104))
ENDPOINT_SEEDS = tuple(range(36200, 36208))
TEMPERATURE = 0.7
NUM_PROPOSALS = 16
NUM_ROOTS = 6
NUM_EIG_ROOTS = 4
MIN_MEMBERS = 3
MAX_MEMBERS = 29
MIN_INITIAL_VALID = 12
MIN_FIRST_VALID = 6
MIN_SECOND_VALID = 4
MIN_TARGET_VALID = 12
EXPECTED_REQUESTS = 49
MAX_TOKENS = 8192
RUN_BUDGET_USD = 0.75
PROJECTED_PLANNING_COST_USD = 0.35
PROJECTED_TARGET_COST_USD = 0.15

OBJECTS = (
    ("apple", "apple"),
    ("banana", "banana"),
    ("carrot", "carrot"),
    ("bread", "bread"),
    ("cheese", "cheese"),
    ("salmon", "salmon"),
    ("dog", "dog"),
    ("dolphin", "dolphin"),
    ("eagle", "eagle"),
    ("ant", "ant"),
    ("oak_tree", "oak tree"),
    ("cactus", "cactus"),
    ("rose", "rose"),
    ("mushroom", "mushroom"),
    ("hammer", "hammer"),
    ("scissors", "scissors"),
    ("compass", "compass"),
    ("thermometer", "thermometer"),
    ("bicycle", "bicycle"),
    ("canoe", "canoe"),
    ("helicopter", "helicopter"),
    ("train", "train"),
    ("violin", "violin"),
    ("drum", "drum"),
    ("flute", "flute"),
    ("piano", "piano"),
    ("smartphone", "smartphone"),
    ("camera", "camera"),
    ("refrigerator", "refrigerator"),
    ("umbrella", "umbrella"),
    ("book", "book"),
    ("candle", "candle"),
)
OBJECT_IDS = tuple(item[0] for item in OBJECTS)
OBJECT_NAMES = dict(OBJECTS)
OBJECT_INDEX = {object_id: index for index, object_id in enumerate(OBJECT_IDS)}


@dataclass(frozen=True)
class SemanticHypothesis:
    name: str
    description: str
    extension: tuple[bool, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "members": [
                object_id
                for object_id, included in zip(
                    OBJECT_IDS, self.extension, strict=True
                )
                if included
            ],
            "extension_sha256": hashlib.sha256(
                bytes(self.extension)
            ).hexdigest(),
        }


def proposal_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_object_concepts",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["concepts"],
                "properties": {
                    "concepts": {
                        "type": "array",
                        "minItems": NUM_PROPOSALS,
                        "maxItems": NUM_PROPOSALS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "name",
                                "description",
                                "members",
                            ],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 80,
                                },
                                "description": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 240,
                                },
                                "members": {
                                    "type": "array",
                                    "minItems": MIN_MEMBERS,
                                    "maxItems": MAX_MEMBERS,
                                    "uniqueItems": True,
                                    "items": {
                                        "type": "string",
                                        "enum": list(OBJECT_IDS),
                                    },
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def _object_catalogue() -> str:
    return "\n".join(
        f"- {object_id}: {display_name}"
        for object_id, display_name in OBJECTS
    )


def initial_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You generate diverse semantic hypotheses for Bayesian active "
                "concept learning. Return only the requested strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "A hidden concept divides the fixed object universe below. "
                f"Propose exactly {NUM_PROPOSALS} distinct, human-plausible "
                "concepts. Each concept must be expressible as one coherent "
                "semantic property, not an arbitrary list, exception rule, or "
                "memorized subset. Include every object that satisfies your "
                "property and no others. Use only exact object IDs. Concepts "
                f"must contain {MIN_MEMBERS}--{MAX_MEMBERS} objects and should "
                "span taxonomic, functional, physical, ecological, cultural, "
                "and relational properties.\n\nObject universe:\n"
                f"{_object_catalogue()}"
            ),
        },
    ]


def history_messages(
    observations: Sequence[tuple[int, bool]],
) -> list[dict[str, str]]:
    if not observations:
        return initial_messages()
    lines = "\n".join(
        f"- Is {OBJECT_IDS[index]} in the concept? "
        f"{'YES' if label else 'NO'}."
        for index, label in observations
    )
    return [
        {
            "role": "system",
            "content": (
                "You regenerate diverse semantic hypotheses after active "
                "concept queries. Return only the requested strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Observed labels:\n{lines}\n\n"
                f"Propose exactly {NUM_PROPOSALS} distinct, human-plausible "
                "concepts consistent with every label. Each concept must be "
                "one coherent semantic property, not an arbitrary list, "
                "exception rule, or encoding of the observations. Include "
                "every object satisfying the property and no others. Use only "
                f"exact IDs and {MIN_MEMBERS}--{MAX_MEMBERS} members.\n\n"
                f"Object universe:\n{_object_catalogue()}"
            ),
        },
    ]


def parse_proposals(
    response: str,
    *,
    observations: Sequence[tuple[int, bool]] = (),
) -> tuple[list[SemanticHypothesis], dict[str, Any]]:
    value = strict_json_object(response, label="semantic object concepts")
    if set(value) != {"concepts"}:
        raise ValueError("concept response has the wrong top-level fields")
    items = value["concepts"]
    if not isinstance(items, list) or len(items) != NUM_PROPOSALS:
        raise ValueError(f"response must contain exactly {NUM_PROPOSALS} items")
    accepted = []
    seen_extensions = set()
    rejected = {
        "wrong_fields": 0,
        "invalid_text": 0,
        "invalid_members": 0,
        "inconsistent": 0,
        "duplicate_extension": 0,
    }
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "description",
            "members",
        }:
            rejected["wrong_fields"] += 1
            continue
        name = item["name"]
        description = item["description"]
        members = item["members"]
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(description, str)
            or not description.strip()
        ):
            rejected["invalid_text"] += 1
            continue
        if (
            not isinstance(members, list)
            or not MIN_MEMBERS <= len(members) <= MAX_MEMBERS
            or len(set(members)) != len(members)
            or any(member not in OBJECT_INDEX for member in members)
        ):
            rejected["invalid_members"] += 1
            continue
        member_set = set(members)
        extension = tuple(
            object_id in member_set for object_id in OBJECT_IDS
        )
        if any(
            extension[index] != label for index, label in observations
        ):
            rejected["inconsistent"] += 1
            continue
        if extension in seen_extensions:
            rejected["duplicate_extension"] += 1
            continue
        seen_extensions.add(extension)
        accepted.append(
            SemanticHypothesis(
                name=name.strip(),
                description=description.strip(),
                extension=extension,
            )
        )
    return accepted, {
        "raw_count": len(items),
        "valid_unique_count": len(accepted),
        "rejected": rejected,
    }


def binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log(probability) - (
        1.0 - probability
    ) * math.log(1.0 - probability)


def query_eig(
    support: Sequence[SemanticHypothesis],
    query: int,
) -> float:
    if not support:
        return 0.0
    return binary_entropy(
        sum(hypothesis.extension[query] for hypothesis in support)
        / len(support)
    )


def best_query(
    support: Sequence[SemanticHypothesis],
    *,
    excluded: Iterable[int] = (),
) -> tuple[int, float]:
    excluded_set = set(excluded)
    candidates = [
        (query_eig(support, query), query)
        for query in range(len(OBJECT_IDS))
        if query not in excluded_set
    ]
    if not candidates:
        raise ValueError("no unqueried semantic objects remain")
    eig, query = max(candidates, key=lambda item: (item[0], -item[1]))
    return query, eig


def fixed_depth_two_score(
    support: Sequence[SemanticHypothesis],
    root: int,
) -> float:
    if not support:
        return 0.0
    future = 0.0
    for label in (False, True):
        branch = [
            hypothesis
            for hypothesis in support
            if hypothesis.extension[root] == label
        ]
        if branch:
            future += len(branch) / len(support) * best_query(
                branch, excluded=(root,)
            )[1]
    return query_eig(support, root) + future


def candidate_roots(
    support: Sequence[SemanticHypothesis],
    *,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    domain = range(len(OBJECT_IDS))
    immediate = {query: query_eig(support, query) for query in domain}
    fixed = {
        query: fixed_depth_two_score(support, query) for query in domain
    }
    myopic = max(domain, key=lambda query: (immediate[query], -query))
    fixed_best = max(domain, key=lambda query: (fixed[query], -query))
    selected = [myopic]
    if fixed_best not in selected:
        selected.append(fixed_best)
    signatures = {
        tuple(h.extension[root] for h in support) for root in selected
    }
    for query in sorted(domain, key=lambda item: (-immediate[item], item)):
        signature = tuple(h.extension[query] for h in support)
        if (
            query not in selected
            and signature not in signatures
            and len(selected) < NUM_EIG_ROOTS
        ):
            selected.append(query)
            signatures.add(signature)
    remaining = [query for query in domain if query not in selected]
    random.Random(seed).shuffle(remaining)
    selected.extend(remaining[: NUM_ROOTS - len(selected)])
    if len(selected) != NUM_ROOTS:
        raise ValueError("could not build distinct semantic root candidates")
    return selected, {
        "myopic_root": myopic,
        "fixed_depth_two_root": fixed_best,
        "immediate_eig_nats": {
            OBJECT_IDS[root]: immediate[root] for root in selected
        },
        "fixed_depth_two_nats": {
            OBJECT_IDS[root]: fixed[root] for root in selected
        },
    }


def retain_parent_hypotheses(
    *,
    parent_support: Sequence[SemanticHypothesis],
    generated_support: Sequence[SemanticHypothesis],
    query: int,
    label: bool,
) -> tuple[list[SemanticHypothesis], dict[str, int]]:
    retained = [
        hypothesis
        for hypothesis in parent_support
        if hypothesis.extension[query] == label
    ]
    merged = []
    seen = set()
    generated_unique = 0
    for source, hypotheses in (
        ("generated", generated_support),
        ("retained", retained),
    ):
        for hypothesis in hypotheses:
            if hypothesis.extension in seen:
                continue
            seen.add(hypothesis.extension)
            merged.append(hypothesis)
            if source == "generated":
                generated_unique += 1
    return merged, {
        "generated_unique_count": generated_unique,
        "retained_parent_consistent_count": len(retained),
        "retained_parent_novel_count": len(merged) - generated_unique,
        "merged_unique_count": len(merged),
    }


def _terminal_row(
    *,
    target: SemanticHypothesis,
    queried: set[int],
    survivors: Sequence[SemanticHypothesis],
) -> dict[str, Any]:
    if not survivors:
        return {
            "posterior_predictive_brier": 1.0,
            "best_hamming_error": 1.0,
            "truth_extension_covered": False,
            "survivor_count": 0,
        }
    unqueried = [
        index for index in range(len(OBJECT_IDS)) if index not in queried
    ]
    probabilities = {
        index: sum(h.extension[index] for h in survivors) / len(survivors)
        for index in unqueried
    }
    brier = sum(
        (
            probabilities[index] - float(target.extension[index])
        )
        ** 2
        for index in unqueried
    ) / len(unqueried)
    hamming = min(
        sum(
            left != right
            for left, right in zip(
                hypothesis.extension, target.extension, strict=True
            )
        )
        / len(OBJECT_IDS)
        for hypothesis in survivors
    )
    return {
        "posterior_predictive_brier": brier,
        "best_hamming_error": hamming,
        "truth_extension_covered": any(
            hypothesis.extension == target.extension
            for hypothesis in survivors
        ),
        "survivor_count": len(survivors),
    }


def evaluate_depth_two_root(
    *,
    root: int,
    target: SemanticHypothesis,
    first_branches: dict[
        tuple[int, bool], Sequence[SemanticHypothesis]
    ],
) -> dict[str, Any]:
    first_label = target.extension[root]
    support = list(first_branches[(root, first_label)])
    second_query, _ = best_query(support, excluded=(root,))
    second_label = target.extension[second_query]
    survivors = [
        hypothesis
        for hypothesis in support
        if hypothesis.extension[second_query] == second_label
    ]
    return {
        "root": root,
        "second_query": second_query,
        **_terminal_row(
            target=target,
            queried={root, second_query},
            survivors=survivors,
        ),
    }


def evaluate_depth_three_root(
    *,
    root: int,
    target: SemanticHypothesis,
    first_branches: dict[
        tuple[int, bool], Sequence[SemanticHypothesis]
    ],
    second_branches: dict[
        tuple[int, bool, int, bool], Sequence[SemanticHypothesis]
    ],
) -> dict[str, Any]:
    first_label = target.extension[root]
    first_support = list(first_branches[(root, first_label)])
    second_query, _ = best_query(first_support, excluded=(root,))
    second_label = target.extension[second_query]
    second_support = list(
        second_branches[
            (root, first_label, second_query, second_label)
        ]
    )
    third_query, _ = best_query(
        second_support, excluded=(root, second_query)
    )
    third_label = target.extension[third_query]
    survivors = [
        hypothesis
        for hypothesis in second_support
        if hypothesis.extension[third_query] == third_label
    ]
    return {
        "root": root,
        "second_query": second_query,
        "third_query": third_query,
        **_terminal_row(
            target=target,
            queried={root, second_query, third_query},
            survivors=survivors,
        ),
    }


def risk_scores(
    *,
    roots: Sequence[int],
    targets: Sequence[SemanticHypothesis],
    first_branches: dict[
        tuple[int, bool], Sequence[SemanticHypothesis]
    ],
    second_branches: dict[
        tuple[int, bool, int, bool], Sequence[SemanticHypothesis]
    ],
) -> tuple[dict[int, float], dict[int, float]]:
    if not targets:
        raise ValueError("cross-fit target support is empty")
    depth_two = {}
    depth_three = {}
    for root in roots:
        rows_two = [
            evaluate_depth_two_root(
                root=root,
                target=target,
                first_branches=first_branches,
            )
            for target in targets
        ]
        rows_three = [
            evaluate_depth_three_root(
                root=root,
                target=target,
                first_branches=first_branches,
                second_branches=second_branches,
            )
            for target in targets
        ]
        depth_two[root] = sum(
            row["posterior_predictive_brier"] for row in rows_two
        ) / len(rows_two)
        depth_three[root] = sum(
            row["posterior_predictive_brier"] for row in rows_three
        ) / len(rows_three)
    return depth_two, depth_three


def choose_lowest_risk(scores: dict[int, float]) -> int:
    return min(scores, key=lambda root: (scores[root], root))


def endpoint_metrics(
    *,
    root: int,
    depth: int,
    targets: Sequence[SemanticHypothesis],
    first_branches: dict[
        tuple[int, bool], Sequence[SemanticHypothesis]
    ],
    second_branches: dict[
        tuple[int, bool, int, bool], Sequence[SemanticHypothesis]
    ],
) -> dict[str, Any]:
    rows = []
    for target in targets:
        row = (
            evaluate_depth_three_root(
                root=root,
                target=target,
                first_branches=first_branches,
                second_branches=second_branches,
            )
            if depth == 3
            else evaluate_depth_two_root(
                root=root,
                target=target,
                first_branches=first_branches,
            )
        )
        rows.append(row)
    return {
        "root": OBJECT_IDS[root],
        "mean_brier": sum(
            row["posterior_predictive_brier"] for row in rows
        )
        / len(rows),
        "mean_hamming": sum(row["best_hamming_error"] for row in rows)
        / len(rows),
        "coverage": sum(row["truth_extension_covered"] for row in rows)
        / len(rows),
        "target_count": len(rows),
    }


def _adapter(
    *,
    model: str,
    run_id: str,
    output_dir: Path,
    request_seed: int,
    concurrency: int,
    projected_cost: float,
) -> SeededStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=500.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=projected_cost,
        openrouter_concurrency=concurrency,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return SeededStructuredAdapter(
        ModelSpec(model=model, backend="openrouter", max_model_len=65536),
        config,
        request_seed=request_seed,
    )


def _usage(adapters: Sequence[SeededStructuredAdapter]) -> dict[str, Any]:
    snapshots = [adapter.usage_snapshot() for adapter in adapters]
    mapping = {
        "adapter_requests": "adapter_requests",
        "http_attempts": "http_attempts",
        "retry_count": "retry_count",
        "provider_error_retries": "provider_error_retries",
        "reasoning_tokens": "adapter_reasoning_tokens",
        "forced_exits": "forced_exits",
        "run_cost_usd": "adapter_cost_usd",
    }
    result = {}
    for output, source in mapping.items():
        value = sum(float(snapshot.get(source, 0) or 0) for snapshot in snapshots)
        result[output] = value if output == "run_cost_usd" else int(value)
    return result


def _generate_independent_supports(
    *,
    seeds: Sequence[int],
    output_dir: Path,
    run_id: str,
) -> tuple[
    list[list[SemanticHypothesis]],
    list[dict[str, Any]],
    list[str],
    list[SeededStructuredAdapter],
]:
    adapters = [
        _adapter(
            model=TARGET_MODEL_ID,
            run_id=run_id,
            output_dir=output_dir,
            request_seed=seed,
            concurrency=1,
            projected_cost=PROJECTED_TARGET_COST_USD,
        )
        for seed in seeds
    ]

    def request(adapter: SeededStructuredAdapter) -> str:
        return adapter.chat_complete_messages_batched_structured(
            [initial_messages()],
            temperature=TEMPERATURE,
            block_size=1,
            response_format=proposal_response_format(),
            max_new_tokens=MAX_TOKENS,
        )[0]

    with ThreadPoolExecutor(max_workers=len(adapters)) as executor:
        responses = list(executor.map(request, adapters))
    supports = []
    diagnostics = []
    for response in responses:
        support, diagnostic = parse_proposals(response)
        supports.append(support)
        diagnostics.append(diagnostic)
    return supports, diagnostics, responses, adapters


def mechanics_gates(
    *,
    usage: dict[str, Any],
    initial: Sequence[SemanticHypothesis],
    first_diagnostics: dict[str, dict[str, Any]],
    second_diagnostics: dict[str, dict[str, Any]],
    validation_supports: Sequence[Sequence[SemanticHypothesis]],
    endpoint_supports: Sequence[Sequence[SemanticHypothesis]],
    roots: Sequence[int],
    depth_two_scores: dict[int, float],
    depth_three_scores: dict[int, float],
) -> dict[str, bool]:
    return {
        "exact_49_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "initial_has_at_least_12_valid_concepts": (
            len(initial) >= MIN_INITIAL_VALID
        ),
        "all_first_branches_have_at_least_6_concepts": all(
            row["valid_unique_count"] >= MIN_FIRST_VALID
            for row in first_diagnostics.values()
        ),
        "all_second_branches_have_at_least_4_concepts": all(
            row["valid_unique_count"] >= MIN_SECOND_VALID
            for row in second_diagnostics.values()
        ),
        "all_validation_supports_have_at_least_12_concepts": all(
            len(support) >= MIN_TARGET_VALID
            for support in validation_supports
        ),
        "all_endpoint_supports_have_at_least_12_concepts": all(
            len(support) >= MIN_TARGET_VALID for support in endpoint_supports
        ),
        "all_six_root_scores_are_finite": (
            len(depth_two_scores) == len(depth_three_scores) == len(roots)
            and all(
                math.isfinite(value)
                for value in (
                    *depth_two_scores.values(),
                    *depth_three_scores.values(),
                )
            )
        ),
        "depth_three_risk_has_nonzero_range": (
            max(depth_three_scores.values())
            - min(depth_three_scores.values())
            > 1e-6
        ),
        "depth_two_and_depth_three_select_different_roots": (
            choose_lowest_risk(depth_two_scores)
            != choose_lowest_risk(depth_three_scores)
        ),
    }


def run_mechanics(
    *,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    planning = _adapter(
        model=PLANNING_MODEL_ID,
        run_id=run_id,
        output_dir=output_dir,
        request_seed=TREE_SEED,
        concurrency=32,
        projected_cost=PROJECTED_PLANNING_COST_USD,
    )
    raw: dict[str, Any] = {
        "initial": None,
        "first": [],
        "second": [],
        "validation": [],
        "endpoint": [],
    }
    try:
        initial_response = planning.chat_complete_messages_batched_structured(
            [initial_messages()],
            temperature=TEMPERATURE,
            block_size=1,
            response_format=proposal_response_format(),
            max_new_tokens=MAX_TOKENS,
        )[0]
        raw["initial"] = initial_response
        checkpoint(raw_path, raw)
        initial, initial_diagnostic = parse_proposals(initial_response)
        if len(initial) < MIN_INITIAL_VALID:
            raise ValueError("initial semantic support is too small")
        roots, root_metadata = candidate_roots(initial, seed=TREE_SEED)

        first_keys = [
            (root, label) for root in roots for label in (False, True)
        ]
        first_responses = planning.chat_complete_messages_batched_structured(
            [history_messages((key,)) for key in first_keys],
            temperature=TEMPERATURE,
            block_size=len(first_keys),
            response_format=proposal_response_format(),
            max_new_tokens=MAX_TOKENS,
        )
        raw["first"] = [
            {"root": root, "label": label, "response": response}
            for (root, label), response in zip(
                first_keys, first_responses, strict=True
            )
        ]
        checkpoint(raw_path, raw)
        first_branches = {}
        first_diagnostics = {}
        for key, response in zip(first_keys, first_responses, strict=True):
            generated, diagnostic = parse_proposals(
                response, observations=(key,)
            )
            support, retention = retain_parent_hypotheses(
                parent_support=initial,
                generated_support=generated,
                query=key[0],
                label=key[1],
            )
            first_branches[key] = support
            first_diagnostics[f"{key[0]}:{int(key[1])}"] = {
                **diagnostic,
                **retention,
                "valid_unique_count": len(support),
            }

        second_keys = []
        for root, first_label in first_keys:
            second_query, _ = best_query(
                first_branches[(root, first_label)],
                excluded=(root,),
            )
            for second_label in (False, True):
                second_keys.append(
                    (root, first_label, second_query, second_label)
                )
        second_responses = planning.chat_complete_messages_batched_structured(
            [
                history_messages(
                    ((root, first_label), (second_query, second_label))
                )
                for root, first_label, second_query, second_label in second_keys
            ],
            temperature=TEMPERATURE,
            block_size=len(second_keys),
            response_format=proposal_response_format(),
            max_new_tokens=MAX_TOKENS,
        )
        raw["second"] = [
            {
                "root": root,
                "first_label": first_label,
                "second_query": second_query,
                "second_label": second_label,
                "response": response,
            }
            for (
                root,
                first_label,
                second_query,
                second_label,
            ), response in zip(second_keys, second_responses, strict=True)
        ]
        checkpoint(raw_path, raw)
        second_branches = {}
        second_diagnostics = {}
        for key, response in zip(second_keys, second_responses, strict=True):
            observations = ((key[0], key[1]), (key[2], key[3]))
            generated, diagnostic = parse_proposals(
                response, observations=observations
            )
            support, retention = retain_parent_hypotheses(
                parent_support=first_branches[(key[0], key[1])],
                generated_support=generated,
                query=key[2],
                label=key[3],
            )
            second_branches[key] = support
            second_diagnostics[
                f"{key[0]}:{int(key[1])}:{key[2]}:{int(key[3])}"
            ] = {
                **diagnostic,
                **retention,
                "valid_unique_count": len(support),
            }

        all_target_seeds = (*VALIDATION_SEEDS, *ENDPOINT_SEEDS)
        (
            target_supports,
            target_diagnostics,
            target_responses,
            target_adapters,
        ) = _generate_independent_supports(
            seeds=all_target_seeds,
            output_dir=output_dir,
            run_id=run_id,
        )
        raw["validation"] = [
            {"seed": seed, "response": response}
            for seed, response in zip(
                VALIDATION_SEEDS,
                target_responses[: len(VALIDATION_SEEDS)],
                strict=True,
            )
        ]
        raw["endpoint"] = [
            {"seed": seed, "response": response}
            for seed, response in zip(
                ENDPOINT_SEEDS,
                target_responses[len(VALIDATION_SEEDS) :],
                strict=True,
            )
        ]
        checkpoint(raw_path, raw)
        validation_supports = target_supports[: len(VALIDATION_SEEDS)]
        endpoint_supports = target_supports[len(VALIDATION_SEEDS) :]
        validation_targets = [
            hypothesis
            for support in validation_supports
            for hypothesis in support
        ]
        endpoint_targets = [
            hypothesis
            for support in endpoint_supports
            for hypothesis in support
        ]
        depth_two_scores, depth_three_scores = risk_scores(
            roots=roots,
            targets=validation_targets,
            first_branches=first_branches,
            second_branches=second_branches,
        )
        depth_two_root = choose_lowest_risk(depth_two_scores)
        depth_three_root = choose_lowest_risk(depth_three_scores)
        endpoint_two = endpoint_metrics(
            root=depth_two_root,
            depth=2,
            targets=endpoint_targets,
            first_branches=first_branches,
            second_branches=second_branches,
        )
        endpoint_three = endpoint_metrics(
            root=depth_three_root,
            depth=3,
            targets=endpoint_targets,
            first_branches=first_branches,
            second_branches=second_branches,
        )
        usage = _usage([planning, *target_adapters])
        gates = mechanics_gates(
            usage=usage,
            initial=initial,
            first_diagnostics=first_diagnostics,
            second_diagnostics=second_diagnostics,
            validation_supports=validation_supports,
            endpoint_supports=endpoint_supports,
            roots=roots,
            depth_two_scores=depth_two_scores,
            depth_three_scores=depth_three_scores,
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(gates.values()) else "gated_null",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "planning_model": PLANNING_MODEL_ID,
                "target_model": TARGET_MODEL_ID,
                "tree_seed": TREE_SEED,
                "validation_seeds": list(VALIDATION_SEEDS),
                "endpoint_seeds": list(ENDPOINT_SEEDS),
                "temperature": TEMPERATURE,
                "num_objects": len(OBJECT_IDS),
                "num_proposals": NUM_PROPOSALS,
                "num_roots": NUM_ROOTS,
                "expected_requests": EXPECTED_REQUESTS,
                "run_budget_usd": RUN_BUDGET_USD,
                "efficacy_is_mechanics_only": True,
            },
            "usage": usage,
            "gates": gates,
            "mechanics": {
                "objects": list(OBJECT_IDS),
                "roots": [OBJECT_IDS[root] for root in roots],
                "root_metadata": root_metadata,
                "initial_diagnostic": initial_diagnostic,
                "first_diagnostics": first_diagnostics,
                "second_diagnostics": second_diagnostics,
                "target_diagnostics": target_diagnostics,
                "depth_two_validation_risk": {
                    OBJECT_IDS[root]: score
                    for root, score in depth_two_scores.items()
                },
                "depth_three_validation_risk": {
                    OBJECT_IDS[root]: score
                    for root, score in depth_three_scores.items()
                },
                "depth_two_root": OBJECT_IDS[depth_two_root],
                "depth_three_root": OBJECT_IDS[depth_three_root],
            },
            "endpoint_descriptive": {
                "depth_two": endpoint_two,
                "depth_three": endpoint_three,
                "depth_three_minus_depth_two_brier": (
                    endpoint_three["mean_brier"] - endpoint_two["mean_brier"]
                ),
            },
            "supports": {
                "initial": [hypothesis.public_dict() for hypothesis in initial],
                "validation": [
                    [hypothesis.public_dict() for hypothesis in support]
                    for support in validation_supports
                ],
                "endpoint": [
                    [hypothesis.public_dict() for hypothesis in support]
                    for support in endpoint_supports
                ],
            },
            "raw_responses_sha256": hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
        }
        checkpoint(output_dir / "RESULT.json", result)
        return result
    except Exception as exc:
        checkpoint(
            output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed",
                "interface_version": INTERFACE_VERSION,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_responses_sha256": (
                    hashlib.sha256(raw_path.read_bytes()).hexdigest()
                    if raw_path.exists()
                    else None
                ),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = run_mechanics(
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "usage": result["usage"],
                "gates": result["gates"],
                "endpoint_descriptive": result["endpoint_descriptive"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
