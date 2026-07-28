#!/usr/bin/env python3
"""Test depth-two BED over an LLM's path-dependent Number Game support."""

from __future__ import annotations

import argparse
import ast
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
from scripts.discoverphysics_dark_matter_structured_replication_v3 import (
    DefaultRoutingStructuredAdapter,
)
from scripts.discoverphysics_oscillator_belief_smoke import (
    SmokeExecutionError,
    checkpoint,
    strict_json_object,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-generator-aware-bed-1"
MODEL_ID = "google/gemini-2.5-flash"
DOMAIN = tuple(range(101))
NUM_PROPOSALS = 24
NUM_ROOTS = 8
NUM_EIG_ROOTS = 4
NUM_PTS_ROOTS = 2
EXPECTED_REQUESTS = 1 + 2 * NUM_ROOTS
SEED = 26068
MAX_TOKENS = 4200
RUN_BUDGET_USD = 0.50
PROJECTED_COST_USD = 0.12
MIN_INITIAL_VALID = 16
MIN_BRANCH_VALID = 8
MIN_LOOKAHEAD_ADVANTAGE_NATS = 0.01
MIN_LEAVE_ONE_OUT_AGREEMENT = 0.75


TARGET_EXPRESSIONS = {
    "square_numbers": "is_square(n)",
    "multiples_of_4": "divisible(n, 4)",
    "odd_numbers": "n % 2 == 1",
    "powers_of_2": "is_power_of_two(n)",
    "even_numbers_below_30": "divisible(n, 2) and n < 30",
    "multiples_of_3_or_7": "divisible(n, 3) or divisible(n, 7)",
    "odd_multiples_of_3": "divisible(n, 3) and n % 2 == 1",
    "numbers_ending_in_6": "ends_with(n, 6)",
    "digit_sum_below_8": "digit_sum(n) < 8",
    "remainder_5_mod_9": "n % 9 == 5",
    "one_less_than_a_prime": "is_prime(n + 1)",
    "twice_a_square_minus_2": "is_square((n + 2) // 2) and (n + 2) % 2 == 0",
}


@dataclass(frozen=True)
class RuleHypothesis:
    name: str
    expression: str
    extension: tuple[bool, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "positive_count": sum(self.extension),
            "extension_sha256": hashlib.sha256(
                bytes(self.extension)
            ).hexdigest(),
        }


def divisible(n: int, divisor: int) -> bool:
    return divisor != 0 and n % divisor == 0


def is_square(n: int) -> bool:
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for divisor in range(2, math.isqrt(n) + 1):
        if n % divisor == 0:
            return False
    return True


def digit_sum(n: int) -> int:
    return sum(int(digit) for digit in str(abs(n)))


def ends_with(n: int, digit: int) -> bool:
    return 0 <= digit <= 9 and abs(n) % 10 == digit


_HELPERS = {
    "divisible": divisible,
    "is_square": is_square,
    "is_power_of_two": is_power_of_two,
    "is_prime": is_prime,
    "digit_sum": digit_sum,
    "ends_with": ends_with,
}
_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.Mod,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Constant,
)


def compile_expression(expression: str) -> tuple[bool, ...]:
    if not expression.strip() or len(expression) > 240:
        raise ValueError("expression is empty or too long")
    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 80:
        raise ValueError("expression is too complex")
    for node in nodes:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(f"forbidden syntax {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"n", *_HELPERS}:
            raise ValueError(f"unknown name {node.id!r}")
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _HELPERS
                or node.keywords
            ):
                raise ValueError("only documented helper calls are allowed")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                continue
            if not isinstance(node.value, int) or abs(node.value) > 1000:
                raise ValueError("only bounded integer constants are allowed")
        if isinstance(node, (ast.FloorDiv, ast.Mod)):
            parent = next(
                (
                    candidate
                    for candidate in nodes
                    if isinstance(candidate, ast.BinOp)
                    and candidate.op is node
                ),
                None,
            )
            if (
                parent is not None
                and isinstance(parent.right, ast.Constant)
                and parent.right.value == 0
            ):
                raise ValueError("division by zero")
    code = compile(tree, "<number-game-rule>", "eval")
    extension = []
    for number in DOMAIN:
        try:
            value = eval(
                code,
                {"__builtins__": {}, **_HELPERS},
                {"n": number},
            )
        except Exception as exc:
            raise ValueError(f"rule failed on n={number}: {exc}") from exc
        if not isinstance(value, bool):
            raise ValueError("expression must return a boolean")
        extension.append(value)
    if all(extension) or not any(extension):
        raise ValueError("constant rules are not informative")
    return tuple(extension)


def proposal_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "number_game_hypotheses",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": NUM_PROPOSALS,
                        "maxItems": NUM_PROPOSALS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "expression"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 100,
                                },
                                "expression": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 240,
                                },
                            },
                        },
                    }
                },
            },
        },
    }


_GRAMMAR = """Return predicates over integers n in 0..100.
Each expression must be a single Python expression returning bool.
Allowed syntax: and/or/not, integer arithmetic + - * // %, comparisons, n,
and only these helpers:
  divisible(n, k), is_square(n), is_power_of_two(n), is_prime(n),
  digit_sum(n), ends_with(n, digit).
Do not use lambdas, comprehensions, containers, indexing, attributes, imports,
or explicit lookup tables. Avoid singleton and near-singleton memorization.
Prefer concise, human-plausible rules, but include genuinely different simple,
modular, digit-based, bounded, and compositional possibilities."""


def initial_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You propose diverse executable hypotheses for Bayesian active "
                "concept learning. Return only the requested strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "This is the classic Number Game. No labels have been observed. "
                f"Propose exactly {NUM_PROPOSALS} distinct plausible concepts.\n\n"
                f"{_GRAMMAR}"
            ),
        },
    ]


def branch_messages(root: int, label: bool) -> list[dict[str, str]]:
    answer = "YES" if label else "NO"
    return [
        {
            "role": "system",
            "content": (
                "You rejuvenate executable hypotheses after an active Number "
                "Game query. Return only the requested strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The only observation is: Is {root} in the concept? {answer}.\n"
                f"Propose exactly {NUM_PROPOSALS} distinct hypotheses consistent "
                "with that observation. Generate coherent general rules rather "
                "than encoding the observation as an exception.\n\n"
                f"{_GRAMMAR}"
            ),
        },
    ]


def parse_proposals(
    response: str,
    *,
    observations: Sequence[tuple[int, bool]] = (),
) -> tuple[list[RuleHypothesis], dict[str, Any]]:
    value = strict_json_object(response, label="number-game hypotheses")
    if set(value) != {"hypotheses"}:
        raise ValueError("hypothesis response has the wrong top-level fields")
    items = value["hypotheses"]
    if not isinstance(items, list) or len(items) != NUM_PROPOSALS:
        raise ValueError(f"response must contain exactly {NUM_PROPOSALS} items")
    accepted: list[RuleHypothesis] = []
    seen_extensions: set[tuple[bool, ...]] = set()
    rejected = {
        "wrong_fields": 0,
        "invalid_name": 0,
        "invalid_expression": 0,
        "inconsistent": 0,
        "duplicate_extension": 0,
    }
    for item in items:
        if not isinstance(item, dict) or set(item) != {"name", "expression"}:
            rejected["wrong_fields"] += 1
            continue
        name = item["name"]
        expression = item["expression"]
        if not isinstance(name, str) or not name.strip():
            rejected["invalid_name"] += 1
            continue
        if not isinstance(expression, str):
            rejected["invalid_expression"] += 1
            continue
        try:
            extension = compile_expression(expression)
        except (SyntaxError, ValueError):
            rejected["invalid_expression"] += 1
            continue
        if any(extension[number] != label for number, label in observations):
            rejected["inconsistent"] += 1
            continue
        if extension in seen_extensions:
            rejected["duplicate_extension"] += 1
            continue
        seen_extensions.add(extension)
        accepted.append(
            RuleHypothesis(
                name=name.strip(),
                expression=expression.strip(),
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
    support: Sequence[RuleHypothesis],
    query: int,
) -> float:
    if not support:
        return 0.0
    positive = sum(hypothesis.extension[query] for hypothesis in support)
    return binary_entropy(positive / len(support))


def best_query(
    support: Sequence[RuleHypothesis],
    *,
    excluded: Iterable[int] = (),
) -> tuple[int, float]:
    excluded_set = set(excluded)
    candidates = [
        (query_eig(support, query), query)
        for query in DOMAIN
        if query not in excluded_set
    ]
    eig, query = max(candidates, key=lambda item: (item[0], -item[1]))
    return query, eig


def fixed_depth_two_score(
    support: Sequence[RuleHypothesis],
    root: int,
) -> float:
    immediate = query_eig(support, root)
    expected_future = 0.0
    for label in (False, True):
        branch = [
            hypothesis
            for hypothesis in support
            if hypothesis.extension[root] == label
        ]
        probability = len(branch) / len(support)
        if branch:
            expected_future += probability * best_query(
                branch, excluded=(root,)
            )[1]
    return immediate + expected_future


def fixed_depth_two_scores(
    support: Sequence[RuleHypothesis],
) -> dict[int, float]:
    return {
        root: fixed_depth_two_score(support, root)
        for root in DOMAIN
    }


def candidate_roots(
    support: Sequence[RuleHypothesis],
    *,
    seed: int = SEED,
) -> tuple[list[int], dict[str, Any]]:
    immediate = {query: query_eig(support, query) for query in DOMAIN}
    fixed = fixed_depth_two_scores(support)
    myopic = max(DOMAIN, key=lambda query: (immediate[query], -query))
    fixed_best = max(DOMAIN, key=lambda query: (fixed[query], -query))
    selected = [myopic]
    if fixed_best not in selected:
        selected.append(fixed_best)
    seen_signatures = {
        tuple(h.extension[root] for h in support) for root in selected
    }
    for query in sorted(DOMAIN, key=lambda q: (-immediate[q], q)):
        signature = tuple(h.extension[query] for h in support)
        if (
            query not in selected
            and signature not in seen_signatures
            and len(selected) < NUM_EIG_ROOTS
        ):
            selected.append(query)
            seen_signatures.add(signature)
    map_hypothesis = min(
        support,
        key=lambda hypothesis: (
            len(hypothesis.expression),
            hypothesis.expression,
        ),
    )
    positive_pool = [
        query
        for query in DOMAIN
        if map_hypothesis.extension[query] and query not in selected
    ]
    rng = random.Random(seed)
    rng.shuffle(positive_pool)
    pts_roots = positive_pool[:NUM_PTS_ROOTS]
    selected.extend(pts_roots)
    random_pool = [query for query in DOMAIN if query not in selected]
    rng.shuffle(random_pool)
    while len(selected) < NUM_ROOTS and random_pool:
        selected.append(random_pool.pop())
    if len(selected) != NUM_ROOTS:
        raise ValueError("could not construct eight distinct root candidates")
    return selected, {
        "myopic_root": myopic,
        "fixed_depth_two_root": fixed_best,
        "pts_roots": pts_roots,
        "random_roots": selected[NUM_EIG_ROOTS + NUM_PTS_ROOTS :],
        "immediate_eig_nats": {str(q): immediate[q] for q in selected},
        "fixed_depth_two_nats": {str(q): fixed[q] for q in selected},
    }


def merge_controlled_support(
    truth: RuleHypothesis,
    generated: Sequence[RuleHypothesis],
) -> list[RuleHypothesis]:
    merged = [truth]
    seen = {truth.extension}
    for hypothesis in generated:
        if hypothesis.extension not in seen:
            merged.append(hypothesis)
            seen.add(hypothesis.extension)
    return merged


def generator_aware_score(
    support: Sequence[RuleHypothesis],
    root: int,
    branches: dict[tuple[int, bool], Sequence[RuleHypothesis]],
) -> float:
    future = 0.0
    for truth in support:
        label = truth.extension[root]
        controlled = merge_controlled_support(
            truth, branches[(root, label)]
        )
        future += best_query(controlled, excluded=(root,))[1]
    return query_eig(support, root) + future / len(support)


def generator_aware_scores(
    support: Sequence[RuleHypothesis],
    roots: Sequence[int],
    branches: dict[tuple[int, bool], Sequence[RuleHypothesis]],
) -> dict[int, float]:
    return {
        root: generator_aware_score(support, root, branches)
        for root in roots
    }


def choose_from_scores(scores: dict[int, float]) -> int:
    return max(scores, key=lambda root: (scores[root], -root))


def leave_one_out_agreement(
    support: Sequence[RuleHypothesis],
    roots: Sequence[int],
    branches: dict[tuple[int, bool], Sequence[RuleHypothesis]],
    selected_root: int,
) -> dict[str, Any]:
    choices = []
    for index in range(len(support)):
        reduced = list(support[:index]) + list(support[index + 1 :])
        scores = generator_aware_scores(reduced, roots, branches)
        choices.append(choose_from_scores(scores))
    agreement = sum(choice == selected_root for choice in choices) / len(
        choices
    )
    return {"agreement": agreement, "choices": choices}


def hamming_error(
    left: Sequence[bool],
    right: Sequence[bool],
) -> float:
    return sum(a != b for a, b in zip(left, right, strict=True)) / len(left)


def evaluate_policy_root(
    *,
    policy: str,
    root: int,
    targets: dict[str, RuleHypothesis],
    branches: dict[tuple[int, bool], Sequence[RuleHypothesis]],
) -> dict[str, Any]:
    rows = []
    for target_name, target in targets.items():
        root_label = target.extension[root]
        branch = list(branches[(root, root_label)])
        second_query, second_eig = best_query(branch, excluded=(root,))
        second_label = target.extension[second_query]
        survivors = [
            hypothesis
            for hypothesis in branch
            if hypothesis.extension[second_query] == second_label
        ]
        exact_matches = [
            hypothesis
            for hypothesis in survivors
            if hypothesis.extension == target.extension
        ]
        if survivors:
            probabilities = [
                sum(hypothesis.extension[number] for hypothesis in survivors)
                / len(survivors)
                for number in DOMAIN
            ]
            brier = sum(
                (probability - float(target.extension[number])) ** 2
                for number, probability in enumerate(probabilities)
                if number not in {root, second_query}
            ) / (len(DOMAIN) - 2)
            best_error = min(
                hamming_error(hypothesis.extension, target.extension)
                for hypothesis in survivors
            )
        else:
            brier = 1.0
            best_error = 1.0
        rows.append(
            {
                "target": target_name,
                "root": root,
                "root_label": root_label,
                "second_query": second_query,
                "second_label": second_label,
                "second_eig_nats": second_eig,
                "branch_support_size": len(branch),
                "survivor_count": len(survivors),
                "truth_extension_covered": bool(exact_matches),
                "posterior_predictive_brier": brier,
                "best_hamming_error": best_error,
            }
        )
    return {
        "policy": policy,
        "root": root,
        "mean_posterior_predictive_brier": sum(
            row["posterior_predictive_brier"] for row in rows
        )
        / len(rows),
        "mean_best_hamming_error": sum(
            row["best_hamming_error"] for row in rows
        )
        / len(rows),
        "truth_extension_coverage_rate": sum(
            row["truth_extension_covered"] for row in rows
        )
        / len(rows),
        "mean_survivor_count": sum(row["survivor_count"] for row in rows)
        / len(rows),
        "targets": rows,
    }


def predictive_bayes_risk_scores(
    *,
    support: Sequence[RuleHypothesis],
    roots: Sequence[int],
    branches: dict[tuple[int, bool], Sequence[RuleHypothesis]],
) -> dict[int, dict[str, Any]]:
    targets = {
        f"particle_{index:02d}": hypothesis
        for index, hypothesis in enumerate(support)
    }
    return {
        root: evaluate_policy_root(
            policy="predictive_bayes_risk",
            root=root,
            targets=targets,
            branches=branches,
        )
        for root in roots
    }


def choose_predictive_bayes_risk_root(
    scores: dict[int, dict[str, Any]],
) -> int:
    return min(
        scores,
        key=lambda root: (
            scores[root]["mean_posterior_predictive_brier"],
            scores[root]["mean_best_hamming_error"],
            -scores[root]["truth_extension_coverage_rate"],
            root,
        ),
    )


def target_hypotheses() -> dict[str, RuleHypothesis]:
    return {
        name: RuleHypothesis(
            name=name,
            expression=expression,
            extension=compile_expression(expression),
        )
        for name, expression in TARGET_EXPRESSIONS.items()
    }


def _adapter(
    *,
    run_id: str,
    output_dir: Path,
) -> DefaultRoutingStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=200.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=16,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return DefaultRoutingStructuredAdapter(
        ModelSpec(
            model=MODEL_ID,
            backend="openrouter",
            max_model_len=65536,
        ),
        config,
    )


def mechanics_gates(
    *,
    initial: Sequence[RuleHypothesis],
    roots: Sequence[int],
    branch_diagnostics: dict[tuple[int, bool], dict[str, Any]],
    branches: dict[tuple[int, bool], Sequence[RuleHypothesis]],
    usage: dict[str, Any],
    selection: dict[str, Any],
    loo: dict[str, Any],
) -> dict[str, bool]:
    generated = selection["generator_aware_root"]
    myopic = selection["myopic_root"]
    fixed = selection["fixed_depth_two_root"]
    scores = selection["generator_aware_nats"]
    return {
        "exact_17_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "initial_has_at_least_16_valid_unique_rules": (
            len(initial) >= MIN_INITIAL_VALID
        ),
        "every_branch_has_at_least_8_valid_unique_rules": all(
            item["valid_unique_count"] >= MIN_BRANCH_VALID
            for item in branch_diagnostics.values()
        ),
        "every_root_has_label_distinct_generated_support": all(
            {hyp.extension for hyp in branches[(root, False)]}
            != {hyp.extension for hyp in branches[(root, True)]}
            for root in roots
        ),
        "generator_aware_root_differs_from_myopic": generated != myopic,
        "generator_aware_root_differs_from_fixed_depth_two": (
            generated != fixed
        ),
        "generator_aware_advantage_over_myopic_at_least_0_01_nats": (
            scores[str(generated)] - scores[str(myopic)]
            >= MIN_LOOKAHEAD_ADVANTAGE_NATS
        ),
        "leave_one_out_root_agreement_at_least_75_percent": (
            loo["agreement"] >= MIN_LEAVE_ONE_OUT_AGREEMENT
        ),
    }


def run_experiment(
    *,
    output_dir: Path,
    run_id: str,
    adapter: DefaultRoutingStructuredAdapter | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    adapter = adapter or _adapter(run_id=run_id, output_dir=output_dir)
    raw: dict[str, Any] = {"initial": None, "branches": []}
    try:
        initial_response = adapter.chat_complete_messages_batched_structured(
            [initial_messages()],
            temperature=0.0,
            block_size=1,
            response_format=proposal_response_format(),
            max_new_tokens=MAX_TOKENS,
        )[0]
        raw["initial"] = initial_response
        checkpoint(raw_path, raw)
        initial, initial_diagnostics = parse_proposals(initial_response)
        if len(initial) < MIN_INITIAL_VALID:
            raise ValueError(
                f"only {len(initial)} valid unique initial hypotheses"
            )
        roots, candidate_metadata = candidate_roots(initial)
        branch_keys = [
            (root, label)
            for root in roots
            for label in (False, True)
        ]
        responses = adapter.chat_complete_messages_batched_structured(
            [branch_messages(root, label) for root, label in branch_keys],
            temperature=0.0,
            block_size=len(branch_keys),
            response_format=proposal_response_format(),
            max_new_tokens=MAX_TOKENS,
        )
        raw["branches"] = [
            {"root": root, "label": label, "response": response}
            for (root, label), response in zip(
                branch_keys, responses, strict=True
            )
        ]
        checkpoint(raw_path, raw)
        branches: dict[tuple[int, bool], list[RuleHypothesis]] = {}
        branch_diagnostics: dict[tuple[int, bool], dict[str, Any]] = {}
        for (root, label), response in zip(
            branch_keys, responses, strict=True
        ):
            parsed, diagnostics = parse_proposals(
                response, observations=((root, label),)
            )
            branches[(root, label)] = parsed
            branch_diagnostics[(root, label)] = diagnostics
        generator_scores = generator_aware_scores(
            initial, roots, branches
        )
        generator_root = choose_from_scores(generator_scores)
        loo = leave_one_out_agreement(
            initial, roots, branches, generator_root
        )
        selection = {
            **candidate_metadata,
            "generator_aware_root": generator_root,
            "generator_aware_nats": {
                str(root): generator_scores[root] for root in roots
            },
        }
        usage = adapter.usage_snapshot()
        gates = mechanics_gates(
            initial=initial,
            roots=roots,
            branch_diagnostics=branch_diagnostics,
            branches=branches,
            usage=usage,
            selection=selection,
            loo=loo,
        )
        policies = {
            "generator_aware_depth_two": generator_root,
            "myopic_eig": selection["myopic_root"],
            "fixed_support_depth_two": selection["fixed_depth_two_root"],
            "deterministic_random": roots[-1],
        }
        targets = target_hypotheses()
        endpoint = {
            name: evaluate_policy_root(
                policy=name,
                root=root,
                targets=targets,
                branches=branches,
            )
            for name, root in policies.items()
        }
        model = {
            "schema_version": SCHEMA_VERSION,
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "model": MODEL_ID,
                "reasoning": "disabled",
                "temperature": 0.0,
                "domain": [min(DOMAIN), max(DOMAIN)],
                "num_proposals_per_call": NUM_PROPOSALS,
                "num_root_candidates": NUM_ROOTS,
                "expected_requests": EXPECTED_REQUESTS,
                "support_weights": "uniform_after_extension_deduplication",
                "controlled_rollout_support": (
                    "simulated_current_truth_plus_valid_generated_branch_rules"
                ),
            },
            "raw_responses_sha256": hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
            "initial": [item.public_dict() for item in initial],
            "roots": roots,
            "branches": {
                f"{root}:{int(label)}": [
                    item.public_dict()
                    for item in branches[(root, label)]
                ]
                for root, label in branch_keys
            },
        }
        checkpoint(output_dir / "MODEL.json", model)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "passed" if all(gates.values()) else "mechanics_failed"
            ),
            "protocol": model["protocol"],
            "gates": gates,
            "initial_diagnostics": initial_diagnostics,
            "branch_diagnostics": {
                f"{root}:{int(label)}": branch_diagnostics[(root, label)]
                for root, label in branch_keys
            },
            "selection": selection,
            "leave_one_out": loo,
            "endpoint": endpoint,
            "usage": usage,
            "model_sha256": hashlib.sha256(
                (output_dir / "MODEL.json").read_bytes()
            ).hexdigest(),
            "raw_responses_sha256": model["raw_responses_sha256"],
        }
        checkpoint(output_dir / "RESULT.json", result)
        return result
    except Exception as exc:
        checkpoint(raw_path, raw)
        usage = adapter.usage_snapshot()
        if isinstance(exc, SmokeExecutionError):
            usage = exc.usage
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "model": MODEL_ID,
            },
            "error": f"{type(exc).__name__}: {exc}",
            "usage": usage,
            "raw_responses_sha256": hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
        }
        checkpoint(output_dir / "FAILURE.json", failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_experiment(
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
