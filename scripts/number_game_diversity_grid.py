#!/usr/bin/env python3
"""Diversity-constrained executable support for the Number Game."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import (
    checkpoint,
    strict_json_object,
)
from scripts.number_game_generator_aware_bed import (
    DOMAIN,
    RuleHypothesis,
    compile_expression,
    is_power_of_two,
    is_prime,
    is_square,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-diversity-grid-1"
FAMILIES = (
    "periodic_digit",
    "ordered",
    "number_theoretic",
    "compositional",
)
ITEMS_PER_FAMILY = 6
TOTAL_ITEMS = len(FAMILIES) * ITEMS_PER_FAMILY
MIN_POSITIVE_COUNT = 3
MAX_POSITIVE_COUNT = len(DOMAIN) - 3
MASK_BYTES = (len(DOMAIN) + 7) // 8
ALL_MASK = (1 << len(DOMAIN)) - 1
MIN_FEASIBLE_PER_FAMILY = 16
MIN_FEASIBLE_UNION = 4_000
SERVING_HISTORIES: tuple[tuple[tuple[int, bool], ...], ...] = (
    (),
    (),
    ((10, True),),
    ((10, False),),
    ((42, True),),
    ((42, False),),
    ((10, True), (20, False)),
    ((10, False), (20, True)),
    ((42, True), (75, True)),
    ((42, False), (75, False)),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_format() -> dict[str, Any]:
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "expression"],
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 100},
            "expression": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "number_game_diversity_grid",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["families"],
                "properties": {
                    "families": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(FAMILIES),
                        "properties": {
                            family: {
                                "type": "array",
                                "minItems": ITEMS_PER_FAMILY,
                                "maxItems": ITEMS_PER_FAMILY,
                                "items": item_schema,
                            }
                            for family in FAMILIES
                        },
                    }
                },
            },
        },
    }


_GRAMMAR = """Each item is a concise Python expression over integer n in 0..100.
It must return bool and may use and/or/not, + - * // %, comparisons, n, and:
  divisible(n, k), is_square(n), is_power_of_two(n), is_prime(n),
  digit_sum(n), ends_with(n, digit).
No lambdas, containers, indexing, attributes, imports, lookup tables, singleton
rules, or explicit exceptions for observed numbers."""


def messages(
    observations: Sequence[tuple[int, bool]],
) -> list[dict[str, str]]:
    if observations:
        observation_text = "\n".join(
            f"- n={number} MUST evaluate to {'true' if label else 'false'}."
            for number, label in observations
        )
    else:
        observation_text = "- No labels have been observed."
    return [
        {
            "role": "system",
            "content": (
                "You generate diverse executable hypotheses for Bayesian "
                "concept learning. Return only the requested strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate exactly six distinct rules in each of four families.\n"
                "periodic_digit: modular, divisibility, final-digit, or digit-sum "
                "structure.\n"
                "ordered: thresholds or bounded intervals without modular or "
                "number-theoretic helpers.\n"
                "number_theoretic: transformed prime, square, or power-of-two "
                "structure.\n"
                "compositional: combine at least two structurally different "
                "ideas with Boolean logic.\n\n"
                f"Observed constraints:\n{observation_text}\n"
                "Verify every expression against every observed constraint. "
                "Within and across families, avoid behaviorally equivalent "
                "rules on 0..100.\n\n"
                f"{_GRAMMAR}"
            ),
        },
    ]


def _expression_features(expression: str) -> dict[str, Any]:
    tree = ast.parse(expression, mode="eval")
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    return {
        "periodic_digit": (
            any(isinstance(node, ast.Mod) for node in ast.walk(tree))
            or any(
                call in {"divisible", "digit_sum", "ends_with"}
                for call in calls
            )
        ),
        "ordered": any(
            isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            for operator in node.ops
        ),
        "number_theoretic": any(
            call in {"is_prime", "is_square", "is_power_of_two"}
            for call in calls
        ),
        "boolean_composition": any(
            isinstance(node, (ast.BoolOp, ast.Not))
            for node in ast.walk(tree)
        ),
        "distinct_helpers": len(set(calls)),
        "tree": tree,
    }


def _matches_family(family: str, features: dict[str, Any]) -> bool:
    if family == "periodic_digit":
        return bool(features["periodic_digit"]) and not bool(
            features["number_theoretic"]
        )
    if family == "ordered":
        return (
            bool(features["ordered"])
            and not bool(features["periodic_digit"])
            and not bool(features["number_theoretic"])
        )
    if family == "number_theoretic":
        return bool(features["number_theoretic"])
    if family == "compositional":
        structural_count = sum(
            bool(features[key])
            for key in ("periodic_digit", "ordered", "number_theoretic")
        )
        return bool(features["boolean_composition"]) and (
            structural_count >= 2 or features["distinct_helpers"] >= 2
        )
    raise ValueError(f"unknown family {family!r}")


def _memorizes_observation(
    tree: ast.AST,
    observations: Sequence[tuple[int, bool]],
) -> bool:
    observed_numbers = {number for number, _ in observations}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            continue
        pairs = zip(
            [node.left, *node.comparators[:-1]],
            node.comparators,
        )
        for left, right in pairs:
            if (
                isinstance(left, ast.Name)
                and left.id == "n"
                and isinstance(right, ast.Constant)
                and right.value in observed_numbers
            ) or (
                isinstance(right, ast.Name)
                and right.id == "n"
                and isinstance(left, ast.Constant)
                and left.value in observed_numbers
            ):
                return True
    return False


def parse_response(
    response: str,
    *,
    observations: Sequence[tuple[int, bool]] = (),
) -> tuple[list[RuleHypothesis], dict[str, Any]]:
    value = strict_json_object(response, label="number-game diversity grid")
    if set(value) != {"families"} or not isinstance(value["families"], dict):
        raise ValueError("diversity-grid response has the wrong top-level fields")
    families = value["families"]
    if set(families) != set(FAMILIES):
        raise ValueError("diversity-grid response has the wrong family fields")

    accepted: list[RuleHypothesis] = []
    accepted_by_family = {family: 0 for family in FAMILIES}
    seen_extensions: set[tuple[bool, ...]] = set()
    rejected = {
        "wrong_items": 0,
        "invalid_expression": 0,
        "wrong_family": 0,
        "observation_memorization": 0,
        "inconsistent": 0,
        "near_constant": 0,
        "duplicate_extension": 0,
    }
    for family in FAMILIES:
        items = families[family]
        if not isinstance(items, list) or len(items) != ITEMS_PER_FAMILY:
            raise ValueError(
                f"{family} must contain exactly {ITEMS_PER_FAMILY} items"
            )
        for item in items:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "expression"}
                or not isinstance(item["name"], str)
                or not item["name"].strip()
                or not isinstance(item["expression"], str)
            ):
                rejected["wrong_items"] += 1
                continue
            expression = item["expression"].strip()
            try:
                features = _expression_features(expression)
                extension = compile_expression(expression)
            except (SyntaxError, ValueError):
                rejected["invalid_expression"] += 1
                continue
            if _memorizes_observation(features["tree"], observations):
                rejected["observation_memorization"] += 1
                continue
            if not _matches_family(family, features):
                rejected["wrong_family"] += 1
                continue
            if any(
                extension[number] != label
                for number, label in observations
            ):
                rejected["inconsistent"] += 1
                continue
            positive_count = sum(extension)
            if not MIN_POSITIVE_COUNT <= positive_count <= MAX_POSITIVE_COUNT:
                rejected["near_constant"] += 1
                continue
            if extension in seen_extensions:
                rejected["duplicate_extension"] += 1
                continue
            seen_extensions.add(extension)
            accepted_by_family[family] += 1
            accepted.append(
                RuleHypothesis(
                    name=item["name"].strip(),
                    expression=expression,
                    extension=extension,
                )
            )
    return accepted, {
        "raw_count": TOTAL_ITEMS,
        "valid_unique_count": len(accepted),
        "valid_unique_by_family": accepted_by_family,
        "rejected": rejected,
    }


def _mask(predicate: Callable[[int], bool]) -> int:
    return sum(
        int(bool(predicate(number))) << number
        for number in DOMAIN
    )


def _eligible_masks(masks: Iterable[int]) -> set[int]:
    return {
        mask
        for mask in masks
        if mask not in {0, ALL_MASK}
        and MIN_POSITIVE_COUNT <= mask.bit_count() <= MAX_POSITIVE_COUNT
    }


def _periodic_digit_masks() -> set[int]:
    masks: list[int] = []
    for modulus in range(2, 21):
        for remainder in range(modulus):
            masks.append(
                _mask(lambda n, k=modulus, r=remainder: n % k == r)
            )
    for cutoff in range(20):
        masks.extend(
            (
                _mask(
                    lambda n, c=cutoff: sum(map(int, str(n))) < c
                ),
                _mask(
                    lambda n, c=cutoff: sum(map(int, str(n))) == c
                ),
                _mask(
                    lambda n, c=cutoff: sum(map(int, str(n))) > c
                ),
            )
        )
    for modulus in range(2, 10):
        for remainder in range(modulus):
            masks.append(
                _mask(
                    lambda n, k=modulus, r=remainder: (
                        sum(map(int, str(n))) % k == r
                    )
                )
            )
    for digit in range(10):
        masks.append(_mask(lambda n, d=digit: n % 10 == d))
    return _eligible_masks(masks)


def _ordered_masks() -> set[int]:
    masks: list[int] = []
    for cutoff in range(102):
        masks.extend(
            (
                _mask(lambda n, c=cutoff: n < c),
                _mask(lambda n, c=cutoff: n <= c),
                _mask(lambda n, c=cutoff: n > c),
                _mask(lambda n, c=cutoff: n >= c),
            )
        )
    for lower in range(0, 101, 2):
        for upper in range(lower + 2, 102, 2):
            masks.append(
                _mask(
                    lambda n, lo=lower, hi=upper: lo <= n <= hi
                )
            )
    return _eligible_masks(masks)


def _number_theoretic_masks() -> set[int]:
    masks: list[int] = []
    predicates = (is_prime, is_square, is_power_of_two)
    for multiplier in range(1, 13):
        for offset in range(-50, 51):
            for predicate in predicates:
                masks.append(
                    _mask(
                        lambda n, a=multiplier, b=offset, p=predicate: (
                            p(a * n + b)
                        )
                    )
                )
    for divisor in range(1, 13):
        for offset in range(-30, 31):
            for predicate in predicates:
                masks.append(
                    _mask(
                        lambda n, a=divisor, b=offset, p=predicate: (
                            (n - b) % a == 0 and p((n - b) // a)
                        )
                    )
                )
    return _eligible_masks(masks)


def _sample_evenly(masks: set[int], count: int = 80) -> list[int]:
    ordered = sorted(masks)
    if len(ordered) <= count:
        return ordered
    return [
        ordered[index * len(ordered) // count]
        for index in range(count)
    ]


def _compositional_masks(
    periodic: set[int],
    ordered: set[int],
    number_theoretic: set[int],
) -> set[int]:
    masks: set[int] = set()
    for left_family, right_family in (
        (periodic, ordered),
        (periodic, number_theoretic),
        (ordered, number_theoretic),
    ):
        for left in _sample_evenly(left_family):
            for right in _sample_evenly(right_family):
                masks.update(
                    (
                        left & right,
                        left | right,
                        left & (ALL_MASK ^ right),
                        right & (ALL_MASK ^ left),
                        left ^ right,
                    )
                )
    return _eligible_masks(masks)


def build_feasibility_banks() -> dict[str, set[int]]:
    periodic = _periodic_digit_masks()
    ordered = _ordered_masks()
    number_theoretic = _number_theoretic_masks()
    return {
        "periodic_digit": periodic,
        "ordered": ordered,
        "number_theoretic": number_theoretic,
        "compositional": _compositional_masks(
            periodic,
            ordered,
            number_theoretic,
        ),
    }


def _bank_sha256(family: str, masks: Iterable[int]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{INTERFACE_VERSION}:{family}\0".encode())
    for mask in sorted(masks):
        digest.update(mask.to_bytes(MASK_BYTES, "little"))
    return digest.hexdigest()


def _consistent(mask: int, history: Sequence[tuple[int, bool]]) -> bool:
    return all(bool(mask & (1 << number)) == label for number, label in history)


def run_feasibility_audit() -> dict[str, Any]:
    banks = build_feasibility_banks()
    cases = []
    for index, history in enumerate(SERVING_HISTORIES):
        consistent = {
            family: {
                mask
                for mask in masks
                if _consistent(mask, history)
            }
            for family, masks in banks.items()
        }
        cases.append(
            {
                "case_index": index,
                "observations": [
                    [number, label] for number, label in history
                ],
                "feasible_unique_by_family": {
                    family: len(masks)
                    for family, masks in consistent.items()
                },
                "feasible_unique_union": len(
                    set().union(*consistent.values())
                ),
            }
        )
    gates = {
        "exact_ten_frozen_histories": len(cases) == 10,
        "every_history_has_at_least_sixteen_per_family": all(
            min(case["feasible_unique_by_family"].values())
            >= MIN_FEASIBLE_PER_FAMILY
            for case in cases
        ),
        "every_history_has_at_least_four_thousand_total": all(
            case["feasible_unique_union"] >= MIN_FEASIBLE_UNION
            for case in cases
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gated_null",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "families": list(FAMILIES),
            "items_per_family": ITEMS_PER_FAMILY,
            "minimum_positive_count": MIN_POSITIVE_COUNT,
            "maximum_positive_count": MAX_POSITIVE_COUNT,
            "minimum_feasible_per_family": MIN_FEASIBLE_PER_FAMILY,
            "minimum_feasible_union": MIN_FEASIBLE_UNION,
            "model_calls": 0,
        },
        "banks": {
            family: {
                "unique_count": len(masks),
                "sha256": _bank_sha256(family, masks),
            }
            for family, masks in banks.items()
        },
        "cases": cases,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    result = run_feasibility_audit()
    checkpoint(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
