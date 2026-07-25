from __future__ import annotations

import json

import pytest

from scripts.pscon_semantic_tree_smoke import (
    Product,
    SemanticQuery,
    _execute_tree_roots,
    _tree_scores,
    entropy,
    parse_answer,
    parse_semantic_query,
    query_eig,
)


def _products(count: int = 6) -> list[Product]:
    return [
        Product(
            product_id=f"P{index}",
            attributes={"title": (f"Product {index}",)},
        )
        for index in range(count)
    ]


def _query(products: list[Product], labels: list[int], name: str) -> SemanticQuery:
    return SemanticQuery(
        question=f"Which {name} option?",
        options=("A", "B", "C")[: max(labels)],
        product_ids=tuple(product.product_id for product in products),
        assignments=tuple(labels),
    )


def test_parse_semantic_query_requires_exact_complete_partition() -> None:
    products = _products()
    parsed = parse_semantic_query(
        json.dumps(
            {
                "question": "Which display style do you prefer?",
                "options": ["LED", "QLED", "OLED"],
                "assignments": [1, 1, 2, 2, 3, 3],
            }
        ),
        products,
        expected_options=3,
    )
    assert parsed.label_for("P4") == 3
    with pytest.raises(ValueError, match="assign every"):
        parse_semantic_query(
            json.dumps(
                {
                    "question": "Which display style do you prefer?",
                    "options": ["LED", "QLED", "OLED"],
                    "assignments": [1, 2],
                }
            ),
            products,
            expected_options=3,
        )


def test_entropy_and_query_eig_match_balanced_partition() -> None:
    products = _products()
    query = _query(products, [1, 1, 2, 2, 3, 3], "balanced")
    assert entropy([1, 1, 2, 2, 3, 3]) == pytest.approx(1.0986122886681098)
    assert query_eig(query, [product.product_id for product in products]) == (
        pytest.approx(1.0986122886681098)
    )


def test_tree_score_adds_expected_best_branch_information() -> None:
    products = _products()
    root = _query(products, [1, 1, 2, 2, 3, 3], "root")
    followups = []
    for branch in range(3):
        branch_products = products[2 * branch : 2 * branch + 2]
        split = SemanticQuery(
            question=f"Which branch {branch} detail?",
            options=("left", "right"),
            product_ids=tuple(product.product_id for product in branch_products),
            assignments=(1, 2),
        )
        flat = SemanticQuery(
            question=f"Which branch {branch} control?",
            options=("same",),
            product_ids=tuple(product.product_id for product in branch_products),
            assignments=(1, 1),
        )
        followups.append([split, flat])
    immediate, depth2 = _tree_scores(
        [root],
        [followups],
        [product.product_id for product in products],
    )
    assert immediate == pytest.approx([math_log(3)])
    assert depth2 == pytest.approx([math_log(3) + math_log(2)])


def math_log(value: float) -> float:
    import math

    return math.log(value)


def test_execution_uses_independent_answers_and_can_drop_truth() -> None:
    products = _products(3)
    root = _query(products, [1, 2, 3], "root")
    followups = []
    for product in products:
        followups.append(
            [
                SemanticQuery(
                    question="Only one candidate remains?",
                    options=("yes",),
                    product_ids=(product.product_id,),
                    assignments=(1,),
                ),
                SemanticQuery(
                    question="Still only one candidate?",
                    options=("yes",),
                    product_ids=(product.product_id,),
                    assignments=(1,),
                ),
            ]
        )
    records = _execute_tree_roots(
        roots=[root],
        followups=[followups],
        root_answers=[2],
        followup_answers=[1],
        target_id="P0",
    )
    assert records[0]["root_consistent"] is False
    assert records[0]["target_mass"] == 0.0


def test_responder_answer_parser_is_strict() -> None:
    assert parse_answer("2", 3) == 2
    with pytest.raises(ValueError):
        parse_answer("Option 2", 3)
