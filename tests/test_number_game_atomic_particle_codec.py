from __future__ import annotations

import json

import pytest

from scripts import number_game_atomic_particle_codec as codec
from scripts.number_game_generator_aware_bed import RuleHypothesis, compile_expression


def rule(name: str, expression: str) -> RuleHypothesis:
    return RuleHypothesis(name, expression, compile_expression(expression))


def raw(name: str, expression: str) -> str:
    return json.dumps({"name": name, "expression": expression})


def test_atomic_schema_prompt_and_parser_are_history_exact():
    schema = codec.response_format()["json_schema"]["schema"]
    assert schema["required"] == ["name", "expression"]
    assert schema["additionalProperties"] is False
    messages = codec.particle_messages(((7, False), (8, True)), slot=9)
    payload = json.loads(messages[1]["content"])
    assert payload["observations"] == [
        {"number": 7, "answer": "NO"},
        {"number": 8, "answer": "YES"},
    ]
    assert payload["diversity_cue"] == codec.DIVERSITY_CUES[1]
    assert codec.parse_atomic(raw("even", "divisible(n, 2)"), ((8, True),)).hypothesis is not None
    assert codec.parse_atomic(raw("even", "divisible(n, 2)"), ((7, True),)).rejection == "history"
    assert codec.parse_atomic("{", ()).rejection == "json"
    assert codec.parse_atomic(raw("unsafe", "[n]"), ()).rejection == "expression"


def test_duplicates_remain_particles_and_systematic_resampling_is_reproducible():
    even = rule("even", "divisible(n, 2)")
    odd = rule("odd", "n % 2 == 1")
    particles, diagnostic = codec.parse_group([raw("even-a", "divisible(n, 2)"), raw("even-b", "n % 2 == 0"), raw("odd", "n % 2 == 1")], ())
    assert len(particles) == 3
    assert diagnostic["unique_extensions"] == 2
    first = codec.systematic_resample([even, even, odd], 12, 4)
    second = codec.systematic_resample([even, even, odd], 12, 4)
    assert [row.extension for row in first] == [row.extension for row in second]
    assert sum(row.extension == even.extension for row in first) > sum(row.extension == odd.extension for row in first)


def test_refresh_has_exact_equal_parent_generated_width_and_no_truth_injection():
    even = rule("even", "divisible(n, 2)")
    even_below = rule("even-below", "divisible(n, 2) and n < 50")
    odd = rule("odd", "n % 2 == 1")
    refreshed = codec.refresh_belief([even, odd], [even_below, odd], ((2, True),), seed=9, width=8)
    assert len(refreshed) == 8
    assert all(row.extension[2] for row in refreshed)
    assert sum(row.extension == even.extension for row in refreshed[:4]) == 4
    assert sum(row.extension == even_below.extension for row in refreshed[4:]) == 4
    with pytest.raises(ValueError, match="empty"):
        codec.refresh_belief([odd], [even_below], ((2, True),), seed=9, width=8)


def test_candidate_roots_and_fixed_plan_are_deterministic():
    particles = [
        rule("even", "divisible(n, 2)"),
        rule("three", "divisible(n, 3)"),
        rule("five", "divisible(n, 5)"),
        rule("square", "is_square(n)"),
        rule("prime", "is_prime(n)"),
        rule("small", "n < 30"),
        rule("ending", "ends_with(n, 6)"),
        rule("mod", "n % 7 == 2"),
    ] * 8
    roots = codec.candidate_roots(particles, seed=11)
    assert len(roots) == len(set(roots)) == 4
    plan = codec.fixed_depth3_score(particles, roots[0], seed=11)
    assert plan.root == roots[0]
    assert plan.score >= codec.query_eig(particles, roots[0])


def test_generated_depth3_uses_conditioned_particles_and_is_finite():
    initial = [
        rule("even", "divisible(n, 2)"),
        rule("three", "divisible(n, 3)"),
        rule("five", "divisible(n, 5)"),
        rule("square", "is_square(n)"),
    ] * 16
    root = 2
    first = {
        False: [rule("odd", "n % 2 == 1"), rule("odd3", "n % 6 == 3")] * 16,
        True: [rule("even", "divisible(n, 2)"), rule("even-small", "divisible(n, 2) and n < 50")] * 16,
    }
    second = {}
    for answer in (False, True):
        first_belief = codec.refresh_belief(initial, first[answer], ((root, answer),), seed=13 ^ (root * 17 + int(answer)))
        query, _ = codec.best_query(first_belief, (root,))
        for second_answer in (False, True):
            compatible = [row for row in first_belief if row.extension[query] is second_answer]
            second[(answer, second_answer)] = (compatible or [first_belief[0]]) * (32 // max(len(compatible), 1) + 1)
    plan = codec.generated_depth3_score(initial, root, first, second, seed=13)
    assert plan.root == root
    assert plan.score >= codec.query_eig(initial, root)
    assert set(plan.second_queries) == {False, True}
    fixed = codec.generated_depth3_score(
        initial,
        root,
        first,
        second,
        seed=13,
        width=32,
        fixed_second_queries=plan.second_queries,
    )
    assert fixed.second_queries == plan.second_queries


def test_brier_spearman_and_frequency_weighting():
    even = rule("even", "divisible(n, 2)")
    odd = rule("odd", "n % 2 == 1")
    belief = [even, even, even, odd]
    assert codec.answer_probability(belief, 2) == .75
    assert codec.posterior_predictive_brier([even], even, excluded=(2, 3, 4)) == 0.0
    assert codec.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
