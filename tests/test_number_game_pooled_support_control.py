from scripts.number_game_generator_aware_bed import RuleHypothesis
from scripts.number_game_pooled_support_control import (
    dedupe_rules,
    global_pooled_static_branches,
    pooled_static_branches,
)


def _rule(name: str, positives: set[int]) -> RuleHypothesis:
    return RuleHypothesis(
        name=name,
        expression="n == 0",
        extension=tuple(number in positives for number in range(101)),
    )


def test_dedupe_rules_uses_exact_extensions():
    first = _rule("first", {1, 2})
    duplicate = _rule("duplicate", {1, 2})
    other = _rule("other", {3})

    assert dedupe_rules([first, duplicate, other]) == [first, other]


def test_pooled_static_support_forgets_origin_then_conditions_on_label():
    negative = _rule("negative", {2})
    positive = _rule("positive", {1, 2})
    duplicate_positive = _rule("duplicate positive", {1, 2})

    pooled, sizes = pooled_static_branches(
        roots=[1],
        branches={
            (1, False): [negative, duplicate_positive],
            (1, True): [positive],
        },
    )

    assert sizes == {1: 2}
    assert pooled[(1, False)] == [negative]
    assert pooled[(1, True)] == [duplicate_positive]


def test_global_pool_is_shared_across_roots_before_label_filtering():
    only_zero = _rule("only zero", {0})
    only_one = _rule("only one", {1})
    both = _rule("both", {0, 1})
    neither = _rule("neither", {2})

    pooled, size = global_pooled_static_branches(
        roots=[0, 1],
        branches={
            (0, False): [only_one, neither],
            (0, True): [only_zero, both],
            (1, False): [only_zero, neither],
            (1, True): [only_one, both],
        },
    )

    assert size == 4
    assert {rule.name for rule in pooled[(0, True)]} == {
        "only zero",
        "both",
    }
    assert {rule.name for rule in pooled[(1, True)]} == {
        "only one",
        "both",
    }
