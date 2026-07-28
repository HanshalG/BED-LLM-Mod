from scripts.number_game_depth_three_development import (
    evaluate_policy_root_depth_three,
    static_depth_three_branches,
)
from scripts.number_game_generator_aware_bed import (
    RuleHypothesis,
    history_messages,
)


def _rule(name: str, positives: set[int]) -> RuleHypothesis:
    return RuleHypothesis(
        name=name,
        expression="n == 0",
        extension=tuple(number in positives for number in range(101)),
    )


def test_history_messages_include_every_observation():
    messages = history_messages(
        ((3, True), (8, False)),
        enforce_constraints=True,
    )
    prompt = messages[1]["content"]

    assert "Is 3 in the concept? YES." in prompt
    assert "Is 8 in the concept? NO." in prompt
    assert "consistent with every observation" in prompt


def test_legacy_single_observation_prompt_remains_default():
    prompt = history_messages(((3, True),))[1]["content"]

    assert "The only observation is: Is 3 in the concept? YES." in prompt
    assert "Hard executable constraints" not in prompt


def test_static_depth_three_branches_condition_sequentially():
    rules = [
        _rule("a", {1, 2}),
        _rule("b", {1, 3}),
        _rule("c", {2, 3}),
        _rule("d", set()),
    ]
    first, second = static_depth_three_branches(
        support=rules,
        roots=[1],
    )
    second_query = next(
        key[2] for key in second if key[:2] == (1, True)
    )

    assert first[(1, True)] == rules[:2]
    assert all(
        rule.extension[second_query] == label
        for label in (False, True)
        for rule in second[(1, True, second_query, label)]
    )


def test_depth_three_evaluator_follows_target_path():
    false_rule = _rule("false", {2})
    true_rule = _rule("true", {1, 2})
    target = _rule("target", {1, 2})
    first = {
        (1, False): [false_rule],
        (1, True): [false_rule, true_rule],
    }
    second_query = 0
    second = {
        (1, False, second_query, False): [false_rule],
        (1, False, second_query, True): [false_rule],
        (1, True, second_query, False): [false_rule, true_rule],
        (1, True, second_query, True): [true_rule],
    }

    result = evaluate_policy_root_depth_three(
        policy="test",
        root=1,
        targets={"target": target},
        first_branches=first,
        second_branches=second,
    )

    assert result["targets"][0]["first_label"] is True
    assert result["targets"][0]["truth_extension_covered"] is True
