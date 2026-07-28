import pytest

from scripts.number_game_depth_three_development import (
    FIRST_SUPPORT_RETAINED_REJUVENATION,
    PLANNING_MODEL_ID,
    TARGET_MODEL_ID,
    TARGET_SEEDS,
    TREE_SEEDS,
    choose_risk_set_root,
    evaluate_policy_root_depth_three,
    retain_parent_hypotheses,
    run_tree_depth_three,
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


def test_v2_uses_fresh_seeds_and_swapped_models():
    assert PLANNING_MODEL_ID == "openai/gpt-5.4-mini"
    assert TARGET_MODEL_ID == "google/gemini-2.5-flash"
    assert TREE_SEEDS == tuple(range(27600, 27608))
    assert TARGET_SEEDS == tuple(range(27700, 27708))
    assert not set(TREE_SEEDS) & set(range(27400, 27408))
    assert not set(TARGET_SEEDS) & set(range(27500, 27508))


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


def test_retained_rejuvenation_merges_consistent_parent_particles():
    generated = [_rule("generated", {1, 2})]
    duplicate = _rule("duplicate", {1, 2})
    retained = _rule("retained", {1, 3})
    inconsistent = _rule("inconsistent", {2, 3})

    merged, diagnostics = retain_parent_hypotheses(
        parent_support=[duplicate, retained, inconsistent],
        generated_support=generated,
        query=1,
        label=True,
    )

    assert merged == [generated[0], retained]
    assert diagnostics == {
        "generated_unique_count": 1,
        "retained_parent_consistent_count": 2,
        "retained_parent_novel_count": 1,
        "merged_unique_count": 2,
    }


def test_first_support_mode_is_opt_in_and_validated(tmp_path):
    assert FIRST_SUPPORT_RETAINED_REJUVENATION == "retained_rejuvenation"

    with pytest.raises(ValueError, match="unsupported first_support_mode"):
        run_tree_depth_three(
            tree_index=0,
            tree_seed=1,
            target_seed=2,
            output_dir=tmp_path,
            run_id="invalid-first-support-mode",
            first_support_mode="not-a-mode",
        )


def test_risk_set_uses_hamming_only_inside_brier_tolerance():
    scores = {
        1: {
            "mean_posterior_predictive_brier": 0.100,
            "mean_best_hamming_error": 0.20,
            "truth_extension_coverage_rate": 0.8,
        },
        2: {
            "mean_posterior_predictive_brier": 0.104,
            "mean_best_hamming_error": 0.10,
            "truth_extension_coverage_rate": 0.4,
        },
        3: {
            "mean_posterior_predictive_brier": 0.106,
            "mean_best_hamming_error": 0.01,
            "truth_extension_coverage_rate": 1.0,
        },
    }

    assert choose_risk_set_root(scores, brier_tolerance=0.0) == 1
    assert choose_risk_set_root(scores, brier_tolerance=0.005) == 2
