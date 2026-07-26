from __future__ import annotations

import math

from scripts import discoverllm_priority_world_mechanics as mechanics


def test_score_parser_requires_exact_canonical_strings():
    keys = {"A_O1_W1", "A_O1_W2"}
    assert mechanics._parse_score_object(
        '{"A_O1_W1":"100","A_O1_W2":"0"}',
        keys,
    ) == {"A_O1_W1": 100, "A_O1_W2": 0}

    for invalid in (
        '{"A_O1_W1":100,"A_O1_W2":"0"}',
        '{"A_O1_W1":"01","A_O1_W2":"0"}',
        '{"A_O1_W1":"101","A_O1_W2":"0"}',
        '{"A_O1_W1":"100"}',
    ):
        try:
            mechanics._parse_score_object(invalid, keys)
        except ValueError:
            pass
        else:  # pragma: no cover - assertion helper
            raise AssertionError("invalid score object parsed")


def test_observation_mapping_is_reproducible_and_permutation():
    first = mechanics._observation_mapping("fixture")
    second = mechanics._observation_mapping("fixture")
    assert first == second
    assert all(sorted(indices) == [0, 1, 2, 3] for indices in first.values())


def test_bayes_analysis_detects_delayed_reversal():
    mapping = {"A": (0, 1, 2, 3), "B": (0, 1, 2, 3)}
    root = {}
    followup = {}
    for action in mechanics.ACTION_LABELS:
        for observation_index, observation in enumerate(
            mechanics.OBSERVATION_LABELS
        ):
            true_world = mechanics.WORLD_LABELS[observation_index]
            for world in mechanics.WORLD_LABELS:
                if action == "A":
                    root_high, root_low = 90, 10
                    followup_high = followup_low = 50
                else:
                    root_high, root_low = 65, 35
                    followup_high, followup_low = 99, 1
                root[f"{action}_{observation}_{world}"] = (
                    root_high if world == true_world else root_low
                )
                followup[f"{action}_{observation}_{world}"] = (
                    followup_high if world == true_world else followup_low
                )

    result = mechanics._analyze_task(mapping, root, followup)
    assert result["myopic_action"] == "A"
    assert result["nonmyopic_action"] == "B"
    assert result["root_changed"] is True
    assert result["delayed_reversal"] is True
    assert (
        result["actions"]["B"]["terminal_truth_log_posterior"]
        > result["actions"]["A"]["terminal_truth_log_posterior"]
    )
    assert math.isclose(
        result["actions"]["B"]["terminal_map_accuracy"],
        1.0,
    )


def test_likelihood_weight_uses_frozen_temperature():
    assert math.isclose(mechanics._likelihood_weight(50), 1.0)
    assert mechanics._likelihood_weight(100) > mechanics._likelihood_weight(0)
