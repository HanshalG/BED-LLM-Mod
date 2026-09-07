import copy
import json

import pytest

from environments.chembench_mopen.structure_proposer import (
    build_messages,
    build_shuffled_feedback_messages,
)


def context():
    return {
        "history_inputs": [[a, 0, 1, 0, 1, 310, 7] for a in [1, 2, 3]],
        "observations": [-0.1, 0.4, 1.2],
        "public_bounds": [[0, 10]] * 5 + [[280, 340], [4, 10]],
        "parameter_bounds": [0.01, 10],
        "sigma": 0.15,
    }


def test_only_pairing_changes_without_sham_cue_or_mutation():
    args = context()
    before = copy.deepcopy(args)
    actual = build_messages(**args, mode="history_aware")
    sham, audit = build_shuffled_feedback_messages(**args, permutation=[1, 2, 0])
    assert args == before
    assert sham[0] == actual[0]
    real_data = json.loads(actual[1]["content"])
    sham_data = json.loads(sham[1]["content"])
    a, b = real_data.pop("history"), sham_data.pop("history")
    assert real_data == sham_data
    assert [r["inputs"] for r in a] == [r["inputs"] for r in b]
    assert sorted(r["observed_log1p_rate"] for r in a) == sorted(
        r["observed_log1p_rate"] for r in b
    )
    assert [r["observed_log1p_rate"] for r in b] == [0.4, 1.2, -0.1]
    assert audit["changed_observation_rows"] == 3
    assert audit["paid_calls_authorized"] is False
    assert audit["new_gate_authority"] is False
    assert build_shuffled_feedback_messages(**args, permutation=[1, 2, 0]) == (
        sham,
        audit,
    )


@pytest.mark.parametrize(
    "permutation",
    [[0, 1, 2], [1, 0, 2], [1, 1, 0], [True, 2, 0], [1, 2], [1, 2, 3], "120"],
)
def test_invalid_or_partially_fixed_assignment_fails(permutation):
    with pytest.raises(ValueError, match="derangement"):
        build_shuffled_feedback_messages(**context(), permutation=permutation)


def test_equal_labels_cannot_masquerade_as_changed_feedback():
    args = context()
    args["observations"] = [0.4, 0.4, 0.4]
    with pytest.raises(ValueError, match="does not change"):
        build_shuffled_feedback_messages(**args, permutation=[1, 2, 0])
    args["observations"] = [0.4, 0.4, 0.8]
    _, audit = build_shuffled_feedback_messages(**args, permutation=[1, 2, 0])
    assert audit["changed_observation_rows"] == 2


def test_no_information_added_to_blind_control_and_no_outcome_transport():
    args = context()
    blind = build_messages(**args, mode="history_blind")
    build_shuffled_feedback_messages(**args, permutation=[1, 2, 0])
    assert blind == build_messages(**args, mode="history_blind")
    assert "permutation" not in json.loads(blind[1]["content"])


def test_shuffling_only_replicates_is_not_a_broken_input_outcome_relationship():
    args = context()
    args["history_inputs"] = [args["history_inputs"][0]] * 3
    with pytest.raises(ValueError, match="replicate"):
        build_shuffled_feedback_messages(**args, permutation=[1, 2, 0])
