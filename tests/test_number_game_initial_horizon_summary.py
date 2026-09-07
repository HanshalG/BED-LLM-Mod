import numpy as np
import pytest

from scripts.number_game_initial_horizon_audit import ExactMembershipHorizon
from scripts.number_game_initial_horizon_summary import summarize


def fixture():
    extensions = sorted(
        set(
            tuple(row)
            for row in (np.random.default_rng(3).random((8, 6)) > 0.5).tolist()
        )
    )
    result = {
        "status": "initial_opportunity_null",
        "rows": [{"tree_seed": 1, **ExactMembershipHorizon(extensions).compare()}],
    }
    initial = [
        {
            "tree_seed": 1,
            "initial": [{"expression": str(i)} for i in range(len(extensions))],
        }
    ]
    return result, initial, lambda expression: extensions[int(expression)]


def test_exact_horizon_sacrifice_is_not_tie_breaking():
    result, initial, compiler = fixture()
    report = summarize(result, initial, compiler)
    assert report["h3_improvements_from_h2_tied_root"] == 0
    assert report["h3_improvements_requiring_h2_objective_sacrifice"] == 1
    assert report["rows"][0]["h2_objective_sacrifice"] == "1/576"
    assert report["original_status_unchanged"] == "initial_opportunity_null"
    assert report["new_gate_authority"] is False


def test_wrong_panel_and_h2_choice_rejected():
    result, initial, compiler = fixture()
    initial[0]["tree_seed"] = 2
    with pytest.raises(ValueError, match="mismatch"):
        summarize(result, initial, compiler)
    initial[0]["tree_seed"] = 1
    result["rows"][0]["first_queries"][1] = 5
    with pytest.raises(ValueError, match="replay"):
        summarize(result, initial, compiler)
