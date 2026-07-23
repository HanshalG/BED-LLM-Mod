from dataclasses import replace
import json

import numpy as np
import pytest

from scripts.audit_nonmyopic_range_gated_rock_depth5_oracle import audit_result
from scripts.nonmyopic_range_gated_rock_depth5_oracle import (
    H5_ROUTE,
    FocusedDepth5Config,
    build_focused_depth5_model,
    run_qualification,
)
from scripts.nonmyopic_rock_depth_oracle import exhaustive_action_values


def test_focused_depth5_config_and_prior_are_frozen() -> None:
    config = FocusedDepth5Config()
    config.validate()
    model = build_focused_depth5_model(config)

    assert model.prior_good_probabilities[6] == 0.5
    assert np.all(model.prior_good_probabilities[np.arange(8) != 6] == 0.005)
    assert float(np.sum(model.initial_belief)) == pytest.approx(1.0)
    assert model.entropy(model.initial_belief) == pytest.approx(
        0.9135006421901125
    )
    with pytest.raises(ValueError, match="p_good=.005"):
        replace(config, secondary_good_probability=0.01).validate()


def test_exact_h5_moves_while_h4_checks_remotely() -> None:
    config = FocusedDepth5Config()
    model = build_focused_depth5_model(config)
    values = {
        depth: exhaustive_action_values(
            model,
            position=model.map_spec.start_position,
            belief=model.initial_belief,
            depth=depth,
        )[0]
        for depth in (4, 5)
    }

    assert max(values[4], key=values[4].get) == "check-6"
    assert max(values[5], key=values[5].get) == "move-NORTH"
    assert values[5]["move-NORTH"] > values[4]["check-6"]


def test_small_focused_depth5_qualification_replays_exactly() -> None:
    config = FocusedDepth5Config(
        num_trials=12,
        bootstrap_replicates=200,
        seed=24_237,
    )
    result = run_qualification(config)
    serialized_result = json.loads(json.dumps(result))
    audit = audit_result(serialized_result, audit_bootstrap_seed=24_238)

    assert all(result["mechanics"].values())
    assert result["comparison"]["entropy_auc_gain_mean"] > 0.0
    assert result["comparison"]["entropy_auc_wins_ties_losses"] == [12, 0, 0]
    assert all(
        tuple(step["action"] for step in row["steps"][:5]) == H5_ROUTE
        for row in result["traces"]["5"]
    )
    assert audit["mechanics"]["all_traces_replayed"]
    assert audit["mechanics"]["source_prior_recomputed"]
    assert audit["mechanics"]["producer_comparison_recomputed"]
    assert audit["mechanics"]["audit_values_match_without_ci"]
