import json
import math

import pytest

from environments.gated_sensor import GatedSensorModel
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError
from scripts.nonmyopic_gated_sensor_strategy_prior_v2 import (
    DeterministicIndexedModel,
    IndexedStrategyConfig,
    IndexedStrategyProvider,
    _branch_menus,
    _fixed_roots,
    compile_indexed_cell,
    run_experiment_v2,
)


def test_indexed_compiler_maps_branch_local_indices_to_legal_actions() -> None:
    model = GatedSensorModel()
    roots = _fixed_roots(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        horizon=2,
        count=6,
    )
    menus = _branch_menus(model, state=model.initial_state, roots=roots, horizon=2)
    response = json.dumps(
        {
            "choices": [[0, 0] for _root_menus in menus]
        }
    )
    strategies = compile_indexed_cell(response, roots=roots, menus=menus)

    assert [strategy.root_action for strategy in strategies] == list(roots)
    assert strategies[0].followups["none"].startswith("precise:")
    assert roots[:3] == ("activate:A", "activate:B", "activate:C")


def test_indexed_compiler_rejects_out_of_range_choice() -> None:
    with pytest.raises(StrategyProposalError, match="must be in"):
        compile_indexed_cell(
            json.dumps(
                {"choices": [[2, 0]]}
            ),
            roots=("activate:A",),
            menus=[{"none": ["precise:bit-0"]}],
        )


def test_v2_dry_run_is_paired_and_uses_no_terminal_llm_calls() -> None:
    config = IndexedStrategyConfig(
        num_trials=4,
        num_rounds=4,
        num_strategies=6,
        bootstrap_replicates=100,
        trial_concurrency=2,
    )
    provider = IndexedStrategyProvider(DeterministicIndexedModel(), config)
    summary = run_experiment_v2(provider, config)

    assert all(summary["mechanics"].values())
    assert summary["provider"]["local_terminal_cells"] > 0
    assert summary["provider"]["physical_requests"] < summary["provider"]["logical_calls"]
    assert all(
        math.isfinite(trace["entropy_auc"])
        for traces in summary["traces"].values()
        for trace in traces
    )
