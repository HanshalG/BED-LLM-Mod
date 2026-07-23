import json

import numpy as np
import pytest

from environments.heart_workup import ORDER_WORKUP_ACTION, HeartWorkupModel
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError
from scripts.nonmyopic_heart_workup_proposal_gate import (
    HeartProposalGateConfig,
    build_proposal_cells,
    run_proposal_gate,
)
from scripts.nonmyopic_heart_workup_strategy import (
    DeterministicIndexedHeartModel,
    HeartStrategyConfig,
    IndexedHeartProvider,
    branch_menus,
    compile_indexed_cell,
    fixed_roots,
    run_smoke,
)


def test_heart_indexed_policy_compiles_workup_contingencies() -> None:
    model = HeartWorkupModel()
    state = model.initial_state
    belief = model.initial_belief
    roots = fixed_roots(model, state=state, belief=belief, count=4)
    menus = branch_menus(model, state=state, belief=belief, roots=roots)
    response = json.dumps(
        {
            f"r{slot}": [0] * len(root_menus)
            for slot, root_menus in enumerate(menus)
        }
    )

    strategies = compile_indexed_cell(response, roots=roots, menus=menus)

    assert roots[0] == ORDER_WORKUP_ACTION
    assert len(strategies) == 4
    assert all(
        ORDER_WORKUP_ACTION in choices
        for root_menus in menus[1:]
        for choices in root_menus.values()
    )


def test_heart_indexed_policy_rejects_wrong_branch_length() -> None:
    model = HeartWorkupModel()
    roots = fixed_roots(
        model, state=model.initial_state, belief=model.initial_belief, count=4
    )
    menus = branch_menus(
        model, state=model.initial_state, belief=model.initial_belief, roots=roots
    )

    with pytest.raises(StrategyProposalError, match="exactly"):
        compile_indexed_cell(
            json.dumps({"r0": [], "r1": [], "r2": [], "r3": []}),
            roots=roots,
            menus=menus,
        )


def test_heart_proposal_cells_are_balanced_distinct_workup_opportunities() -> None:
    model = HeartWorkupModel()
    cells = build_proposal_cells(model, HeartProposalGateConfig())
    unworked = [cell for cell in cells if cell.phase == "unworked"]
    worked = [cell for cell in cells if cell.phase == "worked"]

    assert len(cells) == 32
    assert len(unworked) == len(worked) == 16
    assert all(not cell.state.workup_ordered for cell in unworked)
    assert all(cell.state.workup_ordered for cell in worked)
    assert len({cell.history for cell in unworked}) == 16
    assert len({cell.history for cell in worked}) == 16
    assert all(model.target_entropy(cell.belief) > 0.0 for cell in cells)


def test_deterministic_heart_dry_runs_pass_mechanics_only() -> None:
    strategy_config = HeartStrategyConfig()
    smoke_provider = IndexedHeartProvider(DeterministicIndexedHeartModel(), strategy_config)
    smoke = run_smoke(smoke_provider, strategy_config)
    gate_provider = IndexedHeartProvider(DeterministicIndexedHeartModel(), strategy_config)
    gate = run_proposal_gate(gate_provider, HeartProposalGateConfig())

    assert all(smoke["mechanics"].values())
    assert all(gate["mechanics"].values())
    assert np.isfinite(gate["comparisons"]["matched_random_minus_llm_cost"]["mean"])
    assert gate["endpoint_gate"]["workup_rate_at_least_threshold"] is False
