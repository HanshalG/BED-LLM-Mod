import pytest

from environments.mushroom_feature_acquisition import COLLECT_ACTION, MushroomFeatureModel
from scripts.nonmyopic_mushroom_proposal_gate import (
    MushroomProposalGateConfig,
    _matched_random_strategies,
    build_proposal_cells,
    run_proposal_gate,
    strategy_cost,
)
from scripts.nonmyopic_mushroom_strategy import (
    DeterministicIndexedMushroomModel,
    IndexedMushroomProvider,
    MushroomStrategyConfig,
    _branch_menus,
    _fixed_roots,
)


@pytest.fixture(scope="module")
def frozen_cells() -> tuple[MushroomFeatureModel, list]:
    model = MushroomFeatureModel()
    config = MushroomProposalGateConfig()
    return model, build_proposal_cells(model, config)


def test_proposal_cells_are_balanced_distinct_and_nonmyopic(frozen_cells) -> None:
    _model, cells = frozen_cells

    assert len(cells) == 32
    for phase in ("uncollected", "collected"):
        phase_cells = [cell for cell in cells if cell.phase == phase]
        assert len(phase_cells) == 16
        assert len({cell.history for cell in phase_cells}) == 16
    assert all(not cell.state.specimen_collected for cell in cells if cell.phase == "uncollected")
    assert all(cell.state.specimen_collected for cell in cells if cell.phase == "collected")


def test_matched_random_strategies_are_reproducible_and_legal(frozen_cells) -> None:
    model, cells = frozen_cells
    cell = cells[0]
    roots = _fixed_roots(model, state=cell.state, belief=cell.belief, count=4)
    menus = _branch_menus(model, state=cell.state, belief=cell.belief, roots=roots)

    first = _matched_random_strategies(roots=roots, menus=menus, seed=123)
    second = _matched_random_strategies(roots=roots, menus=menus, seed=123)

    assert first == second
    assert first[0].root_action == COLLECT_ACTION
    for strategy, root_menus in zip(first, menus, strict=True):
        assert all(
            followup in root_menus[outcome]
            for outcome, followup in strategy.followups.items()
        )
        assert strategy_cost(
            model,
            state=cell.state,
            belief=cell.belief,
            strategy=strategy,
        ) >= 0.0


def test_deterministic_proposal_gate_runs_all_exact_controls() -> None:
    strategy_config = MushroomStrategyConfig(seed=24_131)
    provider = IndexedMushroomProvider(
        DeterministicIndexedMushroomModel(),
        strategy_config,
    )
    result = run_proposal_gate(provider, MushroomProposalGateConfig())

    assert all(result["mechanics"].values())
    assert all(result["contribution_gate"].values())
    assert len(result["records"]) == 32
    assert len(provider.physical_requests) == 32
    assert len(provider.invalid_responses) == 0
    assert result["endpoint_gate"]["collection_rate_at_least_threshold"] is False
