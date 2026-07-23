import json

from environments.mushroom_feature_acquisition import COLLECT_ACTION, MushroomFeatureModel
from scripts.nonmyopic_mushroom_strategy import (
    DeterministicIndexedMushroomModel,
    IndexedMushroomProvider,
    MushroomStrategyConfig,
    _branch_menus,
    _fixed_roots,
    build_smoke_cells,
    compile_indexed_cell,
    run_smoke,
)


def test_fixed_roots_cover_collection_and_three_strong_field_queries() -> None:
    model = MushroomFeatureModel()
    roots = _fixed_roots(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        count=4,
    )

    assert roots[0] == COLLECT_ACTION
    assert len(roots) == len(set(roots)) == 4
    assert all(root in model.legal_actions(model.initial_state) for root in roots)


def test_indexed_compiler_maps_every_dynamic_branch_to_a_legal_query() -> None:
    model = MushroomFeatureModel()
    roots = _fixed_roots(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        count=4,
    )
    menus = _branch_menus(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        roots=roots,
    )
    response = json.dumps(
        {"choices": [[0 for _branch in root_menus] for root_menus in menus]}
    )
    strategies = compile_indexed_cell(response, roots=roots, menus=menus)

    assert len(strategies) == 4
    assert all(
        followup in model.legal_actions(model.next_state(model.initial_state, strategy.root_action))
        for strategy in strategies
        for followup in strategy.followups.values()
    )
    assert all(
        model.action_feature(followup) not in {"cap-shape", "cap-surface", "cap-color", "population", "habitat"}
        for followup in strategies[0].followups.values()
    )


def test_prompt_exposes_branch_class_probabilities_without_scores_or_truth() -> None:
    model = MushroomFeatureModel()
    config = MushroomStrategyConfig()
    provider = IndexedMushroomProvider(DeterministicIndexedMushroomModel(), config)
    _, state, belief, history = build_smoke_cells(model, seed=config.seed)[0]
    roots = _fixed_roots(model, state=state, belief=belief, count=4)
    menus = _branch_menus(model, state=state, belief=belief, roots=roots)
    messages = provider._messages(
        model,
        state=state,
        belief=belief,
        history=history,
        roots=roots,
        menus=menus,
    )
    prompt = messages[-1]["content"]

    assert "p_edible" in prompt
    assert "p_poisonous" in prompt
    assert "outcome_label" in prompt
    assert "planning_score" not in prompt
    assert '"eig"' not in prompt.lower()
    assert "truth_index" not in prompt


def test_dry_smoke_completes_ten_distinct_cells_without_repairs() -> None:
    config = MushroomStrategyConfig()
    provider = IndexedMushroomProvider(DeterministicIndexedMushroomModel(), config)
    result = run_smoke(provider, config)

    assert all(result["mechanics"].values())
    assert result["provider"] == {
        "accepted_requests": 10,
        "invalid_responses": 0,
        "physical_requests": 10,
    }
    assert len({record["truth_index"] for record in result["records"]}) == 10
