import json

import pytest

from environments.mushroom_feature_acquisition import COLLECT_ACTION, MushroomFeatureModel
from scripts.nonmyopic_mushroom_strategy import (
    DeterministicIndexedMushroomModel,
    IndexedMushroomProvider,
    MushroomStrategyConfig,
    _branch_menus,
    _fixed_roots,
    build_smoke_cells,
    compile_indexed_cell,
    continuation_predictive_evidence,
    continuation_utility_cards,
    project_indexed_cell,
    run_smoke,
)
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError


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
        {f"r{slot}": [0] * len(root_menus) for slot, root_menus in enumerate(menus)}
    )
    strategies = compile_indexed_cell(response, roots=roots, menus=menus)

    assert len(strategies) == 4
    assert all(
        followup in model.legal_actions(model.next_state(model.initial_state, strategy.root_action))
        for strategy in strategies
        for followup in strategy.followups.values()
    )
    assert all(
        model.action_feature(followup)
        not in {"cap-shape", "cap-surface", "cap-color", "population", "habitat"}
        for followup in strategies[0].followups.values()
    )


def test_indexed_compiler_rejects_nested_or_wrong_length_choices() -> None:
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

    with pytest.raises(StrategyProposalError, match="exactly keys"):
        compile_indexed_cell(
            json.dumps({"choices": [[0] for _root in roots]}),
            roots=roots,
            menus=menus,
        )
    with pytest.raises(StrategyProposalError, match="exactly"):
        compile_indexed_cell(
            json.dumps({"r0": [0], "r1": [0], "r2": [0], "r3": [0]}),
            roots=roots,
            menus=menus,
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
    assert '"r0":[' in prompt
    assert '"index":0' in prompt
    assert prompt.count('"menu":[') == 4
    assert "planning_score" not in prompt
    assert '"eig"' not in prompt.lower()
    assert "truth_index" not in prompt


def test_mushroom_utility_cards_match_exact_branch_entropies() -> None:
    model = MushroomFeatureModel()
    belief = model.initial_belief
    roots = _fixed_roots(
        model, state=model.initial_state, belief=belief, count=4
    )
    menus = _branch_menus(
        model, state=model.initial_state, belief=belief, roots=roots
    )
    cards = continuation_utility_cards(
        model, belief=belief, roots=roots, menus=menus
    )
    posterior = model.posterior(belief, roots[0], None)
    first_action = menus[0]["none"][0]

    assert cards[0]["none"][0]["index"] == 0
    assert cards[0]["none"][0]["feature"] == model.action_feature(first_action)
    assert cards[0]["none"][0]["expected_class_entropy"] == round(
        model.expected_target_entropy(posterior, first_action), 8
    )


def test_mushroom_utility_prompt_is_opt_in() -> None:
    model = MushroomFeatureModel()
    roots = _fixed_roots(
        model, state=model.initial_state, belief=model.initial_belief, count=4
    )
    menus = _branch_menus(
        model, state=model.initial_state, belief=model.initial_belief, roots=roots
    )
    plain = IndexedMushroomProvider(
        DeterministicIndexedMushroomModel(), MushroomStrategyConfig()
    )._messages(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        history=(),
        roots=roots,
        menus=menus,
    )
    grounded = IndexedMushroomProvider(
        DeterministicIndexedMushroomModel(),
        MushroomStrategyConfig(
            utility_summary_mode="branch_local_expected_entropy"
        ),
    )._messages(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        history=(),
        roots=roots,
        menus=menus,
    )

    assert "continuation_utility" not in plain[-1]["content"]
    assert "continuation_utility" in grounded[-1]["content"]


def test_mushroom_predictive_evidence_contains_no_precomputed_scores() -> None:
    model = MushroomFeatureModel()
    belief = model.initial_belief
    roots = _fixed_roots(
        model, state=model.initial_state, belief=belief, count=4
    )
    menus = _branch_menus(
        model, state=model.initial_state, belief=belief, roots=roots
    )
    evidence = continuation_predictive_evidence(
        model, belief=belief, roots=roots, menus=menus
    )
    first = evidence[0]["none"][0]

    assert first["index"] == 0
    assert first["feature"] == model.action_feature(menus[0]["none"][0])
    assert sum(item["probability"] for item in first["predictive_outcomes"]) == pytest.approx(
        1.0, abs=1e-7
    )
    assert all(
        item["p_edible_after"] + item["p_poisonous_after"]
        == pytest.approx(1.0, abs=1e-7)
        for item in first["predictive_outcomes"]
    )
    assert "expected_class_entropy" not in json.dumps(evidence)
    assert "information_gain" not in json.dumps(evidence)


def test_mushroom_predictive_prompt_exposes_evidence_but_not_answer() -> None:
    model = MushroomFeatureModel()
    roots = _fixed_roots(
        model, state=model.initial_state, belief=model.initial_belief, count=4
    )
    menus = _branch_menus(
        model, state=model.initial_state, belief=model.initial_belief, roots=roots
    )
    messages = IndexedMushroomProvider(
        DeterministicIndexedMushroomModel(),
        MushroomStrategyConfig(
            utility_summary_mode="branch_local_predictive_evidence"
        ),
    )._messages(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        history=(),
        roots=roots,
        menus=menus,
    )
    prompt = messages[-1]["content"]

    assert "continuation_predictive_evidence" in prompt
    assert "predictive_outcomes" in prompt
    assert "p_edible_after" in prompt
    assert "expected_class_entropy" not in prompt
    assert "one_step_information_gain" not in prompt
    assert "continuation_utility" not in prompt


def test_mushroom_projection_preserves_valid_indexes_and_repairs_invalid() -> None:
    model = MushroomFeatureModel()
    belief = model.initial_belief
    roots = _fixed_roots(
        model, state=model.initial_state, belief=belief, count=4
    )
    menus = _branch_menus(
        model, state=model.initial_state, belief=belief, roots=roots
    )
    payload = {
        f"r{slot}": [0] * len(root_menus)
        for slot, root_menus in enumerate(menus)
    }
    payload["r0"][0] = 999

    strategies, events = project_indexed_cell(
        json.dumps(payload),
        model=model,
        belief=belief,
        roots=roots,
        menus=menus,
    )
    posterior = model.posterior(belief, roots[0], None)
    choices = menus[0]["none"]
    scores = [model.expected_target_entropy(posterior, action) for action in choices]
    minimum = min(scores)
    expected_index = next(
        index
        for index, score in enumerate(scores)
        if score <= minimum + 1e-15
    )

    assert len(events) == 1
    assert events[0]["replacement_index"] == expected_index
    assert strategies[0].followups["none"] == choices[expected_index]
    assert (
        strategies[1].followups[next(iter(menus[1]))]
        == next(iter(menus[1].values()))[0]
    )


def test_mushroom_projection_runs_only_after_bounded_retry() -> None:
    class AlwaysInvalid:
        def chat_complete(self, messages, temperature, num_responses=1):
            del messages, temperature
            assert num_responses == 1
            return ['{"r0":[999],"r1":[],"r2":[],"r3":[]}']

    model = MushroomFeatureModel()
    config = MushroomStrategyConfig(
        utility_summary_mode="branch_local_expected_entropy",
        project_invalid_after_retries=True,
    )
    provider = IndexedMushroomProvider(AlwaysInvalid(), config)
    cell = provider.propose(
        model,
        cell_index=0,
        state=model.initial_state,
        belief=model.initial_belief,
        history=(),
    )

    assert len(cell.strategies) == 4
    assert len(provider.invalid_responses) == 2
    assert len(provider.projected_responses) == 1
    assert provider.projected_responses[0]["projection_events"]


def test_mushroom_strategy_rejects_unknown_utility_mode() -> None:
    with pytest.raises(ValueError, match="utility summary"):
        MushroomStrategyConfig(utility_summary_mode="unknown").validate()


def test_dry_smoke_completes_ten_distinct_cells_without_repairs() -> None:
    config = MushroomStrategyConfig()
    provider = IndexedMushroomProvider(DeterministicIndexedMushroomModel(), config)
    result = run_smoke(provider, config)

    assert all(result["mechanics"].values())
    assert result["provider"] == {
        "accepted_requests": 10,
        "invalid_responses": 0,
        "projected_cells": 0,
        "projected_branches": 0,
        "physical_requests": 10,
    }
    assert len({record["truth_index"] for record in result["records"]}) == 10
