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
from scripts.nonmyopic_heart_workup_confirmation import (
    HeartConfirmationConfig,
    run_confirmation,
)
from scripts.nonmyopic_heart_workup_strategy import (
    DeterministicIndexedHeartModel,
    DeterministicUtilityHeartModel,
    HeartStrategyConfig,
    IndexedHeartProvider,
    branch_menus,
    compile_indexed_cell,
    continuation_utility_cards,
    fixed_roots,
    project_indexed_cell,
    run_smoke,
)
from scripts.audit_nonmyopic_heart_workup_confirmation import audit as audit_confirmation


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


def test_heart_prompt_ends_with_root_local_index_limits() -> None:
    model = HeartWorkupModel()
    config = HeartStrategyConfig()
    provider = IndexedHeartProvider(DeterministicIndexedHeartModel(), config)
    roots = fixed_roots(
        model, state=model.initial_state, belief=model.initial_belief, count=4
    )
    menus = branch_menus(
        model, state=model.initial_state, belief=model.initial_belief, roots=roots
    )

    messages = provider._messages(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        history=(),
        roots=roots,
        menus=menus,
    )
    final_line = messages[-1]["content"].splitlines()[-1]
    limits = json.loads(final_line.split("=", 1)[1])

    assert final_line.startswith("FINAL_OUTPUT_LIMITS=")
    assert limits["r0"] == {
        "exact_items": 1,
        "each_integer_min": 0,
        "each_integer_max": 12,
    }
    assert limits["r1"]["each_integer_max"] == 4


def test_heart_utility_cards_match_exact_branch_entropies() -> None:
    model = HeartWorkupModel()
    belief = model.initial_belief
    roots = fixed_roots(model, state=model.initial_state, belief=belief, count=4)
    menus = branch_menus(
        model, state=model.initial_state, belief=belief, roots=roots
    )
    cards = continuation_utility_cards(
        model, belief=belief, roots=roots, menus=menus
    )
    posterior = model.posterior(belief, roots[0], None)
    first_action = menus[0]["none"][0]

    assert cards[0]["none"][0] == {
        "index": 0,
        "action": first_action,
        "expected_class_entropy": round(
            model.expected_target_entropy(posterior, first_action), 8
        ),
        "one_step_information_gain": round(
            model.target_entropy(posterior)
            - model.expected_target_entropy(posterior, first_action),
            8,
        ),
    }


def test_heart_utility_prompt_exposes_branch_local_cards_only_when_enabled() -> None:
    model = HeartWorkupModel()
    roots = fixed_roots(
        model, state=model.initial_state, belief=model.initial_belief, count=4
    )
    menus = branch_menus(
        model, state=model.initial_state, belief=model.initial_belief, roots=roots
    )
    plain = IndexedHeartProvider(
        DeterministicIndexedHeartModel(), HeartStrategyConfig()
    )._messages(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        history=(),
        roots=roots,
        menus=menus,
    )
    grounded = IndexedHeartProvider(
        DeterministicIndexedHeartModel(),
        HeartStrategyConfig(
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


def test_heart_projection_preserves_valid_indexes_and_repairs_only_invalid() -> None:
    model = HeartWorkupModel()
    belief = model.initial_belief
    roots = fixed_roots(model, state=model.initial_state, belief=belief, count=4)
    menus = branch_menus(
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
    expected_index = min(range(len(choices)), key=lambda index: (scores[index], index))

    assert len(events) == 1
    assert events[0]["replacement_index"] == expected_index
    assert strategies[0].followups["none"] == choices[expected_index]
    assert strategies[1].followups[next(iter(menus[1]))] == next(iter(menus[1].values()))[0]


def test_heart_projection_runs_only_after_bounded_retry() -> None:
    class AlwaysInvalid:
        def chat_complete(self, messages, temperature, num_responses=1):
            del messages, temperature
            assert num_responses == 1
            return ['{"r0":[999],"r1":[],"r2":[],"r3":[]}']

    model = HeartWorkupModel()
    config = HeartStrategyConfig(
        utility_summary_mode="branch_local_expected_entropy",
        project_invalid_after_retries=True,
    )
    provider = IndexedHeartProvider(AlwaysInvalid(), config)
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


def test_heart_strategy_rejects_unknown_utility_mode() -> None:
    with pytest.raises(ValueError, match="utility summary"):
        HeartStrategyConfig(utility_summary_mode="unknown").validate()


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
    assert all(gate["contribution_gate"].values())
    assert np.isfinite(gate["comparisons"]["matched_random_minus_llm_cost"]["mean"])
    assert gate["endpoint_gate"]["workup_rate_at_least_threshold"] is False


def test_deterministic_grounded_heart_confirmation_replays_independently() -> None:
    strategy_config = HeartStrategyConfig(
        utility_summary_mode="branch_local_expected_entropy",
        project_invalid_after_retries=True,
        allow_fewer_roots_when_exhausted=True,
    )
    provider = IndexedHeartProvider(
        DeterministicUtilityHeartModel(), strategy_config
    )
    result = run_confirmation(provider, HeartConfirmationConfig())
    result["strategy_config"] = {
        "num_strategies": 4,
        "seed": 24_167,
        "temperature": 0.0,
        "validation_retries": 1,
        "max_new_tokens": 128,
        "utility_summary_mode": "branch_local_expected_entropy",
        "project_invalid_after_retries": True,
        "allow_fewer_roots_when_exhausted": True,
    }
    replay = audit_confirmation(result)

    assert all(result["mechanics"].values())
    assert all(result["endpoint_gate"].values())
    assert replay["audit_valid"] is True
    assert replay["registered_scientific_gate_recomputed"] is True
