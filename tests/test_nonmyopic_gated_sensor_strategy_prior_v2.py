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


def test_v3_prompt_supplies_branch_beliefs_without_oracle_scores() -> None:
    model = GatedSensorModel()
    config = IndexedStrategyConfig(
        interface_version="indexed_branch_v3",
        num_trials=1,
        num_rounds=2,
        num_strategies=4,
        bootstrap_replicates=100,
        trial_concurrency=1,
    )
    provider = IndexedStrategyProvider(DeterministicIndexedModel(), config)
    roots = _fixed_roots(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        horizon=2,
        count=4,
    )
    menus = _branch_menus(model, state=model.initial_state, roots=roots, horizon=2)

    messages = provider._messages(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        history=(),
        roots=roots,
        menus=menus,
    )
    slots = json.loads(messages[-1]["content"].split("INDEXED_SLOTS=", 1)[1])

    assert "branch_beliefs" in slots[0]
    assert slots[0]["branch_beliefs"]["none"]["probability"] == pytest.approx(1.0)
    measurement_slot = slots[-1]
    assert measurement_slot["root_action"] == "screen:bit-0"
    positive_marginals = {
        row["predicate"]: row["p_true"]
        for row in measurement_slot["branch_beliefs"]["positive"]["predicate_marginals"]
    }
    negative_marginals = {
        row["predicate"]: row["p_true"]
        for row in measurement_slot["branch_beliefs"]["negative"]["predicate_marginals"]
    }
    assert positive_marginals["bit-0"] == pytest.approx(0.65)
    assert negative_marginals["bit-0"] == pytest.approx(0.35)
    branch_payload = json.dumps(slots, sort_keys=True).lower()
    assert "planning_score" not in branch_payload
    assert '"eig"' not in branch_payload
    assert "truth_index" not in branch_payload


def test_v2_prompt_does_not_add_branch_beliefs() -> None:
    model = GatedSensorModel()
    config = IndexedStrategyConfig(
        num_trials=1,
        num_rounds=2,
        num_strategies=4,
        bootstrap_replicates=100,
        trial_concurrency=1,
    )
    provider = IndexedStrategyProvider(DeterministicIndexedModel(), config)
    roots = _fixed_roots(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        horizon=2,
        count=4,
    )
    menus = _branch_menus(model, state=model.initial_state, roots=roots, horizon=2)

    messages = provider._messages(
        model,
        state=model.initial_state,
        belief=model.initial_belief,
        history=(),
        roots=roots,
        menus=menus,
    )
    slots = json.loads(messages[-1]["content"].split("INDEXED_SLOTS=", 1)[1])

    assert all("branch_beliefs" not in slot for slot in slots)
