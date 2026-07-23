import numpy as np
import pytest

from environments.thyroid_workup import (
    COLLECT_BLOOD_ACTION,
    ThyroidWorkupModel,
)
from scripts.nonmyopic_thyroid_workup_oracle import (
    ThyroidQualificationConfig,
    exact_action_costs,
    run_qualification,
)
from scripts.audit_nonmyopic_thyroid_workup_oracle import audit
from scripts.nonmyopic_thyroid_workup_proposal_gate import (
    ThyroidProposalGateConfig,
    build_proposal_cells,
)
from scripts.nonmyopic_thyroid_workup_strategy import (
    DeterministicNamedThyroidModel,
    NamedThyroidProvider,
    ThyroidStrategyConfig,
    run_smoke,
)


def test_thyroid_cohort_and_acquisition_contract() -> None:
    model = ThyroidWorkupModel()
    assert len(model.targets) == 7_200
    assert {value: int(np.sum(model.targets == value)) for value in (1, 2, 3)} == {
        1: 166,
        2: 368,
        3: 6666,
    }
    assert COLLECT_BLOOD_ACTION in model.legal_actions(model.initial_state)
    assert "query:tsh" not in model.legal_actions(model.initial_state)
    collected = model.next_state(model.initial_state, COLLECT_BLOOD_ACTION)
    assert "query:tsh" in model.legal_actions(collected)
    assert model.expected_information_gain(model.initial_belief, COLLECT_BLOOD_ACTION) == 0.0


def test_exact_depth_two_prefers_blood_collection_at_initial_belief() -> None:
    model = ThyroidWorkupModel()
    d1 = exact_action_costs(
        model, state=model.initial_state, belief=model.initial_belief, depth=1
    )
    d2 = exact_action_costs(
        model, state=model.initial_state, belief=model.initial_belief, depth=2
    )
    assert min(d1, key=d1.get) != COLLECT_BLOOD_ACTION
    assert min(d2, key=d2.get) == COLLECT_BLOOD_ACTION


def test_tiny_thyroid_qualification_has_complete_paired_traces() -> None:
    summary = run_qualification(
        ThyroidQualificationConfig(
            num_trials=4,
            num_rounds=3,
            bootstrap_replicates=50,
            trial_concurrency=1,
        )
    )
    assert summary["mechanics"]["paired_truths_without_replacement"]
    assert summary["mechanics"]["all_traces_complete"]
    assert len(summary["traces"]["depth_one"]) == 4
    assert len(summary["traces"]["depth_two"]) == 4
    assert np.isfinite(summary["comparison"]["entropy_auc_gain"]["mean"])


def test_independent_audit_replays_tiny_qualification() -> None:
    summary = run_qualification(
        ThyroidQualificationConfig(
            num_trials=4,
            num_rounds=2,
            seed=123,
            bootstrap_replicates=20,
            trial_concurrency=1,
        )
    )

    audited = audit(summary)

    assert all(audited["mechanics"].values())
    assert audited["entropy_auc_gain"]["mean"] == pytest.approx(
        summary["comparison"]["entropy_auc_gain"]["mean"]
    )


def test_named_thyroid_smoke_has_fixed_collection_root_and_legal_actions() -> None:
    config = ThyroidStrategyConfig()
    result = run_smoke(
        NamedThyroidProvider(DeterministicNamedThyroidModel(), config), config
    )

    assert all(result["mechanics"].values())


def test_proposal_catalog_has_distinct_exact_collection_opportunities() -> None:
    model = ThyroidWorkupModel()
    cells = build_proposal_cells(model, ThyroidProposalGateConfig())

    assert len(cells) == 32
    assert len({cell.history for cell in cells}) == 32
    assert all(not cell.state.blood_collected for cell in cells)
