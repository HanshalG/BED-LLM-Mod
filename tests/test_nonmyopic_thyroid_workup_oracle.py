import json
from pathlib import Path

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
from scripts.audit_nonmyopic_thyroid_workup_proposal_gate import audit as audit_proposal_gate
from scripts.audit_nonmyopic_thyroid_workup_confirmation import audit as audit_confirmation
from scripts.nonmyopic_thyroid_workup_proposal_gate import (
    ThyroidProposalGateConfig,
    build_proposal_cells,
)
from scripts.nonmyopic_thyroid_workup_confirmation import (
    ThyroidConfirmationConfig,
    _run_arm,
)
from scripts.nonmyopic_thyroid_workup_strategy import (
    DeterministicNamedThyroidModel,
    DeterministicUtilityThyroidModel,
    NamedThyroidProvider,
    ThyroidStrategyConfig,
    branch_menus,
    continuation_utility_cards,
    fixed_roots,
    run_smoke,
)
from scripts.nonmyopic_thyroid_workup_robust_smoke import (
    build_robust_cells,
    run_smoke as run_robust_smoke,
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


def test_banked_named_proposal_gate_passes_independent_replay() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root
            / "results/nonmyopic/thyroid_workup_26b_proposal_gate_20260723/GATE.json"
        ).read_text(encoding="utf-8")
    )

    assert audit_proposal_gate(payload)["passed"]


def test_confirmation_exact_arm_produces_complete_legal_trace() -> None:
    model = ThyroidWorkupModel()
    config = ThyroidConfirmationConfig()
    trace = _run_arm(
        model,
        arm="depth_two",
        trial_index=0,
        truth_index=0,
        config=config,
        provider=None,
    )

    assert len(trace["steps"]) == 8
    assert trace["steps"][0]["action"] == COLLECT_BLOOD_ACTION
    assert np.isfinite(trace["entropy_auc"])


def test_late_state_smoke_covers_shrinking_menus() -> None:
    model = ThyroidWorkupModel()
    cells = build_robust_cells(model, seed=24_157)
    config = ThyroidStrategyConfig(seed=24_157)
    result = run_robust_smoke(
        NamedThyroidProvider(DeterministicNamedThyroidModel(), config), seed=24_157
    )

    assert {len(history) for _, _, _, history in cells} == set(range(7))
    assert all(result["mechanics"].values())


def test_utility_cards_are_branch_local_and_select_best_continuations() -> None:
    model = ThyroidWorkupModel()
    state = model.initial_state
    belief = model.initial_belief
    roots = fixed_roots(model, state=state, belief=belief)
    menus = branch_menus(model, state=state, belief=belief, roots=roots)
    cards = continuation_utility_cards(
        model, belief=belief, roots=roots, menus=menus
    )
    config = ThyroidStrategyConfig(utility_summary_mode="branch_local_expected_entropy")
    provider = NamedThyroidProvider(DeterministicUtilityThyroidModel(), config)

    cell = provider.propose(
        model, cell_index=0, state=state, belief=belief, history=()
    )

    assert cell.strategies[0].root_action == COLLECT_BLOOD_ACTION
    assert cell.strategies[0].followups == {"none": "query:tsh"}
    for strategy in cell.strategies:
        for outcome, followup in strategy.followups.items():
            expected = min(
                enumerate(cards[strategy.root_action][outcome]),
                key=lambda item: (item[1]["expected_class_entropy"], item[0]),
            )[1]["action"]
            assert followup == expected
    request = provider.physical_requests[0]
    assert request["utility_summary_mode"] == "branch_local_expected_entropy"
    assert request["continuation_utility_cards"] == cards
    assert "truth_index" not in json.dumps(request)


def test_utility_grounded_late_state_smoke_keeps_actions_legal() -> None:
    config = ThyroidStrategyConfig(
        seed=24_159, utility_summary_mode="branch_local_expected_entropy"
    )
    result = run_robust_smoke(
        NamedThyroidProvider(DeterministicUtilityThyroidModel(), config), seed=24_159
    )

    assert all(result["mechanics"].values())


def test_thyroid_strategy_rejects_unknown_utility_summary_mode() -> None:
    config = ThyroidStrategyConfig(
        utility_summary_mode="oracle"  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="utility summary mode"):
        config.validate()


class _RepeatingRootThyroidModel:
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        del temperature
        assert num_responses == 1
        prompt = next(
            message["content"] for message in messages if "ROOTS=" in message["content"]
        )
        slots = json.loads(prompt.split("ROOTS=", 1)[1])
        return [
            json.dumps(
                {
                    slot["root_action"]: {
                        branch["outcome"]: slot["root_action"]
                        for branch in slot["branches"]
                    }
                    for slot in slots
                }
            )
        ]


def test_projection_replaces_only_invalid_branches_with_best_legal_actions() -> None:
    model = ThyroidWorkupModel()
    config = ThyroidStrategyConfig(
        utility_summary_mode="branch_local_expected_entropy",
        project_invalid_after_retries=True,
    )
    provider = NamedThyroidProvider(_RepeatingRootThyroidModel(), config)

    cell = provider.propose(
        model,
        cell_index=0,
        state=model.initial_state,
        belief=model.initial_belief,
        history=(),
    )

    assert len(provider.invalid_responses) == 2
    assert len(provider.projected_responses) == 1
    assert provider.physical_requests[0]["projected"]
    cards = provider.physical_requests[0]["continuation_utility_cards"]
    for strategy in cell.strategies:
        for outcome, followup in strategy.followups.items():
            best = min(
                enumerate(cards[strategy.root_action][outcome]),
                key=lambda item: (item[1]["expected_class_entropy"], item[0]),
            )[1]["action"]
            assert followup == best
            assert followup != strategy.root_action


def test_projection_requires_utility_summaries() -> None:
    config = ThyroidStrategyConfig(project_invalid_after_retries=True)

    with pytest.raises(ValueError, match="projection requires"):
        config.validate()


def test_projected_confirmation_audit_treats_machine_zero_entropies_as_ties() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "results/nonmyopic/thyroid_workup_projected_utility_gpt54mini_confirmation_20260723/CONFIRMATION.json"
    )
    if not path.exists():
        pytest.skip("projected confirmation artifact is not present")

    audited = audit_confirmation(json.loads(path.read_text(encoding="utf-8")))

    assert audited["mechanics"]["all_projections_are_exact_legal_minima"]
    assert audited["projection"]["max_expected_entropy_excess"] < 1e-12


def test_banked_gpt_confirmation_failure_replays_independently() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root
            / "results/nonmyopic/thyroid_workup_gpt54mini_confirmation_20260723/CONFIRMATION.json"
        ).read_text(encoding="utf-8")
    )
    audited = audit_confirmation(payload)

    assert audited["audit_valid"]
    assert not audited["registered_scientific_gate_recomputed"]
