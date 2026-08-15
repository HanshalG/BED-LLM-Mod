from __future__ import annotations

import json

import numpy as np
import pytest

from environments.chembench_mopen.factored import (
    FactoredModelBank,
    FactoredPolicyLadderPlanner,
    RegistryEditCompiler,
    TypedRegistryOracleProposer,
    registry_signature,
)
from environments.chembench_mopen.mechanics import BankedProposer, ProposalCache


INITIAL_NAMES = (
    "c0_michaelis_menten",
    "c1_competitive_inhibition",
    "c2_product_inhibition",
    "c3_arrhenius_temperature",
    "c5_pingpong_bisubstrate",
    "c6_uncompetitive_inhibition",
    "c7_substrate_inhibition",
    "c8_hill_cooperativity",
    "c9_noncompetitive_inhibition",
)
OUTSIDE_NAMES = (
    "c10_mm_competitive_arrhenius",
    "c23_pingpong_arrhenius",
    "c33_hill_competitive",
    "c65_ordered_bi_bi",
)


def _bank() -> FactoredModelBank:
    names = INITIAL_NAMES + OUTSIDE_NAMES
    likelihoods = np.empty((len(names), 3, 3), dtype=float)
    for model in range(len(names)):
        if model < len(INITIAL_NAMES):
            likelihoods[model, 0] = (0.49, 0.49, 0.02)
            likelihoods[model, 1] = (0.60 - model * 0.01, 0.30 + model * 0.01, 0.10)
            likelihoods[model, 2] = (0.20, 0.60 - model * 0.01, 0.20 + model * 0.01)
        else:
            outcome = (model - len(INITIAL_NAMES)) % 3
            for action in range(3):
                row = np.full(3, 0.05)
                row[(outcome + action) % 3] = 0.90
                likelihoods[model, action] = row
    features = np.stack(
        (
            np.linspace(0.0, 1.0, len(names)),
            np.linspace(1.0, 0.0, len(names)) ** 2,
        ),
        axis=1,
    )
    return FactoredModelBank(
        likelihoods,
        features,
        model_names=names,
        action_names=("C_A=low", "C_I=high", "T=high"),
        action_groups=("C_A", "C_I", "T"),
        initial_support=tuple(range(len(INITIAL_NAMES))),
        evidence_slots=8,
        diversity_slots=4,
    )


def test_registry_signatures_and_edits_are_typed_and_round_trip() -> None:
    assert registry_signature("c10_mm_competitive_arrhenius").core == "michaelis_menten"
    assert registry_signature("c10_mm_competitive_arrhenius").modifiers == (
        "arrhenius",
        "competitive_inhibition",
    )
    assert registry_signature("c90_ordered_bi_bi_noncomp").core == "ordered_bi_bi"
    assert registry_signature("c92_allosteric_act_feedback").modifiers == ("product_feedback",)
    with pytest.raises(ValueError, match="unknown"):
        registry_signature("c999_magic_curve")

    bank = _bank()
    compiler = RegistryEditCompiler(bank)
    for candidate in range(len(INITIAL_NAMES), bank.num_models):
        edit = compiler.compile(bank.initial_support, candidate)
        compiler.validate(edit)
        assert edit.candidate == candidate
        assert edit.operation in compiler.OPERATIONS


def test_trigger_is_source_calibrated_and_nontrigger_makes_no_request() -> None:
    bank = _bank()
    assert bank.calibration_false_trigger_mass <= 0.10 + 1e-12
    state = bank.initial_state()
    assert bank.expansion_diagnostic(state, 0, 0).triggered is False
    assert bank.expansion_diagnostic(state, 0, 2).triggered is True

    class CountingProposer:
        mode = "counting"

        def __init__(self) -> None:
            self.calls = 0

        def propose(self, state, action, outcome, seed):
            del state, action, outcome, seed
            self.calls += 1
            return (len(INITIAL_NAMES),)

    proposer = CountingProposer()
    cache = ProposalCache(proposer)
    planner = FactoredPolicyLadderPlanner(
        bank,
        cache,
        tuple(range(len(INITIAL_NAMES), bank.num_models)),
        seed=123,
    )
    root = planner.initial_state()
    quiet = planner.transition(root, 0, 0)
    assert proposer.calls == 0
    assert quiet.inference.expansion_triggered is False
    expanded = planner.transition(root, 0, 2)
    assert proposer.calls == 1
    assert expanded.inference.expansion_triggered is True
    assert len(expanded.inference.last_edits) == 1


def test_triggered_transition_caps_support_prunes_and_never_reproposes() -> None:
    bank = _bank()
    proposer = TypedRegistryOracleProposer(bank)
    parent = bank.initial_state()
    proposal = proposer.propose(parent, 0, 2, 7)
    assert len(proposal) == 4
    child = bank.transition(parent, 0, 2, proposal)
    assert len(child.represented_models) == 12
    assert len(child.last_pruned) == 1
    assert len(child.last_edits) == 4
    assert set(proposal).issubset(child.tried)
    assert len({bank.compiler.core_family(model) for model in child.represented_models}) >= 3
    assert proposer.propose(child, 0, 2, 8) == ()
    assert len(child.tried) == len(set(child.tried))


def test_residual_report_is_pool_wide_public_and_category_native() -> None:
    bank = _bank()
    child = bank.transition(
        bank.initial_state(),
        0,
        2,
        TypedRegistryOracleProposer(bank).propose(bank.initial_state(), 0, 2, 1),
    )
    report = bank.residual_report(child, remaining_budget=2)
    assert len(report["models"]) == len(child.represented_models)
    assert report["latest"]["outcome"] == 2
    assert report["latest"]["expansion_triggered"] is True
    assert report["remaining_budget"] == 2
    assert "truth" not in json.dumps(report).lower()
    assert all("categorical_nll" in row for row in report["models"])


def test_factored_policy_ladder_is_calibrated_nonworsening_and_replayable() -> None:
    bank = _bank()
    particles = tuple(range(len(INITIAL_NAMES), bank.num_models))
    cache = ProposalCache(TypedRegistryOracleProposer(bank))
    planner = FactoredPolicyLadderPlanner(bank, cache, particles, seed=456)
    results = [planner.evaluate_policy_level(level, execution_budget=3) for level in (1, 2, 3)]
    for result in results:
        assert result["planned_value"] == pytest.approx(
            result["expected_terminal_mse"], abs=1e-12
        )
    assert results[1]["planned_value"] <= results[0]["planned_value"] + 1e-12
    assert results[2]["planned_value"] <= results[1]["planned_value"] + 1e-12
    assert planner.transition_audit
    assert any(item["triggered"] for item in planner.transition_audit.values())
    assert any(not item["triggered"] for item in planner.transition_audit.values())

    records = cache.records
    replay_cache = ProposalCache(
        BankedProposer(cache.source_mode, records),
        source_mode=cache.source_mode,
    )
    replay = FactoredPolicyLadderPlanner(bank, replay_cache, particles, seed=456)
    assert replay.evaluate_policy_level(3, execution_budget=3) == results[2]
    assert replay.transition_audit == planner.transition_audit
