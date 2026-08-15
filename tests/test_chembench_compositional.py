from __future__ import annotations

import numpy as np
import pytest

from environments.chembench_mopen.compositional import (
    AtomicStructureOracleProposer,
    AuditedCompositionalPolicyPlanner,
    CompoundSignature,
    CompositionalEditCompiler,
    StructureParticleIndex,
)
from environments.chembench_mopen.mechanics import ModelBank, ProposalCache


def _components() -> tuple[
    ModelBank,
    CompositionalEditCompiler,
    StructureParticleIndex,
]:
    signatures = {
        "base": CompoundSignature("hill"),
        "warm": CompoundSignature("hill", temperature="arrhenius"),
        "warm_ph": CompoundSignature(
            "hill", temperature="arrhenius", ph_dependence="bell_curve"
        ),
    }
    particle_structure = ("base",) * 3 + ("warm",) * 3 + ("warm_ph",) * 4
    likelihoods = np.asarray(
        [
            [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]],
            [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]],
            [[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]],
            [[0.2, 0.7, 0.1], [0.2, 0.6, 0.2]],
            [[0.1, 0.8, 0.1], [0.1, 0.7, 0.2]],
            [[0.1, 0.7, 0.2], [0.1, 0.6, 0.3]],
            [[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]],
            [[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
            [[0.2, 0.1, 0.7], [0.2, 0.1, 0.7]],
            [[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
        ]
    )
    bank = ModelBank(
        likelihoods,
        np.arange(20, dtype=float).reshape(10, 2),
        tuple(f"particle-{index}" for index in range(10)),
        ("temperature", "ph"),
        ("temperature", "ph"),
        (0, 1, 2),
        live_cap=10,
        reserve_cap=0,
    )
    particles = StructureParticleIndex(
        particle_structure,
        inference_particles={"base": (0, 1, 2), "warm": (3, 4, 5), "warm_ph": (6, 7, 8)},
        truth_particles=(9,),
    )
    return bank, CompositionalEditCompiler(signatures), particles


def test_compiler_accepts_only_one_atomic_component_change() -> None:
    compiler = CompositionalEditCompiler(
        {
            "base": CompoundSignature("mm"),
            "competitive": CompoundSignature("mm", "competitive"),
            "noncompetitive": CompoundSignature("mm", "noncompetitive"),
            "hill": CompoundSignature("hill"),
            "hill_warm": CompoundSignature("hill", temperature="arrhenius"),
        }
    )
    assert compiler.compile("base", "competitive").operation == "add_inhibitor"
    assert compiler.compile("base", "hill").operation == "replace_core"
    assert compiler.compile("hill", "hill_warm").operation == "add_temperature"
    with pytest.raises(ValueError, match="remove then add"):
        compiler.compile("competitive", "noncompetitive")
    with pytest.raises(ValueError, match="exactly one"):
        compiler.compile("base", "hill_warm")


def test_structure_particle_index_rejects_truth_leakage() -> None:
    with pytest.raises(ValueError, match="overlap"):
        StructureParticleIndex(
            ("a",) * 4,
            inference_particles={"a": (0, 1, 2)},
            truth_particles=(2,),
        )


def test_oracle_adds_complete_neighbor_without_truth_particle() -> None:
    bank, compiler, particles = _components()
    proposer = AtomicStructureOracleProposer(bank, compiler, particles)
    state = bank.initial_state()
    first = proposer.propose(state, 0, 1, 7)
    assert first == (3, 4, 5)
    assert not particles.truth_particles.intersection(first)

    child = bank.transition(state, 0, 1, first)
    second = proposer.propose(child, 1, 2, 8)
    assert second == (6, 7, 8)
    assert not particles.truth_particles.intersection(second)


def test_audited_planner_records_atomic_transitions() -> None:
    bank, compiler, particles = _components()
    proposer = AtomicStructureOracleProposer(bank, compiler, particles)
    planner = AuditedCompositionalPolicyPlanner(
        bank,
        ProposalCache(proposer),
        (9,),
        compiler=compiler,
        particles=particles,
        seed=17,
    )
    result = planner.evaluate_policy_level(2, execution_budget=2)
    assert np.isfinite(result["expected_terminal_mse"])
    proposed = [row for row in planner.transition_audit.values() if row["proposal"]]
    assert proposed
    assert all(len(row["proposal"]) == 3 for row in proposed)
    assert all(row["edit"] is not None for row in proposed)
    assert all(not row["truth_particle_proposed"] for row in proposed)
