from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from environments.chembench_mopen.compositional import (
    AuditedCompositionalPolicyPlanner,
    AtomicStructureOracleProposer,
    CompoundSignature,
    CompositionalEditCompiler,
    StructureParticleIndex,
)
from environments.chembench_mopen.costed import CostedCompositionalPolicyPlanner
from environments.chembench_mopen.mechanics import (
    ModelBank,
    ProposalCache,
    _proposal_outcome_value,
    proposal_key,
)
from scripts.chembench_costed_repeat_corridor import (
    BASE_ASSAY_NAMES,
    SCHEMA_VERSION,
    WELL_BUDGET,
    _audit_summary,
    _canonical_hash,
    _proposal_records_hash,
    _replay_execution,
    _validated_shard_slice,
    costed_likelihoods,
)


def test_costed_thresholds_ignore_heldout_truth_rows() -> None:
    inference = np.linspace(0.1, 8.0, 60).reshape(6, 10)
    truth = np.full((2, 10), 2.0)
    first_likelihoods, first_thresholds, actions = costed_likelihoods(
        np.vstack((inference, truth)), num_inference_particles=6
    )
    changed_likelihoods, changed_thresholds, changed_actions = costed_likelihoods(
        np.vstack((inference, truth * 100.0)), num_inference_particles=6
    )
    assert np.array_equal(first_thresholds, changed_thresholds)
    assert np.array_equal(first_likelihoods[:6], changed_likelihoods[:6])
    assert not np.array_equal(first_likelihoods[6:], changed_likelihoods[6:])
    assert actions == changed_actions
    assert len(actions) == len(BASE_ASSAY_NAMES) * 3


def _planner(*, well_budget: int = WELL_BUDGET) -> CostedCompositionalPolicyPlanner:
    base_means = np.asarray(
        [np.linspace(0.5 + index, 2.0 + index, 10) for index in range(10)]
    )
    likelihoods, _, actions = costed_likelihoods(
        base_means, num_inference_particles=9
    )
    bank = ModelBank(
        likelihoods,
        np.arange(30, dtype=float).reshape(10, 3),
        tuple(f"particle-{index}" for index in range(10)),
        tuple(action.name for action in actions),
        tuple(action.base_name for action in actions),
        (0, 1, 2),
        live_cap=10,
        reserve_cap=0,
    )
    compiler = CompositionalEditCompiler(
        {
            "base": CompoundSignature("hill"),
            "warm": CompoundSignature("hill", temperature="arrhenius"),
            "warm_ph": CompoundSignature(
                "hill", temperature="arrhenius", ph_dependence="bell_curve"
            ),
        }
    )
    particles = StructureParticleIndex(
        ("base",) * 3 + ("warm",) * 3 + ("warm_ph",) * 4,
        inference_particles={
            "base": (0, 1, 2),
            "warm": (3, 4, 5),
            "warm_ph": (6, 7, 8),
        },
        truth_particles=(9,),
    )
    return CostedCompositionalPolicyPlanner(
        bank,
        ProposalCache(AtomicStructureOracleProposer(bank, compiler, particles)),
        (9,),
        compiler=compiler,
        particles=particles,
        actions=actions,
        well_budget=well_budget,
        seed=23,
    )


class _UncachedTransitionPlanner(CostedCompositionalPolicyPlanner):
    def transition(self, state, action, outcome):
        return AuditedCompositionalPolicyPlanner.transition(
            self, state, action, outcome
        )


def _uncached_planner(*, well_budget: int) -> _UncachedTransitionPlanner:
    cached = _planner(well_budget=well_budget)
    return _UncachedTransitionPlanner(
        cached.bank,
        ProposalCache(
            AtomicStructureOracleProposer(
                cached.bank, cached.compiler, cached.particles
            )
        ),
        cached.particle_indices,
        compiler=cached.compiler,
        particles=cached.particles,
        actions=cached.actions,
        well_budget=well_budget,
        seed=cached.seed,
    )


def test_costed_planner_removes_repeat_variants_and_respects_budget() -> None:
    planner = _planner()
    available = tuple(range(planner.bank.num_actions))
    for action in planner.feasible_actions(available, WELL_BUDGET):
        remainder = planner.remaining_actions(available, action)
        chosen_base = planner.actions[action].base_index
        assert all(planner.actions[item].base_index != chosen_base for item in remainder)
    assert not planner.feasible_actions(available, 0)
    assert all(planner.actions[item].cost <= 2 for item in planner.feasible_actions(available, 2))

    scenarios = np.random.default_rng(11).random((1, 4, WELL_BUDGET))
    result = planner.evaluate_policy_level(2, scenario_uniforms=scenarios)
    assert np.isfinite(result["expected_terminal_mse"])
    assert result["root_repeat_count"] in {1, 2, 4}
    assert result["execution_audit"]["within_budget_no_base_reuse"]
    assert result["execution_audit"]["maximum_wells_spent"] <= WELL_BUDGET


def test_cost_blind_planning_uses_unit_future_cost_but_actual_execution_is_bounded() -> None:
    planner = _planner()
    planner.cost_aware = False
    assert all(planner.planning_cost(action) == 1 for action in range(planner.bank.num_actions))
    scenarios = np.random.default_rng(12).random((1, 4, WELL_BUDGET))
    result = planner.evaluate_policy_level(3, scenario_uniforms=scenarios)
    assert np.isfinite(result["expected_terminal_mse"])
    assert result["execution_audit"]["within_budget_no_base_reuse"]


@pytest.mark.parametrize("cost_aware", (True, False))
@pytest.mark.parametrize("level", (1, 2, 3))
def test_transition_cache_is_byte_exact(cost_aware: bool, level: int) -> None:
    scenarios = np.random.default_rng(20260846 + level).random((1, 6, 3))
    cached = _planner(well_budget=3)
    uncached = _uncached_planner(well_budget=3)
    cached.cost_aware = cost_aware
    uncached.cost_aware = cost_aware

    cached_result = cached.evaluate_policy_level(level, scenario_uniforms=scenarios)
    uncached_result = uncached.evaluate_policy_level(level, scenario_uniforms=scenarios)

    assert cached_result == uncached_result
    assert cached.last_execution_policy_records == uncached.last_execution_policy_records
    assert np.array_equal(cached.last_scenario_losses, uncached.last_scenario_losses)
    assert _audit_summary(cached.transition_audit) == _audit_summary(
        uncached.transition_audit
    )
    assert _proposal_records_hash(cached.proposal_cache.frozen_records) == (
        _proposal_records_hash(uncached.proposal_cache.frozen_records)
    )
    assert cached.transition.cache_info().hits > 0


def _materialized_proposal_key(mode, state, action, outcome, seed):
    payload = {
        "mode": mode,
        "seed": int(seed),
        "state": state.public_key(),
        "action": int(action),
        "outcome": _proposal_outcome_value(outcome),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    ("mode", "action", "outcome", "seed"),
    (
        ("speculative-seed", 0, 0, 23),
        ("atomic_compositional_oracle", np.int64(5), np.int64(2), 9182),
        ('quoted " mode \\ path', 7, 1.25, -2),
        ("unicode-\N{GREEK SMALL LETTER DELTA}", 9, -0.0, 2**31 - 1),
    ),
)
def test_proposal_key_cached_state_fragment_is_byte_exact(
    mode, action, outcome, seed
) -> None:
    planner = _planner(well_budget=2)
    initial = planner.initial_state().inference
    transitioned = planner.transition(planner.initial_state(), 0, 1).inference
    for state in (initial, transitioned):
        expected = _materialized_proposal_key(mode, state, action, outcome, seed)
        assert proposal_key(mode, state, action, outcome, seed) == expected
        assert proposal_key(mode, state, action, outcome, seed) == expected


def test_streaming_proposal_digest_matches_materialized_canonical_json() -> None:
    records = {"b": (3, 4, 5), "a": (), "c": (8,)}
    materialized = {key: list(value) for key, value in records.items()}
    assert _proposal_records_hash(records) == _canonical_hash(materialized)


def test_shared_depth_cache_matches_isolated_planners() -> None:
    scenarios = np.random.default_rng(31).random((1, 2, 2))
    shared = _planner(well_budget=2)
    shared_results = {
        level: shared.evaluate_policy_level(level, scenario_uniforms=scenarios)
        for level in (3, 2, 1)
    }
    for level in (3, 2, 1):
        isolated = _planner(well_budget=2)
        isolated_result = isolated.evaluate_policy_level(
            level, scenario_uniforms=scenarios
        )
        assert shared_results[level] == isolated_result


def test_transcript_replay_reproduces_every_crn_trajectory() -> None:
    planner = _planner(well_budget=2)
    scenarios = np.random.default_rng(32).random((1, 3, 2))
    result = planner.evaluate_policy_level(2, scenario_uniforms=scenarios)
    replay = _replay_execution(
        planner,
        planner.proposal_cache,
        result,
        planner.bank,
        planner.particle_indices,
        planner.compiler,
        planner.particles,
        planner.actions,
        level=2,
        seed=planner.seed,
        cost_aware=True,
        scenario_uniforms=scenarios,
    )
    assert replay["exact"]
    assert replay["unused_policy_records"] == 0
    assert replay["scenario_losses_sha256"] == replay["expected_scenario_losses_sha256"]


def test_difficulty_shard_validation_is_fail_closed() -> None:
    binding = {"implementation_commit": "abc", "settings_sha256": "123"}
    payload = {
        "schema_version": f"{SCHEMA_VERSION}-difficulty-shard-v1",
        "status": "slice_complete",
        "difficulty_index": 0,
        "difficulty": "easy",
        "binding": binding,
        "slice": {"difficulty": "easy", "result": 1},
        "model_calls": 0,
        "network_calls": 0,
        "cost_usd": 0.0,
    }
    assert _validated_shard_slice(
        payload,
        expected_binding=binding,
        expected_index=0,
        difficulty="easy",
    ) == payload["slice"]

    malformed = dict(payload)
    malformed["binding"] = {"implementation_commit": "wrong"}
    with pytest.raises(ValueError, match="binding validation"):
        _validated_shard_slice(
            malformed,
            expected_binding=binding,
            expected_index=0,
            difficulty="easy",
        )
