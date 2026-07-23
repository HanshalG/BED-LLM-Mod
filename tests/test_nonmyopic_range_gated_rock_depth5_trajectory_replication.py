from copy import deepcopy

from scripts.audit_nonmyopic_range_gated_rock_depth5_trajectory_replication import (
    audit_pool,
)
from scripts.nonmyopic_range_gated_rock_depth5_trajectory_replication import (
    AUDIT_SEEDS,
    METRICS,
    REPLICATION_SEEDS,
    aggregate_replications,
)


def _replication(seed: int, audit_seed: int, offset: float):
    comparisons = {}
    audit_comparisons = {}
    for index, metric in enumerate(METRICS):
        values = [offset + 0.01 * (index + 1) + 0.001 * i for i in range(50)]
        mean = sum(values) / len(values)
        comparisons[metric] = {
            "mean": mean,
            "paired_values": values,
        }
        audit_comparisons[metric] = {"mean": mean}
    confirmation = {
        "config": {
            "seed": seed,
            "num_trials": 50,
            "num_rounds": 8,
            "max_unique_llm_cells": 64,
        },
        "truth_indices": list(range(50)),
        "gate": {"passed": True},
        "mechanics": {"rollout_scoring_made_no_llm_calls": True},
        "comparisons": comparisons,
        "llm_recovery_fraction_of_exact_h5_gain": 0.9,
        "llm_registered_route_rate": 0.9,
        "llm_onsite_by_round_five_rate": 0.9,
        "logical_requests": [{} for _ in range(400)],
        "candidate_requests": [{} for _ in range(18)],
        "usage": {
            "requests": 36,
            "completion_tokens": 100,
            "forced_exits": 18,
            "run_cost_usd": 0.03,
        },
    }
    audit = {
        "audit_bootstrap_seed": audit_seed,
        "gate": {"passed": True},
        "mechanics": {"no_llm_calls": True},
        "comparisons": audit_comparisons,
    }
    return confirmation, audit


def _replications():
    return [
        _replication(seed, audit_seed, 0.1 + 0.01 * index)
        for index, (seed, audit_seed) in enumerate(
            zip(REPLICATION_SEEDS, AUDIT_SEEDS, strict=True)
        )
    ]


def test_replication_pool_and_independent_audit_pass() -> None:
    replications = _replications()

    result = aggregate_replications(replications)
    replay = audit_pool(result, replications)

    assert result["gate"]["passed"]
    assert result["num_paired_trials"] == 150
    assert result["total_logical_decisions"] == 1200
    assert all(
        row["stratified_ci95"][0] > 0.0
        for row in result["comparisons"].values()
    )
    assert replay["gate"]["passed"]
    assert all(replay["mechanics"].values())


def test_replication_pool_fails_when_one_seed_is_directionally_negative() -> None:
    replications = _replications()
    confirmation, audit = replications[1]
    broken_confirmation = deepcopy(confirmation)
    broken_audit = deepcopy(audit)
    metric = METRICS[0]
    broken_confirmation["comparisons"][metric]["paired_values"] = [
        -0.01 for _ in range(50)
    ]
    broken_confirmation["comparisons"][metric]["mean"] = -0.01
    broken_audit["comparisons"][metric]["mean"] = -0.01
    replications[1] = (broken_confirmation, broken_audit)

    result = aggregate_replications(replications)

    assert not result["gate"]["passed"]
    assert not result["endpoint_gate"][
        f"{metric}_positive_in_every_replication"
    ]


def test_replication_pool_rejects_unregistered_seed() -> None:
    replications = _replications()
    replications[2][0]["config"]["seed"] = 999

    result = aggregate_replications(replications)

    assert not result["gate"]["passed"]
    assert not result["mechanics"]["expected_replication_seeds"]
