"""Aggregate preregistered fresh-seed hierarchical h5 trajectory replications."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPLICATION_SEEDS = (24_250, 24_252, 24_254)
AUDIT_SEEDS = (24_251, 24_253, 24_255)
POOL_BOOTSTRAP_SEED = 24_256
POOL_BOOTSTRAP_REPLICATES = 10_000
EXPECTED_TRIALS = 50
EXPECTED_ROUNDS = 8
METRICS = tuple(
    f"{metric}_auc_gain_vs_{baseline}"
    for baseline in ("shared_h4", "random_h5", "exact_d4")
    for metric in ("entropy", "truth_log")
)


def _stratified_bootstrap(
    strata: Sequence[np.ndarray],
    *,
    seed: int,
    replicates: int = POOL_BOOTSTRAP_REPLICATES,
) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    total = sum(len(values) for values in strata)
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        sums = np.zeros(size, dtype=float)
        for values in strata:
            indices = rng.integers(0, len(values), size=(size, len(values)))
            sums += values[indices].sum(axis=1)
        samples[start : start + size] = sums / total
    return [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def _load_replication(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    confirmation = json.loads(
        (directory / "CONFIRMATION.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (directory / "audit" / "AUDIT.json").read_text(encoding="utf-8")
    )
    return confirmation, audit


def aggregate_replications(
    replications: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    bootstrap_seed: int = POOL_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if len(replications) != len(REPLICATION_SEEDS):
        raise ValueError("the frozen package requires exactly three replications")

    per_replication: list[dict[str, Any]] = []
    strata_by_metric: dict[str, list[np.ndarray]] = {
        metric: [] for metric in METRICS
    }
    mechanics: dict[str, bool] = {
        "expected_replication_seeds": True,
        "expected_independent_audit_seeds": True,
        "all_producer_gates_pass": True,
        "all_independent_audits_pass": True,
        "all_replications_have_50_trials_and_8_rounds": True,
        "all_audit_means_match_producer_means": True,
        "all_paired_values_are_finite": True,
        "exactly_1200_logical_decisions": True,
        "all_per_run_physical_prompt_caps_hold": True,
        "all_scoring_no_llm_invariants_hold": True,
        "all_usage_is_accounted": True,
    }
    total_logical = 0
    total_physical = 0
    total_cost = 0.0

    for index, ((confirmation, audit), seed, audit_seed) in enumerate(
        zip(replications, REPLICATION_SEEDS, AUDIT_SEEDS, strict=True)
    ):
        config = confirmation.get("config", {})
        mechanics["expected_replication_seeds"] &= config.get("seed") == seed
        mechanics["expected_independent_audit_seeds"] &= (
            audit.get("audit_bootstrap_seed") == audit_seed
        )
        mechanics["all_producer_gates_pass"] &= bool(
            confirmation.get("gate", {}).get("passed")
        )
        mechanics["all_independent_audits_pass"] &= bool(
            audit.get("gate", {}).get("passed")
        )
        mechanics["all_replications_have_50_trials_and_8_rounds"] &= (
            config.get("num_trials") == EXPECTED_TRIALS
            and config.get("num_rounds") == EXPECTED_ROUNDS
            and len(confirmation.get("truth_indices", [])) == EXPECTED_TRIALS
        )
        logical = len(confirmation.get("logical_requests", []))
        physical = len(confirmation.get("candidate_requests", []))
        total_logical += logical
        total_physical += physical
        mechanics["all_per_run_physical_prompt_caps_hold"] &= (
            physical <= int(config.get("max_unique_llm_cells", -1))
        )
        mechanics["all_scoring_no_llm_invariants_hold"] &= bool(
            confirmation.get("mechanics", {}).get(
                "rollout_scoring_made_no_llm_calls"
            )
        ) and bool(audit.get("mechanics", {}).get("no_llm_calls"))
        usage = confirmation.get("usage", {})
        mechanics["all_usage_is_accounted"] &= all(
            key in usage
            for key in (
                "requests",
                "completion_tokens",
                "forced_exits",
                "run_cost_usd",
            )
        )
        total_cost += float(usage.get("run_cost_usd", 0.0))

        means: dict[str, float] = {}
        for metric in METRICS:
            row = confirmation["comparisons"][metric]
            values = np.asarray(row["paired_values"], dtype=float)
            if len(values) != EXPECTED_TRIALS or not np.isfinite(values).all():
                mechanics["all_paired_values_are_finite"] = False
            strata_by_metric[metric].append(values)
            means[metric] = float(values.mean())
            mechanics["all_audit_means_match_producer_means"] &= math.isclose(
                means[metric],
                float(audit["comparisons"][metric]["mean"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        per_replication.append(
            {
                "index": index + 1,
                "seed": seed,
                "audit_seed": audit_seed,
                "producer_gate_passed": bool(confirmation["gate"]["passed"]),
                "audit_gate_passed": bool(audit["gate"]["passed"]),
                "metric_means": means,
                "recovery": float(
                    confirmation["llm_recovery_fraction_of_exact_h5_gain"]
                ),
                "route_rate": float(confirmation["llm_registered_route_rate"]),
                "onsite_rate": float(
                    confirmation["llm_onsite_by_round_five_rate"]
                ),
                "logical_decisions": logical,
                "physical_prompts": physical,
                "cost_usd": float(usage.get("run_cost_usd", 0.0)),
            }
        )

    mechanics["exactly_1200_logical_decisions"] &= total_logical == 1_200
    comparisons: dict[str, Any] = {}
    endpoint_gate: dict[str, bool] = {}
    for metric_index, metric in enumerate(METRICS):
        strata = strata_by_metric[metric]
        pooled = np.concatenate(strata)
        ci95 = _stratified_bootstrap(
            strata,
            seed=bootstrap_seed + metric_index,
        )
        comparisons[metric] = {
            "mean": float(pooled.mean()),
            "stratified_ci95": ci95,
            "wins_ties_losses": [
                int(np.sum(pooled > 1e-12)),
                int(np.sum(np.abs(pooled) <= 1e-12)),
                int(np.sum(pooled < -1e-12)),
            ],
            "paired_values_by_seed": [values.tolist() for values in strata],
        }
        endpoint_gate[f"{metric}_pooled_lower_bound_positive"] = ci95[0] > 0.0
        endpoint_gate[f"{metric}_positive_in_every_replication"] = all(
            replication["metric_means"][metric] > 0.0
            for replication in per_replication
        )

    recovery = float(np.mean([row["recovery"] for row in per_replication]))
    route_rate = float(np.mean([row["route_rate"] for row in per_replication]))
    onsite_rate = float(np.mean([row["onsite_rate"] for row in per_replication]))
    endpoint_gate.update(
        {
            "pooled_recovery_at_least_0_60": recovery >= 0.60,
            "pooled_route_rate_at_least_0_75": route_rate >= 0.75,
            "pooled_onsite_rate_at_least_0_75": onsite_rate >= 0.75,
        }
    )
    passed = all(mechanics.values()) and all(endpoint_gate.values())
    return {
        "schema_version": 1,
        "stage": "focused_range_gated_rock_h5_trajectory_replication_pool",
        "replication_seeds": list(REPLICATION_SEEDS),
        "audit_seeds": list(AUDIT_SEEDS),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": POOL_BOOTSTRAP_REPLICATES,
        "num_paired_trials": len(REPLICATION_SEEDS) * EXPECTED_TRIALS,
        "per_replication": per_replication,
        "comparisons": comparisons,
        "pooled_recovery": recovery,
        "pooled_route_rate": route_rate,
        "pooled_onsite_rate": onsite_rate,
        "total_logical_decisions": total_logical,
        "total_physical_prompts": total_physical,
        "total_cost_usd": total_cost,
        "mechanics": mechanics,
        "endpoint_gate": endpoint_gate,
        "gate": {"passed": passed},
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Focused Range-Gated Rock H5 Trajectory Replication Pool",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint | Pooled mean | Stratified 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        row = result["comparisons"][metric]
        lines.append(
            f"| {metric} | {row['mean']:+.6f} | "
            f"[{row['stratified_ci95'][0]:+.6f}, "
            f"{row['stratified_ci95'][1]:+.6f}] | "
            f"{'/'.join(map(str, row['wins_ties_losses']))} |"
        )
    lines.extend(
        [
            "",
            f"Pooled recovery: {result['pooled_recovery']:.1%}.",
            f"Pooled route/on-site rates: {result['pooled_route_rate']:.1%}/"
            f"{result['pooled_onsite_rate']:.1%}.",
            f"Logical/physical prompts: {result['total_logical_decisions']}/"
            f"{result['total_physical_prompts']}.",
            f"OpenRouter cost: ${result['total_cost_usd']:.6f}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replication_dirs",
        type=Path,
        nargs=3,
        help="three directories containing CONFIRMATION.json and audit/AUDIT.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    replications = [_load_replication(path) for path in args.replication_dirs]
    result = aggregate_replications(replications)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "POOL.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "POOL.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({"gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
