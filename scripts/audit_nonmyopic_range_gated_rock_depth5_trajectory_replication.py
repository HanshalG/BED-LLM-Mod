"""Independent pooled audit for fresh-seed hierarchical h5 replications."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.nonmyopic_range_gated_rock_depth5_trajectory_replication import (
    METRICS,
    _load_replication,
    aggregate_replications,
)


AUDIT_POOL_BOOTSTRAP_SEED = 24_257


def audit_pool(
    payload: dict[str, Any],
    replications: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    bootstrap_seed: int = AUDIT_POOL_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    replay = aggregate_replications(replications, bootstrap_seed=bootstrap_seed)
    checks = {
        "source_stage_matches": payload.get("stage")
        == "focused_range_gated_rock_h5_trajectory_replication_pool",
        "registered_seeds_match": (
            payload.get("replication_seeds") == replay["replication_seeds"]
            and payload.get("audit_seeds") == replay["audit_seeds"]
        ),
        "all_paired_values_match": True,
        "all_pooled_means_match": True,
        "all_wins_ties_losses_match": True,
        "all_scalar_endpoints_match": True,
        "independent_lower_bounds_positive": True,
        "all_source_audits_pass": all(
            audit.get("gate", {}).get("passed")
            for _, audit in replications
        ),
    }
    comparisons: dict[str, Any] = {}
    for metric in METRICS:
        source = payload["comparisons"][metric]
        audited = replay["comparisons"][metric]
        checks["all_paired_values_match"] &= (
            source["paired_values_by_seed"] == audited["paired_values_by_seed"]
        )
        checks["all_pooled_means_match"] &= math.isclose(
            float(source["mean"]),
            float(audited["mean"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        checks["all_wins_ties_losses_match"] &= (
            source["wins_ties_losses"] == audited["wins_ties_losses"]
        )
        checks["independent_lower_bounds_positive"] &= (
            audited["stratified_ci95"][0] > 0.0
        )
        comparisons[metric] = {
            "mean": audited["mean"],
            "independent_stratified_ci95": audited["stratified_ci95"],
            "wins_ties_losses": audited["wins_ties_losses"],
        }
    for key in ("pooled_recovery", "pooled_route_rate", "pooled_onsite_rate"):
        checks["all_scalar_endpoints_match"] &= math.isclose(
            float(payload[key]),
            float(replay[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    gate = {
        "passed": all(checks.values())
        and all(replay["endpoint_gate"].values())
    }
    return {
        "schema_version": 1,
        "stage": "focused_range_gated_rock_h5_trajectory_replication_pool_audit",
        "source_stage": payload.get("stage"),
        "audit_bootstrap_seed": bootstrap_seed,
        "mechanics": checks,
        "comparisons": comparisons,
        "endpoint_gate": replay["endpoint_gate"],
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_json", type=Path)
    parser.add_argument("replication_dirs", type=Path, nargs=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-bootstrap-seed",
        type=int,
        default=AUDIT_POOL_BOOTSTRAP_SEED,
    )
    args = parser.parse_args()
    payload = json.loads(args.pool_json.read_text(encoding="utf-8"))
    replications = [_load_replication(path) for path in args.replication_dirs]
    result = audit_pool(
        payload,
        replications,
        bootstrap_seed=args.audit_bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
