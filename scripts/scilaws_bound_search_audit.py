"""Frozen full-menu public-prior bounded search; no source outcomes or calls."""

import argparse
import hashlib
import json
from pathlib import Path

from environments.chembench_mopen.horizon import (
    HorizonPlanner, SearchLimits, SearchLimitExceeded,
)
from scripts.scilaws_moment_audit import corrected
from scripts.scilaws_reference_preflight import DESIGN_SHA


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path("results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError("geometry binding mismatch")
    rows = []
    for task in json.loads(raw)["tasks"]:
        m = corrected(task, 8, "horizon_control_variate")
        planner = HorizonPlanner(m, limits=SearchLimits(max_nodes=100000, max_seconds=5))
        row = {"task_id": task["task_id"], "plans": []}
        for depth in (1, 2, 3):
            try:
                p = planner.plan(m.initial_state, depth, allow_repeats=True,
                                 use_action_bounds=True)
                row["plans"].append(dict(
                    depth=depth, status="completed", action=p.root.action,
                    risk=p.root.expected_risk, nodes=p.expanded_nodes,
                    seconds=p.elapsed_seconds, pruned_actions=p.pruned_actions,
                    root_action_values=p.root_action_values,
                    root_pruned_lower_bounds=p.root_pruned_lower_bounds,
                ))
            except SearchLimitExceeded as exc:
                row["plans"].append(dict(depth=depth, status="resource_limit", reason=str(exc)))
                break
        rows.append(row)
        print(row["task_id"], row["plans"][-1]["status"], flush=True)
    result = dict(
        schema_version=1, tasks=rows, design_sha256=DESIGN_SHA,
        quadrature_order=8, per_plan_seconds_cap=5, per_plan_nodes_cap=100000,
        complete_depth_coverage=all(len(r["plans"]) == 3 and all(
            p["status"] == "completed" for p in r["plans"]) for r in rows),
        interpretation="public_prior_runtime_only_not_calibration_or_depth_efficacy",
        source_measurements=0, model_calls=0, paid_cost_usd=0,
        paid_authorization=False, policy_endpoint_authorization=False,
    )
    with output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(p.parse_args().output)
