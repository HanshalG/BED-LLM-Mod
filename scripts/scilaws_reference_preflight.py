"""Bounded public-prior planning only; never opens source measurements."""

import argparse
import hashlib
import json
from pathlib import Path

from environments.chembench_mopen.horizon import (
    HorizonPlanner,
    SearchLimitExceeded,
    SearchLimits,
)
from environments.scilaws.reference_prior import make_model

DESIGN_SHA = "5e9f2bd902fa9de251cbe033bdd7dd4d5ad8fc79d870e80add7920ed92a18197"


def run(path, output):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError("geometry binding mismatch")
    if Path(output).exists():
        raise FileExistsError(output)
    results = []
    for task in json.loads(raw)["tasks"]:
        model = make_model(task, quadrature_order=8)
        planner = HorizonPlanner(
            model, limits=SearchLimits(max_nodes=100000, max_seconds=5)
        )
        row = dict(
            task_id=task["task_id"],
            prior_risk=model.risk(model.initial_state),
            plans=[],
        )
        for depth in (1, 2, 3):
            try:
                plan = planner.plan(model.initial_state, depth, allow_repeats=True)
                row["plans"].append(
                    dict(
                        depth=depth,
                        status="completed",
                        action=plan.root.action,
                        risk=plan.root.expected_risk,
                        seconds=plan.elapsed_seconds,
                        nodes=plan.expanded_nodes,
                    )
                )
            except SearchLimitExceeded as exc:
                row["plans"].append(
                    dict(depth=depth, status="resource_limit", reason=str(exc))
                )
                break
        results.append(row)
        print(task["task_id"], row["plans"][-1]["status"], flush=True)
    result = dict(
        schema_version=1,
        design_sha256=DESIGN_SHA,
        tasks=results,
        quadrature_order=8,
        per_plan_seconds_cap=5,
        per_plan_nodes_cap=100000,
        complete_depth_coverage=all(
            len(r["plans"]) == 3 and all(p["status"] == "completed" for p in r["plans"])
            for r in results
        ),
        interpretation="public_prior_runtime_only_not_calibration_or_depth_efficacy",
        source_measurements=0,
        model_calls=0,
        paid_cost_usd=0,
        paid_authorization=False,
        policy_endpoint_authorization=False,
    )
    with Path(output).open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    run(args.design, args.output)
