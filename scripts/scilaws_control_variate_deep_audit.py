"""Bounded analytic-depth and public h2 consistency checks, no source outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

from environments.chembench_mopen.horizon import (
    HorizonPlanner,
    SearchLimitExceeded,
    SearchLimits,
)
from environments.scilaws.control_variate import ControlVariateMixture
from environments.scilaws.family_oracle_bound import family_oracle_bound
from environments.scilaws.regression_belief import RegressionBelief
from scripts.scilaws_moment_audit import corrected, DESIGN_SHA


def plan_row(model, state, depth):
    planner = HorizonPlanner(
        model, limits=SearchLimits(max_nodes=100000, max_seconds=5)
    )
    try:
        plan = planner.plan(state, depth, allow_repeats=True)
        return dict(
            status="completed",
            depth=depth,
            action=plan.root.action,
            risk=plan.root.expected_risk,
            root_action_values=list(plan.root_action_values),
            nodes=plan.expanded_nodes,
            seconds=plan.elapsed_seconds,
        )
    except SearchLimitExceeded as exc:
        return dict(status="resource_limit", depth=depth, reason=str(exc))


def run(output):
    path = Path("results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError("geometry mismatch")
    if Path(output).exists():
        raise FileExistsError(output)
    fixture = []
    for order in (4, 8, 16):
        b = RegressionBelief([0.0], [[1.0]], 3.0, 0.2)
        m = ControlVariateMixture(
            [[[1.0], [2.0]]],
            [[[1.0], [3.0]]],
            [b],
            [1.0],
            target_weights=[0.25, 0.75],
            quadrature_order=order,
            include_observation_noise=True,
        )
        for depth in (1, 2, 3):
            row = plan_row(m, m.initial_state, depth)
            exact = 0.1 * (1 + 7 / (1 + 4 * depth))
            row.update(order=order, exact_risk=exact)
            row["absolute_error"] = (
                abs(row["risk"] - exact) if row["status"] == "completed" else None
            )
            row["fixture_pass"] = (
                row["status"] == "completed"
                and row["absolute_error"] <= 1e-4
                and row["action"] == 1
            )
            fixture.append(row)
        print("analytic order", order, flush=True)
    rows = []
    for task in json.loads(raw)["tasks"]:
        m = corrected(task, 8, "control_variate")
        row = plan_row(m, m.initial_state, 2)
        row["task_id"] = task["task_id"]
        lower = family_oracle_bound(m, m.initial_state, 2)["value"]
        row["continuous_family_lower_bound"] = lower
        row["below_bound_by"] = (
            max(0.0, lower - row["risk"]) if row["status"] == "completed" else None
        )
        row["bound_consistency_check"] = (
            row["status"] == "completed" and row["below_bound_by"] <= 1e-10
        )
        rows.append(row)
        print(task["task_id"], row["status"], flush=True)
    result = dict(
        design_sha256=DESIGN_SHA,
        analytic_fixture=fixture,
        public_h2=rows,
        thresholds=dict(
            analytic_absolute_error=1e-4, below_family_bound_tolerance=1e-10
        ),
        cap_seconds=5,
        cap_nodes=100000,
        source_measurements=0,
        model_calls=0,
        paid_cost_usd=0,
        interpretation="analytic_fixture_and_necessary_bound_check_not_deep_mixture_accuracy",
        pruning_authorized=False,
        scientific_execution_authorized=False,
    )
    with Path(output).open("x") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(p.parse_args().output)
