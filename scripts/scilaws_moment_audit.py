"""Frozen zero-outcome public/synthetic-state integration check, not efficacy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.scilaws.moment_quadrature import MomentMatchedMixture
from environments.scilaws.reference_prior import make_model

DESIGN_SHA = "5e9f2bd902fa9de251cbe033bdd7dd4d5ad8fc79d870e80add7920ed92a18197"


def corrected(task, order):
    b = make_model(task, quadrature_order=order)
    return MomentMatchedMixture(
        b.action_features,
        b.target_features,
        b.components,
        [0.25] * 4,
        target_weights=b.target_weights,
        quadrature_order=order,
        include_observation_noise=True,
    )


def run(output):
    path = Path("results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError("geometry mismatch")
    if Path(output).exists():
        raise FileExistsError(output)
    rows = []
    for task in json.loads(raw)["tasks"]:
        for scenario in ("prior", "synthetic_action3_value1.7"):
            reference = make_model(task, quadrature_order=128)
            state = reference.initial_state
            if scenario != "prior":
                state = reference.condition(state, 3, 1.7)
            values = [reference.expected_terminal_risk(state, a)[0] for a in range(8)]
            for order in (8, 16):
                model = corrected(task, order)
                estimates = []
                try:
                    estimates = [
                        model.expected_terminal_risk(state, a)[0] for a in range(8)
                    ]
                    errors = np.abs(np.asarray(estimates) - values)
                    selected = int(np.argmin(estimates))
                    row = dict(
                        status="evaluated",
                        values=estimates,
                        max_absolute_error=float(max(errors)),
                        selected_action=selected,
                        reference_regret=float(values[selected] - min(values)),
                    )
                    row["fixture_pass"] = (
                        row["max_absolute_error"] <= 1e-4
                        and row["reference_regret"] <= 1e-4
                    )
                except ValueError as exc:
                    row = dict(
                        status="failed_closed", reason=str(exc), fixture_pass=False
                    )
                rows.append(
                    dict(
                        task_id=task["task_id"],
                        scenario=scenario,
                        order=order,
                        reference_values=values,
                        **row,
                    )
                )
        print(task["task_id"], flush=True)
    result = dict(
        design_sha256=DESIGN_SHA,
        rows=rows,
        thresholds=dict(max_absolute_error=1e-4, reference_regret=1e-4),
        all_fixtures_pass=all(r["fixture_pass"] for r in rows),
        reference="uncorrected_order128_not_rigorous_true_integral",
        source_measurements=0,
        model_calls=0,
        paid_cost_usd=0,
        scientific_execution_authorized=False,
        pruning_authorized=False,
    )
    with Path(output).open("x") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(p.parse_args().output)
