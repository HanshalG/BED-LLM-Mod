"""Zero-outcome audit of family-revelation lower bounds on the frozen prior."""

import argparse
import hashlib
import json
from pathlib import Path

from environments.scilaws.family_oracle_bound import family_oracle_bound
from environments.scilaws.reference_prior import make_model

DESIGN_SHA = "5e9f2bd902fa9de251cbe033bdd7dd4d5ad8fc79d870e80add7920ed92a18197"
PREFLIGHT_SHA = "4eb4c373ab7974e7771537c8aa8dac5f7af1ffd942b39e4ae70f900bf4b91073"


def read_bound(path, sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("input binding mismatch")
    return json.loads(raw)


def run(output):
    prefix = Path("results/nonmyopic")
    design = read_bound(prefix / "SCILAWS_MEASUREMENT_DESIGN_20260908.json", DESIGN_SHA)
    preflight = read_bound(
        prefix / "SCILAWS_REFERENCE_PREFLIGHT_BATCHED_20260908.json", PREFLIGHT_SHA
    )
    if Path(output).exists():
        raise FileExistsError(output)
    if [t["task_id"] for t in design["tasks"]] != [
        t["task_id"] for t in preflight["tasks"]
    ]:
        raise ValueError("task ordering mismatch")
    rows = []
    for task, previous in zip(design["tasks"], preflight["tasks"], strict=True):
        m = make_model(task, quadrature_order=8)
        h2 = previous["plans"][1]
        if h2["depth"] != 2 or h2["status"] != "completed":
            raise ValueError("requires completed h2 estimate")
        bound = {str(h): family_oracle_bound(m, m.initial_state, h) for h in (1, 2, 3)}
        roots = [
            family_oracle_bound(m, m.initial_state, 2, first_action=a)["value"]
            for a in range(8)
        ]
        rows.append(
            dict(
                task_id=task["task_id"],
                bounds=bound,
                forced_root_h2_bounds=roots,
                h2_quadrature_estimate=h2["risk"],
                roots_with_bound_above_h2_estimate=sum(v > h2["risk"] for v in roots),
            )
        )
    result = dict(
        design_sha256=DESIGN_SHA,
        preflight_sha256=PREFLIGHT_SHA,
        tasks=rows,
        interpretation="continuous_model_bound_diagnostic_not_certified_quadrature_pruning",
        source_measurements=0,
        model_calls=0,
        paid_cost_usd=0,
        pruning_authorized=False,
        policy_endpoint_authorization=False,
    )
    with Path(output).open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(p.parse_args().output)
