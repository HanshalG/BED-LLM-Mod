"""Evaluate prior-predictive myopic terminal risk without hidden worlds."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np

from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.native_belief import NativeEnvelopeGaussianModel
from environments.chembench_mopen.pilot_data import build_public_pilot, read_protocol
from environments.chembench_mopen.policy_value import evaluate_myopic_policy
from scripts.chembench_mopen_nonmyopic_opportunity import load_source
from scripts.chembench_pilot_preflight import preflight


PROTOCOL = Path(
    "results/nonmyopic/CHEMBENCH_EXPECTED_POLICY_VALUE_PROTOCOL_20260908.json"
)
PILOT = Path("results/nonmyopic/chembench_horizon_pilot")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def execute(output, source_root):
    protocol = json.loads(PROTOCOL.read_text())
    if (
        digest(PILOT / "run-20260908-v3/RESULT.json")
        != protocol["primary_result_sha256"]
        or digest(PILOT / "replay-20260908-v2/RESULT.json")
        != protocol["primary_replay_sha256"]
    ):
        raise ValueError("primary result or replay binding mismatch")
    if read_protocol()[1] != protocol["physics_sha256"]:
        raise ValueError("physics binding mismatch")
    started = monotonic()
    check = preflight(source_root, "native")
    if check["status"] != "public_preflight_passed":
        raise ValueError("numerical predecessor not ready")
    public = build_public_pilot(
        load_source(source_root), model_type=NativeEnvelopeGaussianModel
    ).model
    records = []
    for order in protocol["orders"]:
        model = NativeEnvelopeGaussianModel(
            public.means,
            public.sigmas,
            public.targets,
            np.exp(public.initial_state),
            target_weights=public.target_weights,
            branch_count=order,
        )
        remaining = protocol["max_panel_seconds"] - (monotonic() - started)
        myopic = evaluate_myopic_policy(
            model,
            model.initial_state,
            protocol["measurement_budget"],
            max_states=protocol["max_states"],
            max_seconds=min(remaining, protocol["max_seconds_per_evaluation"]),
            max_workspace_bytes=protocol["max_workspace_bytes"],
        )
        if order == 64:
            optimum_path = PILOT / "run-20260908-v3/public_root_h3.json"
            optimum = json.loads(optimum_path.read_text())
            selected_h1 = json.loads(
                (PILOT / "run-20260908-v3/public_root_h1.json").read_text()
            )
            if myopic.root_action != selected_h1["action"] or not np.allclose(
                myopic.root_one_step_values,
                selected_h1["root_values"],
                atol=1e-10,
                rtol=1e-10,
            ):
                raise ValueError("deployed one-step rule did not replay")
            if optimum["effective_horizon"] != protocol["measurement_budget"]:
                raise ValueError("banked comparator has wrong budget")
            provenance = {"reused": True, "sha256": digest(optimum_path)}
        else:
            remaining = protocol["max_panel_seconds"] - (monotonic() - started)
            optimum = asdict(
                plan_batched(
                    model,
                    model.initial_state,
                    protocol["measurement_budget"],
                    max_states=protocol["max_states"],
                    max_seconds=min(remaining, protocol["max_seconds_per_evaluation"]),
                    max_workspace_bytes=protocol["max_workspace_bytes"],
                )
            )
            provenance = {"reused": False}
        record = {
            "order": order,
            "myopic": asdict(myopic),
            "optimal": optimum,
            "optimal_provenance": provenance,
            "expected_gain": myopic.value - optimum["value"],
            "fractional_gain": 1 - optimum["value"] / myopic.value,
        }
        records.append(record)
        (output / f"order{order}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
    deltas = {
        "myopic": abs(records[0]["myopic"]["value"] - records[1]["myopic"]["value"]),
        "optimal": abs(records[0]["optimal"]["value"] - records[1]["optimal"]["value"]),
    }
    return {
        "status": "public_expected_value_complete",
        "records": records,
        "refinement_deltas": deltas,
        "refinement_stable": all(
            d <= protocol["absolute_refinement_tolerance"] for d in deltas.values()
        ),
        "elapsed_seconds": monotonic() - started,
        "interpretation": "Finite public-prior expected risk under quadrature, not a hidden-source replication or a correction to the empirical panel. Three real measurements for both policies. No LLM efficacy or strict h3-over-h2 claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = [
        PROTOCOL,
        Path(__file__),
        Path("environments/chembench_mopen/policy_value.py"),
    ]
    bindings = {str(path): digest(path) for path in paths}
    try:
        report = execute(args.output_dir, args.source_root)
    except Exception as exc:
        report = {
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    report.update(
        {
            "source_hashes": bindings,
            "hidden_worlds_constructed": False,
            "model_calls": 0,
            "cost_usd": 0,
            "paid_calls_authorized": False,
        }
    )
    temporary = args.output_dir / "RESULT.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output_dir / "RESULT.json")
    print(report["status"])


if __name__ == "__main__":
    main()
