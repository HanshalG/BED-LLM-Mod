"""Public-source-only feasibility preflight; hidden worlds remain unopened."""

import argparse
import hashlib
import json
from pathlib import Path

from environments.chembench_mopen.pilot_data import build_public_pilot, read_protocol
from environments.chembench_mopen.crossing_belief import CrossingGaussianModel
from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel
from scripts.chembench_mopen_nonmyopic_opportunity import load_source, verify_source


def preflight(source_root, engine="crossing"):
    if engine not in ("crossing", "envelope", "native"):
        raise ValueError("unknown numerical engine")
    config, digest = read_protocol()
    binding = verify_source(source_root)
    if binding["commit"] != config["source_commit"]:
        raise ValueError("source mismatch")
    root = Path(__file__).resolve().parents[1]
    version = {"envelope": "v5", "native": "v2", "crossing": "v1"}[engine]
    gate = (
        root
        / f"results/nonmyopic/chembench_{engine}_refinement/20260908-{version}/RESULT.json"
    )
    result = json.loads(gate.read_text())
    if engine == "native" and result.get("engine") != "native":
        raise ValueError("native engine qualification missing")
    if result["status"] != "synthetic_refinement_passed" or not all(
        result["checks"].values()
    ):
        raise ValueError("numerical predecessor failed")
    for relative, expected in result["source_hashes"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("numerical binding changed")
    model_type = (
        EnvelopeGaussianModel if engine == "envelope" else CrossingGaussianModel
    )
    if engine == "native":
        from environments.chembench_mopen.native_belief import (
            NativeEnvelopeGaussianModel,
        )

        model_type = NativeEnvelopeGaussianModel
    public = build_public_pilot(load_source(source_root), model_type=model_type)
    action_checks = []
    for action in range(len(public.designs)):
        try:
            branches = public.model.branches(public.model.initial_state, action)
            action_checks.append(
                {"action": action, "status": "ready", "branches": len(branches)}
            )
        except Exception as exc:
            action_checks.append(
                {
                    "action": action,
                    "status": "not_ready",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return {
        "status": "public_preflight_passed"
        if all(a["status"] == "ready" for a in action_checks)
        else "public_preflight_failed",
        "action_checks": action_checks,
        "prior_particles": public.model.num_particles,
        "engine": engine,
        "numerical_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
        "protocol_sha256": digest,
        "source_binding": binding,
        "hidden_worlds_opened": False,
        "model_calls": 0,
        "cost_usd": 0,
        "paid_calls_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--engine", choices=["crossing", "envelope", "native"], default="crossing"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = preflight(args.source_root, args.engine)
    except Exception as exc:
        result = {
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "hidden_worlds_opened": False,
            "paid_calls_authorized": False,
        }
    temp = args.output_dir / "RESULT.tmp"
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output_dir / "RESULT.json")
    print(result["status"])


if __name__ == "__main__":
    main()
