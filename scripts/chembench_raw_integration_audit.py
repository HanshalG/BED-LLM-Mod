"""Zero-call synthetic quadrature audit; never opens chemistry or LLM outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from environments.chembench_mopen.raw_belief import GaussianParticleModel


def build_report() -> dict:
    # Fixed numerical stress panel, not chemistry worlds or efficacy evidence.
    rows = []
    for separation in (0.2, 1.0, 3.0, 8.0):
        for unequal_noise in (False, True):
            means = np.array([[-separation, -0.5], [separation, 0.5]])
            sigmas = (
                np.array([[0.3, 1.0], [2.0, 1.0]]) if unequal_noise else np.ones((2, 2))
            )
            model = GaussianParticleModel(
                means, sigmas, [[0], [1]], [0.3, 0.7], quadrature_order=9
            )
            exact = []
            errors = []
            for action in range(2):

                def integrand(y):
                    densities = np.exp(
                        -0.5 * ((y - means[:, action]) / sigmas[:, action]) ** 2
                    ) / (math.sqrt(2 * math.pi) * sigmas[:, action])
                    masses = np.array([0.3, 0.7]) * densities
                    return (
                        float(masses[0] * masses[1] / masses.sum())
                        if masses.sum()
                        else 0.0
                    )

                lower = float(np.min(means[:, action] - 12 * sigmas[:, action]))
                upper = float(np.max(means[:, action] + 12 * sigmas[:, action]))
                value, error = quad(
                    integrand,
                    lower,
                    upper,
                    points=sorted(set(means[:, action])),
                    epsabs=1e-11,
                    limit=200,
                )
                exact.append(value)
                errors.append(error)
            approximate = [
                math.fsum(
                    b.probability * model.risk(b.state)
                    for b in model.branches(model.initial_state, a)
                )
                for a in range(2)
            ]
            selected = int(np.argmin(approximate))
            rows.append(
                {
                    "separation": separation,
                    "unequal_noise": unequal_noise,
                    "reference_risks": exact,
                    "reference_error_estimates": errors,
                    "quadrature_risks": approximate,
                    "max_absolute_error": max(
                        abs(a - b) for a, b in zip(exact, approximate)
                    ),
                    "selected_action": selected,
                    "reference_regret": exact[selected] - min(exact),
                }
            )
    passed = all(
        max(row["reference_error_estimates"]) <= 1e-8
        and row["max_absolute_error"] <= 1e-3
        and row["reference_regret"] <= 1e-3
        for row in rows
    )
    root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "status": "one_step_reference_passed"
        if passed
        else "one_step_reference_failed",
        "settings": {
            "order": 9,
            "absolute_error_cap": 1e-3,
            "regret_cap": 1e-3,
            "reference_error_cap": 1e-8,
        },
        "cases": rows,
        "source_hashes": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in (
                "scripts/chembench_raw_integration_audit.py",
                "environments/chembench_mopen/raw_belief.py",
            )
        },
        "model_calls": 0,
        "cost_usd": 0,
        "chemistry_outcomes_opened": False,
        "depth_two_three_qualified": False,
        "authorizes_paid_calls": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        report = build_report()
    except Exception as exc:
        failure = {
            "status": "audit_execution_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "authorizes_paid_calls": False,
        }
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, allow_nan=False) + "\n"
        )
        raise
    temporary = args.output_dir / "RESULT.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output_dir / "RESULT.json")
    print(report["status"])


if __name__ == "__main__":
    main()
