#!/usr/bin/env python3
"""Audit operating characteristics of the frozen RegretBench SMC claim gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import regretbench_deepseek_smc_dynamic_depth2_experiment as smc


DEFAULT_OUTPUT = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_SMC_PRIMARY_CLAIM_POWER_AUDIT_20260807.json"
)
DEFAULT_REPLICATES = 1_000
DEFAULT_BOOTSTRAP_SAMPLES = 1_000
DEFAULT_SEED = 202608370000


@dataclass(frozen=True)
class Scenario:
    name: str
    refresh_changed_roots: int
    blind_changed_roots: int
    refresh_overall_brier_gain: float
    blind_overall_brier_gain: float
    predicted_overall_gain: float
    predicted_realized_rho: float
    changed_task_gain_sd: float


SCENARIOS = (
    Scenario("threshold_surface", 16, 12, 0.020, 0.015, 0.010, 0.15, 0.08),
    Scenario("design_target", 28, 22, 0.035, 0.025, 0.018, 0.35, 0.08),
    Scenario("strong_signal", 36, 28, 0.050, 0.035, 0.025, 0.50, 0.08),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gain_vector(
    rng: np.random.Generator,
    *,
    changed: np.ndarray,
    overall_gain: float,
    sd: float,
) -> np.ndarray:
    values = np.zeros(64, dtype=float)
    values[changed] = overall_gain * 64 / len(changed) + sd * rng.normal(
        size=len(changed)
    )
    return values


def synthetic_tasks(
    rng: np.random.Generator, scenario: Scenario
) -> list[dict[str, Any]]:
    refresh_changed = rng.choice(
        64, scenario.refresh_changed_roots, replace=False
    )
    blind_changed = rng.choice(64, scenario.blind_changed_roots, replace=False)
    refresh_mask = np.zeros(64, dtype=bool)
    blind_mask = np.zeros(64, dtype=bool)
    refresh_mask[refresh_changed] = True
    blind_mask[blind_changed] = True

    latent = rng.normal(size=scenario.refresh_changed_roots)
    residual = rng.normal(size=scenario.refresh_changed_roots)
    realized_changed = (
        scenario.refresh_overall_brier_gain
        * 64
        / scenario.refresh_changed_roots
        + scenario.changed_task_gain_sd
        * (
            scenario.predicted_realized_rho * latent
            + math.sqrt(1.0 - scenario.predicted_realized_rho**2) * residual
        )
    )
    predicted_changed = np.maximum(
        1e-6,
        scenario.predicted_overall_gain
        * 64
        / scenario.refresh_changed_roots
        + 0.012 * latent,
    )
    predicted_changed *= (
        scenario.predicted_overall_gain * 64 / predicted_changed.sum()
    )

    refresh_gain = np.zeros(64, dtype=float)
    predicted_gain = np.zeros(64, dtype=float)
    refresh_gain[refresh_changed] = realized_changed
    predicted_gain[refresh_changed] = predicted_changed
    blind_gain = _gain_vector(
        rng,
        changed=blind_changed,
        overall_gain=scenario.blind_overall_brier_gain,
        sd=scenario.changed_task_gain_sd,
    )

    tasks = []
    for index in range(64):
        roots = {
            "dynamic_depth2": 0,
            "myopic_refresh_brier": 1 if refresh_mask[index] else 0,
            "myopic_brier": 1,
            "myopic_width": 1,
            "history_blind_depth2": 2 if blind_mask[index] else 0,
            "fixed_depth2": 3,
            "random": 3,
        }
        gains = {
            "myopic_refresh_brier": refresh_gain[index],
            "myopic_brier": 0.05,
            "myopic_width": 0.05,
            "history_blind_depth2": blind_gain[index],
            "fixed_depth2": 0.04,
            "random": 0.06,
        }
        policies: dict[str, dict[str, float]] = {
            "dynamic_depth2": {"brier": 0.20, "log_loss": 0.20}
        }
        for name, gain in gains.items():
            policies[name] = {
                "brier": 0.20 + gain,
                # Hold the safeguard nonworsening so this audit isolates power.
                "log_loss": 0.20 + max(gain, 0.0),
            }
        for policy in policies.values():
            policy["fresh_brier"] = policy["brier"]
            policy["fresh_log_loss"] = policy["log_loss"]
        tasks.append(
            {
                "selected_roots": roots,
                "conditioned_root_risks": [
                    {"brier": 0.10},
                    {"brier": 0.10 + predicted_gain[index]},
                    {"brier": 0.20},
                    {"brier": 0.22},
                ],
                "policies": policies,
            }
        )
    return tasks


def _wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [center - radius, center + radius]


def simulate_scenario(
    scenario: Scenario,
    *,
    replicates: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0 or bootstrap_samples <= 0:
        raise ValueError("replicates and bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    gate_passes = {name: 0 for name in smc.PRIMARY_CLAIM_GATE_NAMES}
    conjunction_passes = 0
    observed = {
        "refresh_brier_difference": [],
        "blind_brier_difference": [],
        "predicted_gain": [],
        "refresh_spearman": [],
        "refresh_spearman_probability_positive": [],
    }
    for _ in range(replicates):
        summary = smc._smc_scientific_summary(
            synthetic_tasks(rng, scenario), samples=bootstrap_samples
        )
        for name, passed in summary["primary_claim_gates"].items():
            gate_passes[name] += int(passed)
        conjunction_passes += int(summary["primary_claim_all_pass"])
        observed["refresh_brier_difference"].append(
            summary["comparisons"]["myopic_refresh_brier"][
                "brier_dynamic_minus_baseline"
            ]["mean"]
        )
        observed["blind_brier_difference"].append(
            summary["comparisons"]["history_blind_depth2"][
                "brier_dynamic_minus_baseline"
            ]["mean"]
        )
        observed["predicted_gain"].append(
            summary["mean_conditioned_predicted_gain_over_myopic_refresh_brier"]
        )
        correlation = summary[
            "predicted_to_realized_dynamic_myopic_refresh_brier"
        ]
        observed["refresh_spearman"].append(correlation["spearman"])
        observed["refresh_spearman_probability_positive"].append(
            correlation["probability_positive"]
        )

    return {
        "scenario": asdict(scenario),
        "replicates": replicates,
        "bootstrap_samples_per_replicate": bootstrap_samples,
        "seed": seed,
        "primary_conjunction": {
            "passes": conjunction_passes,
            "rate": conjunction_passes / replicates,
            "wilson95": _wilson(conjunction_passes, replicates),
            "nominal_two_cohort_rate_if_independent": (
                conjunction_passes / replicates
            )
            ** 2,
        },
        "gate_pass_rates": {
            name: gate_passes[name] / replicates
            for name in smc.PRIMARY_CLAIM_GATE_NAMES
        },
        "observed_metric_means": {
            name: float(np.mean(values)) for name, values in observed.items()
        },
    }


def build_audit(
    *,
    replicates: int = DEFAULT_REPLICATES,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    results = [
        simulate_scenario(
            scenario,
            replicates=replicates,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index * 1_000_000,
        )
        for index, scenario in enumerate(SCENARIOS)
    ]
    return {
        "schema_version": 1,
        "interface_version": "regretbench-smc-primary-claim-power-audit-1",
        "status": "complete",
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "claim_gate_amendment_sha256": smc.CLAIM_GATE_AMENDMENT_SHA256,
        "experiment_sha256": sha256_file(Path(smc.__file__).resolve()),
        "primary_claim_gate_names": list(smc.PRIMARY_CLAIM_GATE_NAMES),
        "method": {
            "task_count": 64,
            "paired_task_level_simulation": True,
            "exact_frozen_scorer_used": True,
            "secondary_controls_forced_nonbinding": True,
            "log_loss_safeguards_held_nonworse": True,
            "changed_root_counts_fixed_per_scenario": True,
            "limitations": (
                "Stylized operating-characteristic audit, not a forecast of live "
                "model behavior; it isolates the frozen primary conjunction."
            ),
            "engineering_calibration_disclosure": (
                "A 100-replicate implementation calibration using these same three "
                "scenarios was inspected before the 1,000-replicate artifact; no "
                "scenario, parameter, scorer, or gate changed afterward."
            ),
        },
        "results": results,
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def render_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# RegretBench SMC Primary-Claim Power Audit",
        "",
        "This zero-call diagnostic sends synthetic paired task rows through the exact frozen 13-gate scorer. It is an operating-characteristic audit, not a forecast of live DeepSeek behavior.",
        "",
        "| Scenario | Refresh changes | Blind changes | Refresh gain | Blind gain | Target rho | One-cohort pass | Nominal two-cohort pass | 95% Wilson interval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in audit["results"]:
        scenario = result["scenario"]
        conjunction = result["primary_conjunction"]
        interval = conjunction["wilson95"]
        lines.append(
            f"| {scenario['name']} | {scenario['refresh_changed_roots']} | "
            f"{scenario['blind_changed_roots']} | "
            f"{scenario['refresh_overall_brier_gain']:.3f} | "
            f"{scenario['blind_overall_brier_gain']:.3f} | "
            f"{scenario['predicted_realized_rho']:.2f} | "
            f"{conjunction['rate']:.3f} | "
            f"{conjunction['nominal_two_cohort_rate_if_independent']:.3f} | "
            f"[{interval[0]:.3f}, {interval[1]:.3f}] |"
        )
    lines.extend(["", "## Gate Pass Rates", ""])
    for result in audit["results"]:
        lines.append(f"### {result['scenario']['name']}")
        lines.append("")
        for name, rate in result["gate_pass_rates"].items():
            lines.append(f"- `{name}`: `{rate:.3f}`")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The threshold-surface scenario is expected to have low conjunction power because several observed statistics sit exactly on one-sided decision boundaries. Under the design-target scenario, paired effect gates are well powered and changed-root ranking fidelity is the dominant failure mode. A strong signal is detected reliably. The audit therefore supports retaining 64 tasks while treating root diversity and predicted-to-realized fidelity as the key live diagnostics.",
            "",
            "Secondary controls and log-loss safeguards are held nonbinding here, so these rates must not be presented as unconditional probabilities of a live pass.",
            "The two-cohort column is only the square of the one-cohort Monte Carlo rate under an independence assumption. It is not a forecast because model-level errors can be shared across cohorts.",
            "",
            str(audit["method"]["engineering_calibration_disclosure"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_audit(output: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown = output.with_suffix(".md")
    markdown.write_text(render_markdown(audit), encoding="utf-8")
    return {
        "status": "written",
        "json_path": str(output),
        "json_sha256": sha256_file(output),
        "markdown_path": str(markdown),
        "markdown_sha256": sha256_file(markdown),
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = build_audit(
        replicates=args.replicates,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(write_audit(args.output.resolve(), audit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
