#!/usr/bin/env python3
"""Decompose RegretBench delayed value from immediate regeneration cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import regretbench_branch_draw_fidelity as fidelity


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_HORIZON_VALUE_DECOMPOSITION_PROTOCOL_20260807.md"
)
PROTOCOL_SHA256 = "2ff70036188dc45ccb82bff9a1570e796e6f1bce5d1c12bc9f081f70bbbb4028"
BINDING = REPO_ROOT / (
    "results/nonmyopic/regretbench_horizon_value_decomposition/"
    "EXECUTION_BINDING.json"
)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 202608430000
IDENTITY_TOLERANCE = 1e-10
POLICIES = fidelity.ALLOWED_INTERFACES
QUANTITIES = (
    "immediate_penalty",
    "dynamic_horizon_value",
    "refresh_horizon_value",
    "differential_horizon_value",
    "predicted_terminal_advantage",
    "realized_terminal_advantage",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"nonnumeric horizon value: {label}")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"nonfinite horizon value: {label}")
    return output


def validate_binding() -> dict[str, Any]:
    if _sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("horizon-value protocol changed")
    binding = _load(BINDING)
    expected = {
        "protocol": PROTOCOL,
        "script": Path(__file__).resolve(),
        "source_verifier_binding": fidelity.BINDING,
    }
    if binding.get("status") != "frozen_before_any_regretbench_policy_response":
        raise ValueError("horizon-value binding status changed")
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)):
            raise ValueError(f"horizon-value {name} path changed")
        if row.get("sha256") != _sha256(path):
            raise ValueError(f"horizon-value {name} hash changed")
    return binding


def _summary(values: np.ndarray, boot: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "ci95": [float(value) for value in np.quantile(boot, [0.025, 0.975])],
        "probability_positive": float(np.mean(boot > 0.0)),
    }


def _bootstrap(
    rows: Mapping[str, np.ndarray], *, samples: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    size = len(next(iter(rows.values())))
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, size, size=(samples, size))
    summaries = {
        name: _summary(values, np.mean(values[indexes], axis=1))
        for name, values in rows.items()
    }
    predicted = rows["predicted_terminal_advantage"]
    realized = rows["realized_terminal_advantage"]
    correlations = []
    for sampled in indexes:
        value = fidelity.spearman(predicted[sampled], realized[sampled])
        if value is not None:
            correlations.append(float(value))
    point = fidelity.spearman(predicted, realized)
    correlation = {
        "spearman": point,
        "requested_samples": samples,
        "retained_samples": len(correlations),
        "ci95": (
            [float(value) for value in np.quantile(correlations, [0.025, 0.975])]
            if correlations
            else [None, None]
        ),
        "probability_positive": (
            float(np.mean(np.asarray(correlations) > 0.0))
            if correlations
            else None
        ),
    }
    return summaries, correlation


def _unavailable(
    *, source_interface: str, result_sha256: str, verification_sha256: str,
    binding_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "interface_version": "regretbench-horizon-value-decomposition-1",
        "status": "unavailable_mechanics_failed",
        "region": "unavailable_mechanics_failed",
        "source_interface": source_interface,
        "result_sha256": result_sha256,
        "verification_sha256": verification_sha256,
        "execution_binding_sha256": binding_sha256,
        "metrics": None,
        "can_change_status_authorization_or_claim_tier": False,
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def analyze(
    result: Mapping[str, Any], *, result_sha256: str,
    verification_sha256: str, binding_sha256: str,
    samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    interface = str(result.get("interface_version"))
    if interface not in POLICIES:
        raise ValueError("unsupported horizon-value source interface")
    dynamic, refresh = POLICIES[interface]
    mechanics = result.get("mechanics_gates") or {}
    if result.get("status") == "mechanics_failed" or mechanics.get("all_pass") is not True:
        return _unavailable(
            source_interface=interface,
            result_sha256=result_sha256,
            verification_sha256=verification_sha256,
            binding_sha256=binding_sha256,
        )
    tasks = result.get("tasks") or []
    if len(tasks) != 64:
        raise ValueError("horizon-value decomposition requires 64 tasks")

    values = {name: [] for name in QUANTITIES}
    task_ids = []
    wins = ties = losses = 0
    for task in tasks:
        selected = task.get("selected_roots") or {}
        dynamic_root = int(selected[dynamic])
        refresh_root = int(selected[refresh])
        if dynamic_root == refresh_root:
            continue
        terminal = task.get("conditioned_root_risks") or []
        immediate = task.get("myopic_refresh_brier_root_risks") or []
        policies = task.get("policies") or {}
        if len(terminal) != 4 or len(immediate) != 4:
            raise ValueError("stored horizon risk shape changed")
        i_dynamic = _finite(immediate[dynamic_root]["brier"], "I(dynamic)")
        i_refresh = _finite(immediate[refresh_root]["brier"], "I(refresh)")
        t_dynamic = _finite(terminal[dynamic_root]["brier"], "T(dynamic)")
        t_refresh = _finite(terminal[refresh_root]["brier"], "T(refresh)")
        r_dynamic = _finite(policies[dynamic]["brier"], "R(dynamic)")
        r_refresh = _finite(policies[refresh]["brier"], "R(refresh)")
        row = {
            "immediate_penalty": i_dynamic - i_refresh,
            "dynamic_horizon_value": i_dynamic - t_dynamic,
            "refresh_horizon_value": i_refresh - t_refresh,
            "differential_horizon_value": (i_dynamic - t_dynamic)
            - (i_refresh - t_refresh),
            "predicted_terminal_advantage": t_refresh - t_dynamic,
            "realized_terminal_advantage": r_refresh - r_dynamic,
        }
        residual = row["predicted_terminal_advantage"] - (
            row["differential_horizon_value"] - row["immediate_penalty"]
        )
        if abs(residual) > IDENTITY_TOLERANCE:
            raise ValueError("horizon-value accounting identity changed")
        if row["immediate_penalty"] < -IDENTITY_TOLERANCE:
            raise ValueError("refresh-myopic root is not immediate-risk minimizing")
        if row["predicted_terminal_advantage"] < -IDENTITY_TOLERANCE:
            raise ValueError("dynamic root is not terminal-risk minimizing")
        for name in QUANTITIES:
            values[name].append(row[name])
        task_ids.append(str(task["task_id"]))
        realized = row["realized_terminal_advantage"]
        if realized > 1e-12:
            wins += 1
        elif realized < -1e-12:
            losses += 1
        else:
            ties += 1

    arrays = {name: np.asarray(row, dtype=float) for name, row in values.items()}
    changed = len(task_ids)
    if changed == 0:
        summaries = {name: None for name in QUANTITIES}
        correlation = {
            "spearman": None,
            "requested_samples": samples,
            "retained_samples": 0,
            "ci95": [None, None],
            "probability_positive": None,
        }
    else:
        summaries, correlation = _bootstrap(arrays, samples=samples, seed=seed)

    if changed:
        predicted = arrays["predicted_terminal_advantage"]
        realized = arrays["realized_terminal_advantage"]
        nonzero = (np.abs(predicted) > 1e-12) & (np.abs(realized) > 1e-12)
        sign_accuracy = (
            float(np.mean(np.sign(predicted[nonzero]) == np.sign(realized[nonzero])))
            if np.any(nonzero)
            else None
        )
        rmse = float(np.sqrt(np.mean((predicted - realized) ** 2)))
        mae = float(np.mean(np.abs(predicted - realized)))
    else:
        nonzero = np.asarray([], dtype=bool)
        sign_accuracy = None
        rmse = mae = None

    retained = int(correlation["retained_samples"])
    enough = changed >= 16 and retained >= 0.9 * samples
    differential = summaries["differential_horizon_value"] if changed else None
    realized_summary = summaries["realized_terminal_advantage"] if changed else None
    forecast = bool(
        enough
        and differential["mean"] >= 0.01
        and differential["probability_positive"] >= 0.80
    )
    correlation_ok = bool(
        correlation["spearman"] is not None
        and correlation["spearman"] >= 0.15
        and correlation["probability_positive"] is not None
        and correlation["probability_positive"] >= 0.80
    )
    supported = bool(
        forecast
        and realized_summary["mean"] >= 0.02
        and realized_summary["ci95"][0] > 0.0
        and sign_accuracy is not None
        and sign_accuracy >= 0.60
        and correlation_ok
    )
    if not enough:
        region = "insufficient_changed_roots"
    elif not forecast:
        region = "no_material_horizon_forecast"
    elif (
        realized_summary["mean"] <= 0.0
        or sign_accuracy is None
        or sign_accuracy < 0.55
        or not correlation_ok
    ):
        region = "forecast_horizon_not_realized"
    elif supported:
        region = "descriptive_horizon_value_supported"
    else:
        region = "partial_horizon_value_evidence"

    return {
        "schema_version": 1,
        "interface_version": "regretbench-horizon-value-decomposition-1",
        "status": "complete_descriptive_non_gating",
        "region": region,
        "source_interface": interface,
        "source_result_status": result.get("status"),
        "result_sha256": result_sha256,
        "verification_sha256": verification_sha256,
        "execution_binding_sha256": binding_sha256,
        "metrics": {
            "task_count": len(tasks),
            "changed_root_task_count": changed,
            "changed_task_ids_sha256": hashlib.sha256(
                _canonical(task_ids).encode()
            ).hexdigest(),
            "quantities": summaries,
            "predicted_to_realized": {
                **correlation,
                "sign_accuracy": sign_accuracy,
                "sign_accuracy_n": int(np.sum(nonzero)),
                "rmse": rmse,
                "mae": mae,
            },
            "realized_dynamic_vs_refresh": {
                "wins": wins,
                "ties": ties,
                "losses": losses,
            },
            "bootstrap_seed": seed,
            "identity_tolerance": IDENTITY_TOLERANCE,
        },
        "literal_interpretation_gates": {
            "at_least_16_changed_roots": changed >= 16,
            "at_least_90pct_correlation_bootstraps_retained": retained >= 0.9 * samples,
            "differential_horizon_mean_at_least_001": bool(
                differential and differential["mean"] >= 0.01
            ),
            "differential_horizon_probability_positive_at_least_080": bool(
                differential and differential["probability_positive"] >= 0.80
            ),
            "realized_advantage_mean_at_least_002": bool(
                realized_summary and realized_summary["mean"] >= 0.02
            ),
            "realized_advantage_ci_lower_positive": bool(
                realized_summary and realized_summary["ci95"][0] > 0.0
            ),
            "sign_accuracy_at_least_060": bool(
                sign_accuracy is not None and sign_accuracy >= 0.60
            ),
            "spearman_at_least_015": bool(
                correlation["spearman"] is not None and correlation["spearman"] >= 0.15
            ),
            "spearman_probability_positive_at_least_080": bool(
                correlation["probability_positive"] is not None
                and correlation["probability_positive"] >= 0.80
            ),
        },
        "can_change_status_authorization_or_claim_tier": False,
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def run(run_dir: Path) -> dict[str, Any]:
    validate_binding()
    result, result_sha, verification_sha = fidelity._validated_inputs(run_dir)
    return analyze(
        result,
        result_sha256=result_sha,
        verification_sha256=verification_sha,
        binding_sha256=_sha256(BINDING),
    )


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# RegretBench Horizon-Value Decomposition",
        "",
        f"- Status: `{result['status']}`",
        f"- Region: `{result['region']}`",
        f"- Source result: `{result.get('source_result_status', 'unavailable')}`",
        "- Model calls: `0`; cost: `$0.00`.",
        "- This diagnostic cannot alter status, authorization, confirmation, or claim tier.",
    ]
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "",
            f"Changed-root tasks: `{metrics['changed_root_task_count']}` / `{metrics['task_count']}`.",
            "",
            "| Quantity | Mean | Sample SD | 95% CI | P(positive) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in QUANTITIES:
        row = metrics["quantities"][name]
        if row is None:
            lines.append(f"| `{name}` | n/a | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| `{name}` | {row['mean']:.6f} | {row['sample_sd']:.6f} | "
                f"[{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}] | "
                f"{row['probability_positive']:.4f} |"
            )
    pred = metrics["predicted_to_realized"]
    lines.extend(
        [
            "",
            f"Predicted-realized Spearman: `{pred['spearman']}`; sign accuracy: `{pred['sign_accuracy']}`; RMSE: `{pred['rmse']}`.",
            "",
            "## Literal Gates",
            "",
        ]
    )
    for name, value in result["literal_interpretation_gates"].items():
        lines.append(f"- `{name}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    result = run(args.run_dir.resolve())
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.markdown:
        args.markdown.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.markdown.resolve().write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
