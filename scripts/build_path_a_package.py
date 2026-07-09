from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.compare_location_depth_sweeps import (
    compare_depth_sweeps,
    plot_comparison,
    plot_headline_rmse,
    write_comparison_report,
)
from scripts.cost_vs_depth_table import write_cost_table
from scripts.extract_location_qualitative_examples import extract_qualitative_examples
from scripts.location_fixed_root_depth_sweep import _plot_paired_trial_differences
from scripts.validate_path_a_package import summary_payload, validate_path_a_package


DEFAULT_CONSTRAINED = Path("runs/loc_branch_decoy_local_constrained_final50_26b_a4b")
DEFAULT_UNCONSTRAINED = Path("runs/loc_branch_decoy_local_unconstrained_final50_26b_a4b")


def _load_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "fixed_root_depth_sweep_metrics.json" if path.is_dir() else path
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if "aggregate" not in data or "paired_delta_vs_eig" not in data:
        raise ValueError(f"{summary_path} does not look like a fixed-root depth sweep summary")
    return data


def build_path_a_package(
    *,
    constrained: Path,
    unconstrained: Path,
    output_dir: Path,
    plot_dir: Path,
    cost_dir: Path,
    qualitative_dir: Path,
    run_name: str,
    validate_root: Path,
) -> dict[str, Any]:
    constrained_summary = _load_summary(constrained)
    unconstrained_summary = _load_summary(unconstrained)
    comparison = compare_depth_sweeps(constrained_summary, unconstrained_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    comparison_summary_path = output_dir / f"{run_name}_summary.json"
    report_path = output_dir / f"{run_name}_REPORT.md"
    contrast_plot_path = plot_dir / f"{run_name}.png"
    headline_plot_path = plot_dir / f"{run_name}_headline_rmse.png"
    paired_trial_delta_plot_path = plot_dir / f"{run_name}_paired_trial_rmse_deltas.png"

    comparison_summary_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_comparison_report(report_path, comparison)
    plot_comparison(contrast_plot_path, comparison)
    plot_headline_rmse(headline_plot_path, comparison)
    _plot_paired_trial_differences(constrained_summary, paired_trial_delta_plot_path)

    cost_json_path, cost_md_path, cost_plot_path = write_cost_table(
        [constrained, unconstrained],
        cost_dir,
        run_name,
    )
    qualitative_constrained = extract_qualitative_examples(
        summary_path=constrained,
        output_dir=qualitative_dir,
        run_label=f"{run_name}_constrained",
    )
    qualitative_unconstrained = extract_qualitative_examples(
        summary_path=unconstrained,
        output_dir=qualitative_dir,
        run_label=f"{run_name}_unconstrained",
    )
    validation = summary_payload(validate_path_a_package(validate_root))
    return {
        "comparison_summary": str(comparison_summary_path),
        "comparison_report": str(report_path),
        "contrast_plot": str(contrast_plot_path),
        "headline_plot": str(headline_plot_path),
        "paired_trial_delta_plot": str(paired_trial_delta_plot_path),
        "cost_json": str(cost_json_path),
        "cost_report": str(cost_md_path),
        "cost_plot": str(cost_plot_path) if cost_plot_path is not None else None,
        "qualitative_constrained_json": qualitative_constrained["json_path"],
        "qualitative_constrained_report": qualitative_constrained["markdown_path"],
        "qualitative_unconstrained_json": qualitative_unconstrained["json_path"],
        "qualitative_unconstrained_report": qualitative_unconstrained["markdown_path"],
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Path A MPP report package from completed sweep runs.")
    parser.add_argument("--constrained", type=Path, default=DEFAULT_CONSTRAINED)
    parser.add_argument("--unconstrained", type=Path, default=DEFAULT_UNCONSTRAINED)
    parser.add_argument("--output-dir", type=Path, default=Path("results/location_depth_sweeps"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots/location_depth_sweeps"))
    parser.add_argument("--cost-dir", type=Path, default=Path("results/cost_vs_depth"))
    parser.add_argument("--qualitative-dir", type=Path, default=Path("results/location_qualitative"))
    parser.add_argument("--run-name", default="location_branch_decoy_depth_contrast_26b_a4b")
    parser.add_argument("--validate-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    payload = build_path_a_package(
        constrained=args.constrained,
        unconstrained=args.unconstrained,
        output_dir=args.output_dir,
        plot_dir=args.plot_dir,
        cost_dir=args.cost_dir,
        qualitative_dir=args.qualitative_dir,
        run_name=args.run_name,
        validate_root=args.validate_root,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            if key != "validation":
                print(f"{key}: {value}")
        print(f"validation_ok: {payload['validation']['ok']}")
    raise SystemExit(0 if payload["validation"]["ok"] else 1)


if __name__ == "__main__":
    main()
