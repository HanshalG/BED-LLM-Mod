#!/usr/bin/env python3
"""Render a verified RegretBench SMC confirmation report as deterministic TeX."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts import regretbench_deepseek_smc_confirmation_report as frozen_report


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_SMC_CONFIRMATION_PAPER_FRAGMENT_PROTOCOL_20260807.md"
)
FRAGMENT_PROTOCOL_SHA256 = (
    "ea43e839a079aba6ab60bf2c850d62b9de9fc43437eda210230e4c015ad5dc4c"
)
REPORT_GENERATOR_SHA256 = (
    "5f030ed57360e9876de473d0037ffd103bf10efdd9d8d891320a85ba34d5e338"
)
PRERESULT_MANUSCRIPT_SHA256 = (
    "6ece61e00c284c961e08375b873a410a959bf9bab978f847c0d099dbaf7453bf"
)
DEFAULT_OUTPUT = REPO_ROOT / "paper/generated/regretbench_result.tex"
BASELINES = (
    "smc_myopic_refresh_brier",
    "smc_myopic_brier",
    "smc_myopic_width",
    "smc_history_blind_depth2",
    "smc_fixed_depth2",
    "random",
)
CONTROL_LABELS = {
    "smc_myopic_refresh_brier": "Refresh-matched myopic",
    "smc_myopic_brier": "Fixed-parent Brier myopic",
    "smc_myopic_width": "Myopic EIG",
    "smc_history_blind_depth2": "History-blind SMC d2",
    "smc_fixed_depth2": "Fixed-parent d2",
    "random": "Random",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _number(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_fragment(
    run_dir: Path, *, parent_dir: Path
) -> tuple[str, dict[str, Any]]:
    if sha256_file(FRAGMENT_PROTOCOL) != FRAGMENT_PROTOCOL_SHA256:
        raise ValueError("SMC confirmation paper-fragment protocol changed")
    if sha256_file(Path(frozen_report.__file__).resolve()) != REPORT_GENERATOR_SHA256:
        raise ValueError("SMC confirmation report generator changed")
    saved_path = run_dir / "FROZEN_REPORT.json"
    saved = _load(saved_path)
    expected = frozen_report.build_report(run_dir, parent_dir=parent_dir)
    if _canonical(saved) != _canonical(expected):
        raise ValueError("SMC confirmation report does not replay exactly")
    tier = str(saved["claim_tier"])
    interpretation = str(saved["interpretation"])
    if tier not in {
        "smc_confirmation_mechanics_failure_no_result",
        "smc_confirmation_null_no_headline_result",
        "smc_confirmed_nonmyopic_semantic_particle_result",
    }:
        raise ValueError("unregistered SMC confirmation claim tier")

    lines = [
        "% Generated only from an independently verified frozen RegretBench SMC confirmation.",
        "\\paragraph{RegretBench: confirmation of non-myopic LLM semantic-particle planning.}",
        "The released RegretBench intents, aliases, facets, slots, and finite CIG were hidden from the planner. On a prospectively untouched 64-task cohort, DeepSeek annotated semantic reply likelihoods for eight banked parent slots and generated path-dependent SMC transitions that retained two through six particles and revised the remainder.",
        "",
        "\\textbf{Frozen confirmation interpretation.} " + interpretation,
    ]
    comparisons = saved.get("paired_primary_comparisons")
    if comparisons is None:
        lines.extend(
            [
                "",
                "No confirmation efficacy table is shown because the frozen confirmation mechanics contract failed. The development signal remains provisional rather than confirmed.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "\\begin{table}[t]",
                "\\centering",
                "\\small",
                "\\setlength{\\tabcolsep}{3pt}",
                "\\begin{tabular}{lrrrrr}",
                "\\toprule",
                "Control & Root $\\Delta$ & $\\Delta$ Brier (SD) & 95\\% CI & $P(\\mathrm{improve})$ & W/T/L \\\\",
                "\\midrule",
            ]
        )
        for name in BASELINES:
            row = comparisons[name]
            brier = row["brier_dynamic_minus_baseline"]
            wtl = row["wins_ties_losses"]
            ci = brier["ci95"]
            lines.append(
                f"{CONTROL_LABELS[name]} & {saved['root_disagreements'][name]} & "
                f"{_number(brier['mean'])} ({_number(brier['sample_sd'])}) & "
                f"[{_number(ci[0])}, {_number(ci[1])}] & "
                f"{_number(brier['probability_improvement'], 3)} & "
                f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} \\\\"
            )
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\caption{Preregistered RegretBench SMC confirmation comparisons on aligned generated-likelihood terminal Brier. Differences are SMC dynamic depth two minus control; development and confirmation are not pooled.}",
                "\\label{tab:regretbench-smc-confirmation}",
                "\\end{table}",
            ]
        )
        correlation = saved["predicted_to_realized"]["refresh_matched"]
        lines.extend(
            [
                "",
                "The headline horizon-isolating control is refresh-matched myopic Brier: it uses the same generated one-step SMC transitions and terminal utility but does not value the second question. On changed roots, predicted versus realized Brier advantage had Spearman $"
                + _number(correlation.get("spearman"), 3)
                + "$ (95\\% CI $["
                + _number(correlation.get("ci95", [None, None])[0], 3)
                + ","
                + _number(correlation.get("ci95", [None, None])[1], 3)
                + "]$, $n="
                + str(correlation.get("n"))
                + "$).",
                "The frozen confirmation tier uses the same prospectively amended 13-gate conjunction for the refresh-matched and history-blind primary claims. All 34 original diagnostics remain reported, but the 21 secondary fixed-support, EIG, and calibration gates cannot veto or rescue that tier.",
            ]
        )
    stability = saved["draw_stability_diagnostic"]
    lines.extend(
        [
            "",
            "The non-gating two-draw diagnostic selected identical dynamic roots on "
            + str(stability["draw_agreement_count"])
            + "/64 tasks.",
            "",
            "Alignment-complete subsets, branch-draw stability, fresh final SMC regeneration, optional baselines, pooled development--confirmation analyses, and subgroups are descriptive and cannot change this confirmation tier.",
            "",
        ]
    )
    metadata = {
        "schema_version": 1,
        "interface_version": "regretbench-smc-confirmation-paper-fragment-1",
        "status": "rendered",
        "stage": "confirmation",
        "claim_tier": tier,
        "interpretation": interpretation,
        "fragment_protocol_sha256": FRAGMENT_PROTOCOL_SHA256,
        "report_generator_sha256": REPORT_GENERATOR_SHA256,
        "report_sha256": sha256_file(saved_path),
        "result_sha256": saved["result_sha256"],
        "verification_sha256": saved["verification_sha256"],
        "claim_gate_amendment_sha256": saved["claim_gate_amendment_sha256"],
        "primary_claim_all_pass": saved["primary_claim_all_pass"],
        "all_34_diagnostic_gates_pass": saved["all_34_diagnostic_gates_pass"],
        "confirmed_claim_authorized": tier
        == "smc_confirmed_nonmyopic_semantic_particle_result",
        "development_and_confirmation_are_not_pooled": True,
        "manuscript_claim_is_deterministic": True,
        "model_calls": 0,
        "cost_usd": 0.0,
    }
    return "\n".join(lines), metadata


def write_fragment(
    run_dir: Path,
    *,
    parent_dir: Path,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    tex, metadata = build_fragment(run_dir, parent_dir=parent_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    metadata_path = output.with_suffix(".json")
    metadata["tex_sha256"] = sha256_file(output)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "written",
        "claim_tier": metadata["claim_tier"],
        "tex_path": str(output),
        "tex_sha256": metadata["tex_sha256"],
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = write_fragment(
        args.run_dir.resolve(),
        parent_dir=args.parent_dir.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
