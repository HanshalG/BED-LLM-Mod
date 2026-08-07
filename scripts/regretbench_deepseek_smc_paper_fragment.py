#!/usr/bin/env python3
"""Render a verified RegretBench SMC report as deterministic TeX."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


from scripts import regretbench_deepseek_smc_frozen_report as frozen_report


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_SMC_PAPER_FRAGMENT_PROTOCOL_20260807.md"
)
FRAGMENT_PROTOCOL_SHA256 = (
    "f13cd93f4674e211ce33e51825da6f28123256128c7ebb9e192a7aac84ddc7da"
)
REPORT_GENERATOR_SHA256 = (
    "2996d0cfd18e2ade3a7947492e0f07f13b1e452ee7d57a5cfbda27659bb11d43"
)
PRERESULT_MANUSCRIPT_SHA256 = (
    "6ece61e00c284c961e08375b873a410a959bf9bab978f847c0d099dbaf7453bf"
)
DEFAULT_OUTPUT = REPO_ROOT / "paper/generated/regretbench_result.tex"
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
    run_dir: Path, *, primary_dir: Path
) -> tuple[str, dict[str, Any]]:
    if sha256_file(FRAGMENT_PROTOCOL) != FRAGMENT_PROTOCOL_SHA256:
        raise ValueError("RegretBench SMC paper-fragment protocol changed")
    if (
        sha256_file(Path(frozen_report.__file__).resolve())
        != REPORT_GENERATOR_SHA256
    ):
        raise ValueError("RegretBench SMC report generator changed")
    saved_path = run_dir / "FROZEN_REPORT.json"
    saved = _load(saved_path)
    expected = frozen_report.build_report(run_dir, primary_dir=primary_dir)
    if _canonical(saved) != _canonical(expected):
        raise ValueError("SMC frozen report does not replay exactly")
    tier = str(saved["claim_tier"])
    interpretation = str(saved["interpretation"])
    lines = [
        "% Generated only from an independently verified frozen RegretBench SMC report.",
        "\\paragraph{RegretBench: non-myopic planning over LLM semantic particles.}",
        "The released RegretBench intents, aliases, facets, slots, and finite CIG were hidden from the planner. DeepSeek annotated reply likelihoods for eight exact banked semantic parent slots and generated path-dependent SMC transitions that retained two through six particles and revised the remainder. The development cohort contains 64 tasks.",
        "",
        "\\textbf{Frozen interpretation.} " + interpretation,
    ]
    comparisons = saved.get("paired_primary_comparisons")
    if comparisons is None:
        lines.extend(
            [
                "",
                "No policy-efficacy table is shown because the frozen SMC mechanics contract failed.",
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
        for name in frozen_report.BASELINES:
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
                "\\caption{Frozen RegretBench SMC development comparisons on aligned generated-likelihood terminal Brier. Differences are SMC dynamic depth two minus control.}",
                "\\label{tab:regretbench-smc-dynamic}",
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
            ]
        )
    stability = saved["draw_stability_diagnostic"]
    stable = stability["stable_tasks_descriptive"]
    unstable = stability["unstable_tasks_descriptive"]
    lines.extend(
        [
            "",
            "The non-gating two-draw diagnostic selected identical dynamic roots on "
            + str(stability["draw_agreement_count"])
            + "/64 tasks. Dynamic-minus-refresh-myopic Brier was $"
            + _number(stable["mean_dynamic_minus_refresh_myopic_brier"])
            + "$ on stable tasks ($n="
            + str(stable["task_count"])
            + "$) and $"
            + _number(unstable["mean_dynamic_minus_refresh_myopic_brier"])
            + "$ on unstable tasks ($n="
            + str(unstable["task_count"])
            + "$).",
            "",
            "Alignment-complete subsets, branch-draw stability, fresh final SMC regeneration, optional Luna thinking, pooled analyses, and subgroups are descriptive and cannot change this development tier. A passed tier is provisional and requires independent confirmation.",
            "",
        ]
    )
    metadata = {
        "schema_version": 1,
        "interface_version": "regretbench-smc-paper-fragment-1",
        "status": "rendered",
        "stage": "development",
        "claim_tier": tier,
        "interpretation": interpretation,
        "fragment_protocol_sha256": FRAGMENT_PROTOCOL_SHA256,
        "report_generator_sha256": REPORT_GENERATOR_SHA256,
        "report_sha256": sha256_file(saved_path),
        "result_sha256": saved["result_sha256"],
        "verification_sha256": saved["verification_sha256"],
        "development_can_be_called_confirmed": False,
        "manuscript_claim_is_deterministic": True,
        "model_calls": 0,
        "cost_usd": 0.0,
    }
    return "\n".join(lines), metadata


def write_fragment(
    run_dir: Path,
    *,
    primary_dir: Path,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    tex, metadata = build_fragment(run_dir, primary_dir=primary_dir)
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
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = write_fragment(
        args.run_dir.resolve(),
        primary_dir=args.primary_dir.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
