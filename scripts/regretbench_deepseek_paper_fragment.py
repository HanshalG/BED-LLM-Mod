#!/usr/bin/env python3
"""Render a verified frozen RegretBench report as a deterministic TeX fragment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


from scripts import regretbench_deepseek_frozen_report as frozen_report


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_PAPER_FRAGMENT_PROTOCOL_20260807.md"
)
FRAGMENT_PROTOCOL_SHA256 = (
    "510ca51af21e40d36e7548eceed16085e44f4d1f8ed0a8d943476041b82fa823"
)
DEFAULT_OUTPUT = REPO_ROOT / "paper/generated/regretbench_result.tex"
CONTROL_LABELS = {
    "myopic_width": "Myopic width",
    "history_blind_depth2": "History-blind d2",
    "fixed_depth2": "Fixed-support d2",
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


def _context_paragraph(stage: str) -> str:
    return (
        "The released RegretBench intents, aliases, facets, slots, and finite CIG "
        "were hidden from the planner. DeepSeek generated eight semantic "
        "hypotheses, four clarification questions, and per-hypothesis reply "
        "likelihoods; the official mapper supplied exact environment replies only "
        "after selection. The "
        + stage
        + " cohort contains 64 tasks."
    )


def build_fragment(run_dir: Path, *, stage: str) -> tuple[str, dict[str, Any]]:
    if sha256_file(FRAGMENT_PROTOCOL) != FRAGMENT_PROTOCOL_SHA256:
        raise ValueError("RegretBench paper-fragment protocol changed")
    saved_path = run_dir / "FROZEN_REPORT.json"
    saved = _load(saved_path)
    expected = frozen_report.build_report(run_dir, stage=stage)
    if _canonical(saved) != _canonical(expected):
        raise ValueError("frozen report does not replay exactly")
    tier = str(saved["claim_tier"])
    interpretation = str(saved["interpretation"])
    lines = [
        "% Generated only from an independently verified frozen RegretBench report.",
        "\\paragraph{RegretBench: LLM-native path-dependent belief planning.}",
        _context_paragraph(stage),
        "",
        "\\textbf{Frozen interpretation.} " + interpretation,
    ]
    comparisons = saved.get("paired_primary_comparisons")
    if comparisons is not None:
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
                "\\caption{Frozen RegretBench "
                + stage
                + " comparisons on aligned generated-likelihood terminal Brier. "
                "Differences are dynamic depth two minus control; action-invalid "
                "or first-reply-unmodelled trajectories receive the "
                "preregistered penalty.}",
                "\\label{tab:regretbench-dynamic}",
                "\\end{table}",
            ]
        )
        correlation = saved["predicted_to_realized_dynamic_myopic"]
        lines.extend(
            [
                "",
                "On changed dynamic/myopic roots, predicted advantage versus "
                "realized Brier advantage has Spearman $"
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
        alignment = saved["alignment_complete_diagnostic"]
        myopic_alignment = alignment["controls"]["myopic_width"]
        aligned_brier = myopic_alignment["brier_dynamic_minus_control"]
        aligned_ci = aligned_brier["ci95"]
        lines.extend(
            [
                "",
                "On the preregistered subset where dynamic and myopic both had "
                "action-valid, truth-consistent first-reply paths ($n="
                + str(myopic_alignment["eligible_task_count"])
                + "$), dynamic-minus-myopic Brier was $"
                + _number(aligned_brier["mean"])
                + "$ (95\\% CI $["
                + _number(aligned_ci[0])
                + ","
                + _number(aligned_ci[1])
                + "]$); the alignment-complete corroboration flag was "
                + str(
                    alignment["alignment_complete_corroboration"]["all_pass"]
                ).lower()
                + ". This non-rescuing diagnostic cannot change the frozen tier.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "No policy-efficacy table is shown because the frozen mechanics "
                "contract failed.",
            ]
        )
    lines.extend(
        [
            "",
            "Fresh regeneration, the optional Luna baseline, pooled analyses, "
            "subgroups, and the alignment-complete diagnostic cannot change this "
            "tier.",
            "",
        ]
    )
    metadata = {
        "schema_version": 1,
        "interface_version": "regretbench-paper-fragment-1",
        "status": "rendered",
        "stage": stage,
        "claim_tier": tier,
        "interpretation": interpretation,
        "fragment_protocol_sha256": FRAGMENT_PROTOCOL_SHA256,
        "report_sha256": sha256_file(saved_path),
        "result_sha256": saved["result_sha256"],
        "verification_sha256": saved["verification_sha256"],
        "manuscript_claim_is_deterministic": True,
        "model_calls": 0,
        "cost_usd": 0.0,
    }
    return "\n".join(lines), metadata


def write_fragment(
    run_dir: Path, *, stage: str, output: Path = DEFAULT_OUTPUT
) -> dict[str, Any]:
    tex, metadata = build_fragment(run_dir, stage=stage)
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
    parser.add_argument(
        "--stage", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = write_fragment(
        args.run_dir.resolve(), stage=args.stage, output=args.output.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
