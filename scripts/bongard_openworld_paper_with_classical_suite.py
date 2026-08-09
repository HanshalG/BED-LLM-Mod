#!/usr/bin/env python3
"""Render the Bongard paper fragment with mandatory DINO and SigLIP controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_classical_suite_outcome as suite_outcome
from scripts import bongard_openworld_luna_confirmation64 as confirmation
from scripts import bongard_openworld_luna_confirmation64_daily_execute as confirmation_daily
from scripts import bongard_openworld_luna_development32_daily_execute as development_daily
from scripts import bongard_openworld_luna_paper_fragment as luna_fragment
from scripts import bongard_openworld_luna_vlm_development as development


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-paper-with-classical-suite-2"
LUNA_RENDERER_SHA256 = (
    "e9462325703667cb2e2133c27134643c6fc13f269bc5096768f72076124c84d1"
)
CLASSICAL_SUITE_OUTCOME_SHA256 = (
    "8c2bc93b3d416a47c2d1e19112a670f270b79712c22e4ba3d2181308e3d0e032"
)
DEFAULT_OUTPUT = luna_fragment.DEFAULT_OUTPUT


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _number(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("classical-suite paper metric is not finite")
    return f"{float(value):.{digits}f}"


def verify_bound_implementations() -> dict[str, str]:
    observed = {
        "luna_renderer": suite_outcome.siglip.sha256_file(
            REPO_ROOT / "scripts/bongard_openworld_luna_paper_fragment.py"
        ),
        "classical_suite_outcome": suite_outcome.siglip.sha256_file(
            REPO_ROOT / "scripts/bongard_openworld_classical_suite_outcome.py"
        ),
    }
    expected = {
        "luna_renderer": LUNA_RENDERER_SHA256,
        "classical_suite_outcome": CLASSICAL_SUITE_OUTCOME_SHA256,
    }
    if observed != expected:
        raise ValueError(
            "bound Bongard classical-suite renderers changed: expected "
            f"{expected}, observed {observed}"
        )
    suite_outcome.verify_frozen_inputs()
    return observed


def _encoder_tex(
    *,
    subject: str,
    comparison_name: str,
    pooled: Mapping[str, Any],
    myopic_policy: str,
    depth2_policy: str,
    horizon: Mapping[str, Any],
    luna_comparison: Mapping[str, Any],
) -> str:
    horizon_ci = horizon["bootstrap_95pct_ci"]
    luna_ci = luna_comparison["bootstrap_95pct_ci"]
    values = (
        pooled[myopic_policy]["mean_brier"],
        pooled[depth2_policy]["mean_brier"],
        horizon["mean"],
        *horizon_ci,
        luna_comparison["mean"],
        *luna_ci,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("classical-suite paper values are non-finite")
    return (
        f"{subject} obtained endpoint Brier "
        f"{_number(pooled[myopic_policy]['mean_brier'])} with one-step PIG and "
        f"{_number(pooled[depth2_policy]['mean_brier'])} with exact depth-two "
        "lookahead. Its paired depth-two minus myopic difference was "
        f"{_number(horizon['mean'])} (95\\% bootstrap CI "
        f"$[{_number(horizon_ci[0])},{_number(horizon_ci[1])}]$). Luna dynamic "
        f"depth two minus {comparison_name} depth two was "
        f"{_number(luna_comparison['mean'])} in Brier (95\\% bootstrap CI "
        f"$[{_number(luna_ci[0])},{_number(luna_ci[1])}]$)."
    )


def classical_tex_lines(result: Mapping[str, Any]) -> list[str]:
    if (
        result.get("status") != "classical_suite_complete"
        or result.get("all_gates_pass") is not True
        or result.get("authorizes_paid_calls") is not False
    ):
        raise ValueError("paper input is not a complete frozen classical suite")
    dino = result["dino"]
    siglip = result["siglip"]
    dino_comparison = result["paired_luna_minus_dino"][
        "luna_dynamic_minus_dinov2_depth2"
    ]["mean_brier"]
    siglip_comparison = result["paired_luna_minus_siglip"][
        "luna_dynamic_minus_siglip_depth2"
    ]["mean_brier"]
    return [
        "\\paragraph{Frozen classical vision comparators.}",
        _encoder_tex(
            subject="The fixed DINOv2-small prototype model",
            comparison_name="DINOv2",
            pooled=dino["pooled"],
            myopic_policy="dinov2_myopic",
            depth2_policy="dinov2_depth2",
            horizon=dino["paired_depth2_minus_myopic"]["mean_brier"],
            luna_comparison=dino_comparison,
        ),
        _encoder_tex(
            subject="The fixed SigLIP2-So400m prototype model",
            comparison_name="SigLIP2",
            pooled=siglip["pooled"],
            myopic_policy="siglip_myopic",
            depth2_policy="siglip_depth2",
            horizon=siglip["paired_depth2_minus_myopic"]["mean_brier"],
            luna_comparison=siglip_comparison,
        ),
        (
            "For Luna-minus-classical Brier, negative values favor Luna. Both "
            "comparators are reported regardless of direction and do not test "
            "universal classical impossibility."
        ),
    ]


def _validated_paired_summary(
    comparison: Mapping[str, Any], metric: str
) -> dict[str, Any]:
    summary = comparison.get(metric)
    if not isinstance(summary, Mapping):
        raise ValueError(f"classical challenge lacks {metric}")
    mean = summary.get("mean")
    interval = summary.get("bootstrap_95pct_ci")
    if (
        not isinstance(mean, (int, float))
        or not math.isfinite(float(mean))
        or not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes))
        or len(interval) != 2
        or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in interval
        )
        or float(interval[0]) > float(interval[1])
        or summary.get("bootstrap_draws") != 20_000
    ):
        raise ValueError(f"classical challenge has invalid {metric}")
    return {
        "mean": float(mean),
        "bootstrap_95pct_ci": [float(interval[0]), float(interval[1])],
        "bootstrap_draws": 20_000,
    }


def classical_claim_scope(result: Mapping[str, Any]) -> dict[str, Any]:
    """Classify task-level scope without changing the within-Luna claim tier."""
    if (
        result.get("status") != "classical_suite_complete"
        or result.get("all_gates_pass") is not True
        or result.get("authorizes_paid_calls") is not False
    ):
        raise ValueError("classical claim scope requires the complete frozen suite")
    comparisons = {
        "dinov2_depth2": result.get("paired_luna_minus_dino", {}).get(
            "luna_dynamic_minus_dinov2_depth2"
        ),
        "siglip_depth2": result.get("paired_luna_minus_siglip", {}).get(
            "luna_dynamic_minus_siglip_depth2"
        ),
    }
    details: dict[str, Any] = {}
    for name, comparison in comparisons.items():
        if not isinstance(comparison, Mapping):
            raise ValueError(f"classical challenge lacks {name}")
        brier = _validated_paired_summary(comparison, "mean_brier")
        log_loss = _validated_paired_summary(comparison, "mean_log_loss")
        details[name] = {
            "luna_minus_classical_brier": brier,
            "luna_minus_classical_log_loss": log_loss,
            "brier_interval_strictly_below_zero": (
                brier["bootstrap_95pct_ci"][1] < 0.0
            ),
            "log_loss_nonworse": log_loss["mean"] <= 0.0,
        }
        details[name]["clears_challenge"] = (
            brier["mean"] < 0.0
            and details[name]["brier_interval_strictly_below_zero"]
            and details[name]["log_loss_nonworse"]
        )
    clears_both = all(row["clears_challenge"] for row in details.values())
    return {
        "status": (
            "luna_clears_both_fixed_classical_challengers"
            if clears_both
            else "within_luna_only_no_task_level_necessity"
        ),
        "clears_both_fixed_classical_challengers": clears_both,
        "criterion": (
            "For both DINOv2 depth two and SigLIP2 depth two, Luna dynamic minus "
            "classical Brier must have mean below zero and paired two-sided 95% "
            "bootstrap upper endpoint below zero, with mean log loss nonworse."
        ),
        "changes_within_luna_claim_tier": False,
        "authorizes_paid_calls": False,
        "comparisons": details,
    }


def _headline_with_classical_scope(
    *,
    original_headline: str,
    original_metadata: Mapping[str, Any],
    scope: Mapping[str, Any] | None,
) -> str:
    headline = original_metadata.get("headline")
    if not isinstance(headline, Mapping):
        raise ValueError("original Luna headline metadata is missing")
    authorized = headline.get("authorized") is True
    abstract_tex = headline.get("abstract_tex")
    contribution_tex = headline.get("contribution_tex")
    if not isinstance(abstract_tex, str) or not isinstance(contribution_tex, str):
        raise ValueError("original Luna headline copy is malformed")
    if authorized is not bool(abstract_tex and contribution_tex):
        raise ValueError("original Luna headline authorization is inconsistent")
    if not authorized:
        return original_headline
    if (
        original_metadata.get("stage") != "confirmation"
        or original_metadata.get("claim_tier") != "full_llm_native_confirmation"
        or scope is None
    ):
        raise ValueError("classical headline qualification lacks full confirmation")
    if scope.get("clears_both_fixed_classical_challengers") is True:
        abstract_qualifier = (
            " Luna dynamic depth two also clears both frozen fixed semantic-vision "
            "challengers: its paired endpoint-Brier intervals versus DINOv2 and "
            "SigLIP2 are strictly below zero with nonworse mean log loss."
        )
        contribution_qualifier = (
            "\\item a prospectively scoped comparison showing lower endpoint Brier "
            "than both frozen DINOv2 and SigLIP2 depth-two planners, with paired "
            "intervals below zero and nonworse mean log loss;"
        )
    elif scope.get("status") == "within_luna_only_no_task_level_necessity":
        abstract_qualifier = (
            " This is a within-Luna path-dependent-belief result; it does not clear "
            "both frozen classical challengers and therefore does not establish "
            "task-level LLM necessity."
        )
        contribution_qualifier = (
            "\\item a prospectively scoped classical comparison that limits the "
            "Bongard finding to within-Luna path-dependent belief dynamics rather "
            "than task-level LLM necessity;"
        )
    else:
        raise ValueError("unknown classical headline scope")
    return "\n".join(
        [
            "\\renewcommand{\\BongardAbstractResult}{%",
            abstract_tex + abstract_qualifier,
            "}",
            "\\renewcommand{\\BongardContributionResult}{%",
            contribution_tex,
            contribution_qualifier,
            "}",
            "",
        ]
    )


def replay_classical_suite(
    *,
    stage: str,
    saved_path: Path,
    result_path: Path,
    block_results: Sequence[Path],
    wrapper_result: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    replay = suite_outcome.run_outcome(
        stage=stage,
        result_path=result_path,
        output_path=output_path,
        block_results=block_results,
        wrapper_result=wrapper_result,
    )
    saved = _load(saved_path)
    if _canonical(saved) != _canonical(replay):
        raise ValueError("saved classical-suite outcome does not independently replay")
    return replay


def write_combined_fragment(
    *,
    stage: str,
    output: Path,
    classical_suite_path: Path | None,
    claim_report_path: Path | None = None,
    combined_result: Path | None = None,
    block_results: Sequence[Path] = (),
    failure_path: Path | None = None,
) -> dict[str, Any]:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    bound = verify_bound_implementations()
    with tempfile.TemporaryDirectory(prefix="bongard-paper-classical-") as tmp:
        temporary = Path(tmp)
        original_output = temporary / "bongard_openworld_result.tex"
        original = luna_fragment.write_fragment(
            stage=stage,
            output=original_output,
            claim_report_path=claim_report_path,
            combined_result=combined_result,
            block_results=block_results,
            failure_path=failure_path,
        )
        original_tex = original_output.read_text(encoding="utf-8")
        original_headline = original_output.with_name(
            luna_fragment.HEADLINE_FILENAME
        ).read_text(encoding="utf-8")
        original_metadata = _load(original_output.with_suffix(".json"))

        if stage == "confirmation-mechanics-failure":
            if classical_suite_path is not None:
                raise ValueError(
                    "mechanics failure cannot have a classical endpoint result"
                )
            addendum = [
                "\\paragraph{Frozen classical vision comparators.}",
                "No DINO or SigLIP endpoint comparison is rendered because no replay-verified combined endpoint result exists.",
            ]
            suite_metadata = None
            claim_scope = None
        else:
            if classical_suite_path is None or combined_result is None:
                raise ValueError(
                    "endpoint result rendering requires the frozen classical suite"
                )
            replay = replay_classical_suite(
                stage=stage,
                saved_path=classical_suite_path,
                result_path=combined_result,
                block_results=block_results,
                wrapper_result=None,
                output_path=temporary / "CLASSICAL_SUITE_REPLAY.json",
            )
            addendum = classical_tex_lines(replay)
            claim_scope = classical_claim_scope(replay)
            suite_metadata = {
                "outcome_sha256": suite_outcome.siglip.sha256_file(
                    classical_suite_path
                ),
                "stage_result_sha256": replay["stage_result_sha256"],
                "status": replay["status"],
                "claim_scope": claim_scope,
            }

        combined_tex = original_tex.rstrip() + "\n\n" + "\n".join(addendum) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(combined_tex, encoding="utf-8")
        headline_path = output.with_name(luna_fragment.HEADLINE_FILENAME)
        qualified_headline = _headline_with_classical_scope(
            original_headline=original_headline,
            original_metadata=original_metadata,
            scope=claim_scope,
        )
        headline_path.write_text(qualified_headline, encoding="utf-8")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "stage": stage,
            "claim_tier": original_metadata["claim_tier"],
            "bound_implementations": bound,
            "original_renderer": original,
            "original_metadata": original_metadata,
            "classical_suite": suite_metadata,
            "classical_claim_scope": claim_scope,
            "tex_sha256": suite_outcome.siglip.sha256_file(output),
            "headline_tex_sha256": suite_outcome.siglip.sha256_file(
                headline_path
            ),
            "model_calls": 0,
            "cost_usd": 0.0,
        }
        metadata_path = output.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "status": "written_with_mandatory_classical_suite",
        "stage": stage,
        "claim_tier": metadata["claim_tier"],
        "tex_path": str(output),
        "tex_sha256": metadata["tex_sha256"],
        "headline_path": str(headline_path),
        "headline_sha256": metadata["headline_tex_sha256"],
        "metadata_path": str(metadata_path),
        "metadata_sha256": suite_outcome.siglip.sha256_file(metadata_path),
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("development", "confirmation", "confirmation-mechanics-failure"),
        required=True,
    )
    parser.add_argument("--classical-suite", type=Path)
    parser.add_argument("--claim-report", type=Path)
    parser.add_argument("--combined-result", type=Path)
    parser.add_argument("--block-result", type=Path, action="append", default=[])
    parser.add_argument("--failure-record", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocks = args.block_result
    if not blocks:
        blocks = (
            [
                development_daily.BLOCK_DIRS[block_id] / "RESULT.json"
                for block_id in development.BLOCK_ORDER
            ]
            if args.stage == "development"
            else [
                confirmation_daily.BLOCK_DIRS[block_id] / "RESULT.json"
                for block_id in confirmation.BLOCK_ORDER
            ]
        )
    combined = args.combined_result
    claim = args.claim_report
    if args.stage == "development":
        combined = combined or development_daily.COMBINED_RESULT
        claim = claim or development_daily.ROOT / "CLAIM_REPORT.json"
    elif combined is None:
        combined = confirmation_daily.COMBINED_RESULT
    result = write_combined_fragment(
        stage=args.stage,
        output=args.output.resolve(),
        classical_suite_path=args.classical_suite,
        claim_report_path=claim,
        combined_result=combined,
        block_results=blocks,
        failure_path=args.failure_record,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
