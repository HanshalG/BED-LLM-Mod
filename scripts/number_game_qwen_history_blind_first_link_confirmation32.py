#!/usr/bin/env python3
"""Confirm the Number Game history-blind first-link calibration signal."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_qwen_history_blind_matched32 as base
from scripts import number_game_qwen_history_blind_matched32_v3 as v3
from scripts.number_game_qwen_history_blind_serving_smoke import sha256_file


SCHEMA_VERSION = 1
INTERFACE_VERSION = (
    "number-game-qwen-history-blind-first-link-confirmation32-1"
)
SOURCE_TREE_START = 32
SOURCE_TREE_SEEDS = tuple(range(80_032, 80_064))
CONTROL_SEED_START = 9_600_000
BOOTSTRAP_SEED = 9_700_000
ORIGINAL_BASE_SUMMARIZE_SCORES = base.summarize_scores

DEVELOPMENT_DIR = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_history_blind_matched32_v3"
    / "number-game-qwen-history-blind-matched32-v3-20260730T173000Z"
)
DEVELOPMENT_RESULT = DEVELOPMENT_DIR / "RESULT.json"
DEVELOPMENT_CONTROLS = DEVELOPMENT_DIR / "CONTROLS.json"
DEVELOPMENT_RESULT_SHA256 = (
    "29bb76074dfb53a6efef352fde88fbca6c3a7dc591d0936d132c82050f1dc71d"
)
DEVELOPMENT_CONTROLS_SHA256 = (
    "6db0e3e272cb843250174a2252098c4a88d3d4a154d3287ba5542f9d2eb09a1b"
)
DEVELOPMENT_SPEARMAN = 0.7167277167277167


def validate_development(
    *,
    result_path: Path = DEVELOPMENT_RESULT,
    controls_path: Path = DEVELOPMENT_CONTROLS,
) -> dict[str, Any]:
    if sha256_file(result_path) != DEVELOPMENT_RESULT_SHA256:
        raise ValueError("V3 development result hash changed")
    if sha256_file(controls_path) != DEVELOPMENT_CONTROLS_SHA256:
        raise ValueError("V3 development controls hash changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    analysis = result.get("analysis") or {}
    gates = analysis.get("scientific_gates") or {}
    selected = analysis.get("selected_root_prompt_conditioning") or {}
    if result.get("status") != "gated_null":
        raise ValueError("V3 development result status changed")
    if result.get("interface_version") != v3.INTERFACE_VERSION:
        raise ValueError("V3 development interface changed")
    if protocol.get("tree_indices") != list(range(32)):
        raise ValueError("V3 development cohort changed")
    if protocol.get("source_result_sha256") != base.SOURCE_RESULT_SHA256:
        raise ValueError("V3 development source result changed")
    if protocol.get("source_trees_sha256") != base.SOURCE_TREES_SHA256:
        raise ValueError("V3 development source trees changed")
    if protocol.get("source_targets_sha256") != base.SOURCE_TARGETS_SHA256:
        raise ValueError("V3 development source targets changed")
    if protocol.get("controls_sha256") != DEVELOPMENT_CONTROLS_SHA256:
        raise ValueError("V3 development controls binding changed")
    if gates.get(
        "changed_root_prompt_benefit_contrast_ci_above_zero"
    ) is not False:
        raise ValueError("V3 selected-root mean gate is not the frozen null")
    if analysis.get("directionally_coherent") is not False:
        raise ValueError("V3 conjunctive status changed")
    observed = selected.get(
        "prompt_benefit_contrast_to_realized_advantage_spearman"
    )
    if observed != DEVELOPMENT_SPEARMAN:
        raise ValueError("V3 development Spearman changed")
    return result


def summarize_scores(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    summary = ORIGINAL_BASE_SUMMARIZE_SCORES(
        rows,
        bootstrap_samples=bootstrap_samples,
    )
    gates = summary["scientific_gates"]
    prior_mean_gate = gates.pop(
        "changed_root_prompt_benefit_contrast_ci_above_zero"
    )
    spearman_interval = summary["bootstrap"][
        "prompt_benefit_contrast_to_realized_spearman_95pct"
    ]
    gates[
        "prompt_benefit_contrast_to_realized_spearman_ci_above_zero"
    ] = spearman_interval[0] > 0.0
    summary["descriptive_selected_root_mean_gate"] = {
        "changed_root_prompt_benefit_contrast_ci_above_zero": (
            prior_mean_gate
        ),
        "used_for_confirmation_success": False,
    }
    summary["directionally_coherent"] = all(gates.values())
    return summary


@contextmanager
def configured_base() -> Iterator[None]:
    overrides = {
        "INTERFACE_VERSION": INTERFACE_VERSION,
        "SOURCE_TREE_START": SOURCE_TREE_START,
        "SOURCE_TREE_SEEDS": SOURCE_TREE_SEEDS,
        "CONTROL_SEED_START": CONTROL_SEED_START,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "mechanics_gates": v3.mechanics_gates,
        "summarize_scores": summarize_scores,
    }
    originals = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def run_formal(
    *,
    output_dir: Path,
    run_id: str,
    smoke_result_path: Path = base.SMOKE_RESULT,
    adapter=None,
    remaining_credit: float | None = None,
    bootstrap_samples: int = base.BOOTSTRAP_SAMPLES,
    development_result_path: Path = DEVELOPMENT_RESULT,
    development_controls_path: Path = DEVELOPMENT_CONTROLS,
) -> dict[str, Any]:
    validate_development(
        result_path=development_result_path,
        controls_path=development_controls_path,
    )
    with configured_base():
        result = base.run_formal(
            output_dir=output_dir,
            run_id=run_id,
            smoke_result_path=smoke_result_path,
            adapter=adapter,
            remaining_credit=remaining_credit,
            bootstrap_samples=bootstrap_samples,
        )
    controls_path = output_dir / "CONTROLS.json"
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    result["protocol"].update(
        {
            "interface_version": INTERFACE_VERSION,
            "development_result_sha256": DEVELOPMENT_RESULT_SHA256,
            "development_controls_sha256": DEVELOPMENT_CONTROLS_SHA256,
            "development_tree_indices": list(range(32)),
            "development_spearman_was_post_hoc": True,
            "development_selected_root_mean_gate_passed": False,
            "development_responses_reused": False,
            "confirmation_tree_block_was_contiguous": True,
            "confirmation_block_was_not_outcome_optimized": True,
            "confirmation_primary_endpoint": (
                "changed-root conditioning-benefit contrast to realized "
                "advantage Spearman correlation"
            ),
            "source_status_unchanged": True,
            "development_status_unchanged": True,
            "cannot_rescue_or_relabel_source": True,
            "confirmation_change": (
                "use disjoint source trees 32..63 and replace the failed "
                "selected-root mean success gate with the preregistered "
                "positive Spearman-bootstrap gate"
            ),
        }
    )
    result["second_draw_novelty"] = v3.second_draw_novelty(controls)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def replay_saved_controls(
    *,
    controls_path: Path,
    result_path: Path,
    bootstrap_samples: int = base.BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if sha256_file(controls_path) != result["protocol"]["controls_sha256"]:
        raise ValueError("confirmation controls hash changed")
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    with configured_base():
        source_result, source_trees, canonical_targets = (
            base.quality.load_and_validate_sources(
                result_path=base.SOURCE_RESULT,
                trees_path=base.SOURCE_TREES,
                targets_path=base.SOURCE_TARGETS,
            )
        )
        stop = SOURCE_TREE_START + base.TREE_COUNT
        public_trees = source_trees["trees"][SOURCE_TREE_START:stop]
        scored_source = source_result["trees"][SOURCE_TREE_START:stop]
        if controls.get("interface_version") != INTERFACE_VERSION:
            raise ValueError("confirmation controls interface changed")
        scored_rows = [
            base.score_tree(
                public_tree=public_tree,
                scored_tree=scored_tree,
                control_tree=control_tree,
                canonical_targets=canonical_targets,
            )
            for public_tree, scored_tree, control_tree in zip(
                public_trees,
                scored_source,
                controls["trees"],
                strict=True,
            )
        ]
        analysis = base.summarize_scores(
            scored_rows,
            bootstrap_samples=bootstrap_samples,
        )
    if analysis != result["analysis"]:
        raise ValueError("zero-call confirmation replay differs from result")
    if scored_rows != result["trees"]:
        raise ValueError("zero-call confirmation tree rows differ from result")
    return {"analysis": analysis, "trees": scored_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--smoke-result",
        type=Path,
        default=base.SMOKE_RESULT,
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    try:
        result = run_formal(
            output_dir=args.output_dir,
            run_id=args.run_id,
            smoke_result_path=args.smoke_result,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / "RUNNER_FAILURE.json"
        if not failure_path.exists():
            checkpoint(
                failure_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    print(json.dumps(result["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
