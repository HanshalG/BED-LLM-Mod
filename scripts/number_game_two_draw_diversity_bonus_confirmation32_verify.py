#!/usr/bin/env python3
"""Independently replay a completed diversity-bonus confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_two_draw_diversity_bonus_audit as audit
from scripts import number_game_two_draw_diversity_bonus_confirmation32 as confirmation
from scripts.number_game_ranking_fidelity_audit import spearman_correlation


SCHEMA_VERSION = 1
INTERFACE_VERSION = (
    "number-game-two-draw-diversity-bonus-confirmation32-verify-1"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    original_correlations = []
    bonus_correlations = []
    original_regrets = []
    bonus_regrets = []
    for row in rows:
        root_rows = row["root_rows"]
        roots = [int(item["root"]) for item in root_rows]
        predicted = {
            int(item["root"]): float(item["predicted_brier"])
            for item in root_rows
        }
        realized = {
            int(item["root"]): float(item["realized_brier"])
            for item in root_rows
        }
        adjusted = {
            int(root): float(value)
            for root, value in row["adjusted_scores"].items()
        }
        original_correlations.append(
            spearman_correlation(
                [predicted[root] for root in roots],
                [realized[root] for root in roots],
            )
        )
        bonus_correlations.append(
            spearman_correlation(
                [adjusted[root] for root in roots],
                [realized[root] for root in roots],
            )
        )
        oracle = min(realized.values())
        original_regrets.append(
            realized[int(row["original_root"])] - oracle
        )
        bonus_regrets.append(realized[int(row["bonus_root"])] - oracle)
    return {
        "original_mean_candidate_root_spearman": statistics.fmean(
            original_correlations
        ),
        "bonus_mean_candidate_root_spearman": statistics.fmean(
            bonus_correlations
        ),
        "original_mean_candidate_set_oracle_regret": statistics.fmean(
            original_regrets
        ),
        "bonus_mean_candidate_set_oracle_regret": statistics.fmean(
            bonus_regrets
        ),
    }


def replay_source(source_dir: Path) -> dict[str, Any]:
    spec = {
        "name": "prospective_confirmation32",
        "role": "prospective_confirmation",
        "directory": source_dir,
        "tree_count": confirmation.TREE_COUNT,
        "result_sha256": audit.sha256_file(source_dir / "RESULT.json"),
        "trees_sha256": audit.sha256_file(source_dir / "TREES.json"),
        "raw_sha256": audit.sha256_file(
            source_dir / "private" / "RAW_RESPONSES.json"
        ),
    }
    rows = audit.load_source(
        spec,
        coefficients=(0.0, audit.DIVERSITY_COEFFICIENT),
    )
    summary = audit.source_summary(
        rows,
        seed=confirmation.BOOTSTRAP_SEED,
        include_coefficient_grid=False,
    )
    public_rows = []
    for row in rows:
        public_row = dict(row)
        public_row.pop("coefficient_grid_roots")
        public_row["adjusted_scores"] = {
            str(root): float(value)
            for root, value in public_row["adjusted_scores"].items()
        }
        public_rows.append(public_row)
    return {
        "source_artifacts": {
            key: spec[key]
            for key in ("result_sha256", "trees_sha256", "raw_sha256")
        },
        "comparisons": summary["comparisons"],
        "rank_metrics": _rank_metrics(public_rows),
        "rows": public_rows,
    }


def verification_checks(
    *,
    result: dict[str, Any],
    source_result: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, bool]:
    protocol = result.get("protocol") or {}
    expected_gates = confirmation.scientific_gates(
        {"comparisons": replay["comparisons"]}
    )
    mechanics = source_result.get("mechanics_gates") or {}
    expected_status = (
        "passed"
        if mechanics
        and all(mechanics.values())
        and source_result.get("usage", {}).get("adapter_requests")
        == confirmation.EXPECTED_REQUESTS
        and all(expected_gates.values())
        else "gated_null"
    )
    rows = result.get("rows") or []
    source_protocol = source_result.get("protocol") or {}
    expected_validation_seeds = [
        list(
            range(
                confirmation.VALIDATION_SEED_START + tree_index * 16,
                confirmation.VALIDATION_SEED_START + (tree_index + 1) * 16,
            )
        )
        for tree_index in range(confirmation.TREE_COUNT)
    ]
    return {
        "interface_version_exact": (
            result.get("interface_version")
            == confirmation.INTERFACE_VERSION
        ),
        "preregistration_hash_matches": (
            protocol.get("preregistration_sha256")
            == confirmation.PREREGISTRATION_SHA256
            == audit.sha256_file(confirmation.PREREGISTRATION)
        ),
        "fresh_tree_seeds_exact": (
            protocol.get("tree_seeds") == list(confirmation.TREE_SEEDS)
        ),
        "actual_source_tree_seeds_exact": (
            [row.get("tree_seed") for row in replay["rows"]]
            == list(confirmation.TREE_SEEDS)
        ),
        "source_protocol_tree_seeds_exact": (
            source_protocol.get("tree_seeds")
            == list(confirmation.TREE_SEEDS)
        ),
        "fresh_target_seeds_exact": (
            protocol.get("target_seeds") == list(confirmation.TARGET_SEEDS)
        ),
        "source_protocol_target_seeds_exact": (
            source_protocol.get("target_seeds")
            == list(confirmation.TARGET_SEEDS)
        ),
        "validation_seed_exact": (
            protocol.get("validation_seed_start")
            == confirmation.VALIDATION_SEED_START
        ),
        "source_protocol_validation_seeds_exact": (
            source_protocol.get("validation_seeds")
            == expected_validation_seeds
        ),
        "bootstrap_protocol_exact": (
            protocol.get("bootstrap_seed") == confirmation.BOOTSTRAP_SEED
            and protocol.get("bootstrap_samples") == audit.BOOTSTRAP_SAMPLES
        ),
        "source_protocol_bootstrap_seed_exact": (
            source_protocol.get("bootstrap_seed")
            == confirmation.BOOTSTRAP_SEED
        ),
        "source_interface_exact": (
            source_protocol.get("interface_version")
            == confirmation.SOURCE_INTERFACE_VERSION
        ),
        "source_tree_and_validation_counts_exact": (
            source_protocol.get("tree_count") == confirmation.TREE_COUNT
            and source_protocol.get("validation_draws_per_tree") == 16
        ),
        "frozen_coefficient_exact": (
            protocol.get("diversity_coefficient")
            == audit.DIVERSITY_COEFFICIENT
        ),
        "no_fresh_coefficient_sweep_declared": (
            protocol.get("no_coefficient_sweep_on_fresh_data") is True
        ),
        "no_coefficient_grid_published": all(
            "coefficient_grid_roots" not in row for row in rows
        ),
        "source_artifact_hashes_match": (
            result.get("source_artifacts") == replay["source_artifacts"]
        ),
        "source_usage_embedded_exactly": (
            result.get("usage") == source_result.get("usage")
        ),
        "source_mechanics_embedded_exactly": all(
            result.get("mechanics_gates", {}).get(name) == passed
            for name, passed in mechanics.items()
        ),
        "accepted_request_count_exact": (
            source_result.get("usage", {}).get("adapter_requests")
            == confirmation.EXPECTED_REQUESTS
            == protocol.get("accepted_requests_expected")
        ),
        "all_fixed_selector_rows_replay_exactly": rows == replay["rows"],
        "all_comparisons_and_bootstraps_replay_exactly": (
            result.get("comparisons") == replay["comparisons"]
        ),
        "rank_metrics_replay_exactly": (
            result.get("rank_metrics") == replay["rank_metrics"]
        ),
        "scientific_gates_replay_exactly": (
            result.get("scientific_gates") == expected_gates
        ),
        "final_status_replays_exactly": result.get("status") == expected_status,
    }


def verify_completed_confirmation(*, run_dir: Path) -> dict[str, Any]:
    result = _load(run_dir / "RESULT.json")
    source_dir = run_dir / "source"
    source_result = _load(source_dir / "RESULT.json")
    replay = replay_source(source_dir)
    checks = verification_checks(
        result=result,
        source_result=source_result,
        replay=replay,
    )
    failed = [name for name, passed in checks.items() if not passed]
    verification = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "verified" if not failed else "verification_failed",
        "model_calls": 0,
        "cost_usd": 0.0,
        "checks": checks,
        "failed_checks": failed,
        "result_sha256": audit.sha256_file(run_dir / "RESULT.json"),
        "source_artifacts": replay["source_artifacts"],
    }
    checkpoint(run_dir / "VERIFICATION.json", verification)
    if failed:
        raise ValueError(
            "diversity-bonus confirmation verification failed: "
            + ", ".join(failed)
        )
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    verification = verify_completed_confirmation(run_dir=args.run_dir)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
