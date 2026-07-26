#!/usr/bin/env python3
"""Run the fresh-split InfoQuest opportunity audit with missingness caps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Sequence

from scripts import infoquest_cached_trajectory_opportunity as v1
from scripts import infoquest_llm_bed_manifest as source_manifest


INTERFACE_VERSION = "infoquest-cached-trajectory-opportunity-2"
SELECTION_SEED = 24_417
SPLIT_SIZES = {
    "opportunity": 80,
    "development": 30,
}
EXPECTED_SPLIT_HASHES = {
    "opportunity": (
        "15a85bd67dc68862b4c77a1ac2ec40f3b921a26ce2362cf16190c7966291c9c1"
    ),
    "development": (
        "7ec12e636bf8e083638a15b9b1ec3ea05087e99e54fd0107e178a28baa1e1642"
    ),
    "holdout": (
        "f99de5f2f33c9499517ac31057e19962817b9ffed217e17d17461eda7028083b"
    ),
}
EXPECTED_COMBINED_SPLITS_HASH = (
    "acc65f8de579b3f4c5103aed997574769d6338ad367ee60203ad834951d8f4a6"
)
MAX_EMPTY_LATER_MESSAGE_FRACTION = 0.005
MAX_AFFECTED_TRAJECTORY_FRACTION = 0.02


def split_record_ids_v2(manifest: dict[str, Any]) -> dict[str, list[int]]:
    original_holdout = manifest["selection"]["splits"]["holdout"]
    if (
        original_holdout["ordered_sha256"]
        != source_manifest.EXPECTED_SPLIT_HASHES["holdout"]
        or source_manifest.canonical_sha256(original_holdout["record_ids"])
        != original_holdout["ordered_sha256"]
    ):
        raise ValueError("original InfoQuest holdout split changed")
    candidates = sorted(
        set(original_holdout["record_ids"]) - v1.QUARANTINED_IDS
    )
    if len(candidates) != 387:
        raise ValueError("effective InfoQuest holdout size changed")
    random.Random(SELECTION_SEED).shuffle(candidates)
    opportunity_end = SPLIT_SIZES["opportunity"]
    development_end = opportunity_end + SPLIT_SIZES["development"]
    splits = {
        "opportunity": candidates[:opportunity_end],
        "development": candidates[opportunity_end:development_end],
        "holdout": candidates[development_end:],
    }
    hashes = {
        name: source_manifest.canonical_sha256(record_ids)
        for name, record_ids in splits.items()
    }
    if hashes != EXPECTED_SPLIT_HASHES:
        raise ValueError("InfoQuest V2 split hashes changed")
    if (
        source_manifest.canonical_sha256(splits)
        != EXPECTED_COMBINED_SPLITS_HASH
    ):
        raise ValueError("InfoQuest V2 combined split hash changed")
    return splits


def build_audit(
    *,
    manifest_path: Path,
    settings_path: Path,
    trajectory_paths: Sequence[Path],
) -> dict[str, Any]:
    if len(trajectory_paths) != 3:
        raise ValueError("exactly three trajectory files are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("interface_version")
        != "infoquest-llm-bed-manifest-1"
        or manifest["source"]["revision"] != source_manifest.SOURCE_REVISION
        or manifest["selection"]["combined_splits_sha256"]
        != source_manifest.EXPECTED_COMBINED_SPLITS_HASH
    ):
        raise ValueError("InfoQuest manifest does not match the frozen source")
    splits = split_record_ids_v2(manifest)
    opportunity_ids = splits["opportunity"]

    settings_rows = v1._rows_by_id(
        settings_path,
        expected_sha256=source_manifest.SOURCE_SHA256["settings"],
    )
    runs = [
        v1._rows_by_id(path, expected_sha256=expected_hash)
        for path, expected_hash in zip(
            trajectory_paths,
            v1.BASELINE_SHA256,
        )
    ]

    trajectories = []
    for run, rows_by_id in enumerate(runs):
        for record_id in opportunity_ids:
            setting = settings_rows[record_id]
            if setting["id"] != record_id:
                raise ValueError("InfoQuest setting id changed")
            seed_message = setting["seed_message"]
            if not isinstance(seed_message, str) or not seed_message.strip():
                raise ValueError("InfoQuest seed message is empty")
            for world in (1, 2):
                trajectories.append(
                    v1.trajectory_metrics(
                        rows_by_id[record_id],
                        seed_message=seed_message,
                        run=run,
                        record_id=record_id,
                        world=world,
                        allow_empty_later=True,
                    )
                )

    metrics, scientific_gates = v1.summarize_metrics(trajectories)
    later_messages = sum(row["later_message_count"] for row in trajectories)
    empty_messages = sum(
        row["empty_later_message_count"] for row in trajectories
    )
    affected_trajectories = sum(
        row["empty_later_message_count"] > 0 for row in trajectories
    )
    metrics.update(
        {
            "later_messages": later_messages,
            "empty_later_messages": empty_messages,
            "empty_later_message_fraction": empty_messages / later_messages,
            "trajectories_with_empty_later_message": affected_trajectories,
            "affected_trajectory_fraction": (
                affected_trajectories / len(trajectories)
            ),
        }
    )
    missingness_gates = {
        "empty_later_message_fraction_within_cap": (
            metrics["empty_later_message_fraction"]
            <= MAX_EMPTY_LATER_MESSAGE_FRACTION
        ),
        "affected_trajectory_fraction_within_cap": (
            metrics["affected_trajectory_fraction"]
            <= MAX_AFFECTED_TRAJECTORY_FRACTION
        ),
    }
    structural_gates = {
        "fresh_v2_split_hashes_match": True,
        "v2_split_drawn_only_from_effective_original_holdout": True,
        "quarantined_and_v1_opportunity_ids_absent": (
            not (
                set(opportunity_ids)
                & (
                    v1.QUARANTINED_IDS
                    | set(
                        manifest["selection"]["splits"]["opportunity"][
                            "record_ids"
                        ]
                    )
                )
            )
        ),
        "exactly_three_hash_pinned_runs": True,
        "each_run_has_exact_ids_0_through_499": True,
        "exactly_480_opportunity_trajectories": (
            len(trajectories) == 3 * 80 * 2
        ),
        "all_histories_and_evaluations_validate": True,
        "all_reward_traces_monotone": True,
        "empty_later_messages_retained_as_zero_information": True,
        "semantic_content_not_emitted": True,
        "openrouter_calls_zero": True,
        "oatml_jobs_zero": True,
    }
    gates = {
        **structural_gates,
        **missingness_gates,
        **scientific_gates,
    }
    return {
        "interface_version": INTERFACE_VERSION,
        "source": {
            "repository": source_manifest.SOURCE_REPOSITORY,
            "revision": source_manifest.SOURCE_REVISION,
            "manifest_sha256": source_manifest.sha256_file(manifest_path),
            "settings_sha256": source_manifest.SOURCE_SHA256["settings"],
            "trajectory_files": [
                {
                    "filename": filename,
                    "sha256": digest,
                }
                for filename, digest in zip(
                    v1.BASELINE_FILENAMES,
                    v1.BASELINE_SHA256,
                )
            ],
        },
        "selection": {
            "seed": SELECTION_SEED,
            "source_pool": "original effective holdout",
            "combined_splits_sha256": EXPECTED_COMBINED_SPLITS_HASH,
            "splits": {
                name: {
                    "record_ids": record_ids,
                    "records": len(record_ids),
                    "ordered_sha256": EXPECTED_SPLIT_HASHES[name],
                }
                for name, record_ids in splits.items()
            },
            "v1_opportunity_reused": False,
            "quarantined_ids": sorted(v1.QUARANTINED_IDS),
        },
        "thresholds": {
            **v1.THRESHOLDS,
            "maximum_empty_later_message_fraction": (
                MAX_EMPTY_LATER_MESSAGE_FRACTION
            ),
            "maximum_affected_trajectory_fraction": (
                MAX_AFFECTED_TRAJECTORY_FRACTION
            ),
        },
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "trajectories": trajectories,
        "semantic_content_emitted": False,
        "seed_message_content_emitted": False,
        "hidden_setting_content_emitted": False,
        "message_content_emitted": False,
        "checklist_content_emitted": False,
        "causal_policy_efficacy_claimed": False,
        "development_read": False,
        "holdout_read": False,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        required=True,
        help="Pass exactly three times in run-index order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_audit(
        manifest_path=args.manifest,
        settings_path=args.settings,
        trajectory_paths=args.trajectory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
