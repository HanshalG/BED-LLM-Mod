#!/usr/bin/env python3
"""Audit and split GuessWhat?! for a visual-semantic BED experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "guesswhat-semantic-bed-source-audit-1"
SOURCE_ROOT = REPO_ROOT / "external/guesswhat"
SOURCE_COMMIT = "346b7de65d5f18fb8c7d357b7c743d02be429d8a"
DATA_PATH = SOURCE_ROOT / "data/guesswhat.test.jsonl.gz"
DATA_SHA256 = "c26c08fbb860786f25ab6940dab135c4f61a6404d0c89bbf5b7a21716306548c"
SOURCE_URL = "https://github.com/GuessWhatGame/guesswhat"
DATA_URL = "https://florian-strub.com/guesswhat.test.jsonl.gz"

EXPECTED_ROWS = 23_115
MIN_ELIGIBLE_UNIQUE_IMAGES = 5_000
MIN_OBJECTS = 5
MAX_OBJECTS = 12
MIN_HUMAN_QUESTIONS = 4
SPLIT_SEED = 39_000
SERVING_COUNT = 2
DEVELOPMENT_COUNT = 20
HOLDOUT_COUNT = 60
COCO_FILE_RE = re.compile(r"^COCO_(train|val)2014_[0-9]{12}\.jpg$")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str:
    return str(row["dialogue_id"])


def image_id(row: dict[str, Any]) -> str:
    return str(row["picture_id"])


def coco_image_url(row: dict[str, Any]) -> str:
    filename = row["picture"]["file_name"]
    split = "train2014" if "_train2014_" in filename else "val2014"
    return f"http://images.cocodataset.org/{split}/{filename}"


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_source() -> dict[str, Any]:
    observed = {
        "commit": _git_value("rev-parse", "HEAD"),
        "data_sha256": sha256_file(DATA_PATH),
    }
    expected = {
        "commit": SOURCE_COMMIT,
        "data_sha256": DATA_SHA256,
    }
    if observed != expected:
        raise ValueError(
            f"GuessWhat source changed: expected {expected}, observed {observed}"
        )
    return {
        **observed,
        "repository": SOURCE_URL,
        "data_url": DATA_URL,
    }


def load_rows(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} is not an object")
            rows.append(value)
    return rows


def eligibility_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "status",
        "picture",
        "picture_id",
        "qas",
        "object_id",
        "dialogue_id",
        "objects",
    }
    if not required.issubset(row):
        return ["missing_required_field"]
    if row["status"] != "success":
        errors.append("not_successful")
    if not isinstance(row["picture"], dict):
        return errors + ["invalid_picture"]
    filename = row["picture"].get("file_name")
    if not isinstance(filename, str) or COCO_FILE_RE.fullmatch(filename) is None:
        errors.append("invalid_coco_filename")
    if not isinstance(row["qas"], list) or len(row["qas"]) < MIN_HUMAN_QUESTIONS:
        errors.append("insufficient_human_questions")
    elif not all(
        isinstance(qa, dict)
        and isinstance(qa.get("q"), str)
        and qa["q"].strip()
        and qa.get("a") in {"Yes", "No", "N/A"}
        for qa in row["qas"]
    ):
        errors.append("invalid_human_qa")

    objects = row["objects"]
    if not isinstance(objects, dict):
        return errors + ["invalid_objects"]
    if not MIN_OBJECTS <= len(objects) <= MAX_OBJECTS:
        errors.append("object_count_outside_range")
    target = objects.get(str(row["object_id"]))
    if not isinstance(target, dict):
        errors.append("target_missing")
    elif target.get("iscrowd") is not False:
        errors.append("target_is_crowd")

    valid_objects = True
    categories: Counter[str] = Counter()
    for key, obj in objects.items():
        if (
            not isinstance(key, str)
            or not isinstance(obj, dict)
            or str(obj.get("object_id")) != key
            or not isinstance(obj.get("category"), str)
            or not obj["category"].strip()
            or not isinstance(obj.get("bbox"), list)
            or len(obj["bbox"]) != 4
            or not all(
                isinstance(value, (int, float)) and value >= 0
                for value in obj["bbox"]
            )
        ):
            valid_objects = False
            continue
        categories[obj["category"]] += 1
    if not valid_objects:
        errors.append("invalid_object_annotation")
    if not any(count >= 2 for count in categories.values()):
        errors.append("no_same_category_ambiguity")
    return errors


def _selection_key(row: dict[str, Any]) -> tuple[str, str]:
    digest = hashlib.sha256(
        (
            f"{SPLIT_SEED}|{image_id(row)}|{row_id(row)}|{row_sha256(row)}"
        ).encode("utf-8")
    ).hexdigest()
    return digest, row_id(row)


def dedupe_by_image(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=_selection_key):
        selected.setdefault(image_id(row), row)
    return sorted(selected.values(), key=_selection_key)


def split_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    ordered = dedupe_by_image(rows)
    serving_end = SERVING_COUNT
    development_end = serving_end + DEVELOPMENT_COUNT
    holdout_end = development_end + HOLDOUT_COUNT
    return (
        ordered[:serving_end],
        ordered[serving_end:development_end],
        ordered[development_end:holdout_end],
        ordered[holdout_end:],
    )


def candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    objects = []
    for index, obj in enumerate(row["objects"].values(), start=1):
        objects.append(
            {
                "candidate_index": index,
                "object_id": str(obj["object_id"]),
                "bbox": list(obj["bbox"]),
            }
        )
    return {
        "dialogue_id": row_id(row),
        "picture_id": image_id(row),
        "image_file_name": row["picture"]["file_name"],
        "image_width": int(row["picture"]["width"]),
        "image_height": int(row["picture"]["height"]),
        "candidate_objects": objects,
    }


def hidden_state_boundary_holds(row: dict[str, Any]) -> bool:
    payload = candidate_payload(row)
    serialized = canonical_json(payload)
    return (
        set(payload)
        == {
            "dialogue_id",
            "picture_id",
            "image_file_name",
            "image_width",
            "image_height",
            "candidate_objects",
        }
        and str(row["object_id"]) not in {
            str(payload.get("target_object_id", "")),
        }
        and '"qas":' not in serialized
        and '"status":' not in serialized
        and '"target_object_id":' not in serialized
        and all("category" not in obj for obj in payload["candidate_objects"])
    )


def selected_public_row(row: dict[str, Any]) -> dict[str, Any]:
    category_counts = Counter(
        obj["category"] for obj in row["objects"].values()
    )
    return {
        "dialogue_id": row_id(row),
        "picture_id": image_id(row),
        "row_sha256": row_sha256(row),
        "image_file_name": row["picture"]["file_name"],
        "image_url": coco_image_url(row),
        "object_count": len(row["objects"]),
        "same_category_group_count": sum(
            count >= 2 for count in category_counts.values()
        ),
        "human_question_count": len(row["qas"]),
    }


def run_audit(*, output_path: Path) -> dict[str, Any]:
    source = verify_source()
    rows = load_rows()
    target_present_count = sum(
        isinstance(row.get("objects"), dict)
        and str(row.get("object_id")) in row["objects"]
        for row in rows
    )
    errors: Counter[str] = Counter()
    eligible = []
    for row in rows:
        row_errors = eligibility_errors(row)
        errors.update(row_errors)
        if not row_errors:
            eligible.append(row)
    unique_eligible = dedupe_by_image(eligible)
    serving, development, holdout, unused = split_rows(eligible)
    selected = serving + development + holdout
    selected_dialogues = [row_id(row) for row in selected]
    selected_images = [image_id(row) for row in selected]
    public_selected = [selected_public_row(row) for row in selected]
    public_text = canonical_json(public_selected)

    gates = {
        "source_hashes_match": bool(source),
        "exact_released_row_count": len(rows) == EXPECTED_ROWS,
        "all_released_targets_are_annotated_objects": (
            target_present_count == len(rows)
        ),
        "at_least_five_thousand_eligible_unique_images": (
            len(unique_eligible) >= MIN_ELIGIBLE_UNIQUE_IMAGES
        ),
        "split_has_exact_sizes": (
            len(serving) == SERVING_COUNT
            and len(development) == DEVELOPMENT_COUNT
            and len(holdout) == HOLDOUT_COUNT
        ),
        "split_dialogues_and_images_are_disjoint": (
            len(selected_dialogues) == len(set(selected_dialogues))
            and len(selected_images) == len(set(selected_images))
        ),
        "selected_rows_preserve_eligibility": all(
            not eligibility_errors(row) for row in selected
        ),
        "candidate_payload_hides_target_and_human_answers": all(
            hidden_state_boundary_holds(row) for row in selected
        ),
        "public_manifest_excludes_target_and_human_answers": (
            '"object_id":' not in public_text
            and '"qas":' not in public_text
            and '"target":' not in public_text
        ),
    }
    passed = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model_calls": 0,
            "cost_usd": 0.0,
            "split_seed": SPLIT_SEED,
            "adaptation": (
                "Visual-semantic Bayesian object identification using the "
                "released hidden target and annotated candidate objects."
            ),
            "candidate_observation": {
                "visible": [
                    "source image",
                    "numbered candidate bounding boxes",
                    "past policy questions and realized answers",
                ],
                "hidden": [
                    "released target object id",
                    "released human question-answer dialogue",
                    "released game outcome",
                    "candidate object category annotations",
                ],
            },
        },
        "source": source,
        "data": {
            "total_rows": len(rows),
            "target_present_count": target_present_count,
            "eligible_rows": len(eligible),
            "eligible_unique_images": len(unique_eligible),
            "eligibility_error_counts": dict(sorted(errors.items())),
            "eligible_object_count_distribution": dict(
                sorted(Counter(len(row["objects"]) for row in eligible).items())
            ),
            "eligible_human_question_count_distribution": dict(
                sorted(Counter(len(row["qas"]) for row in eligible).items())
            ),
        },
        "splits": {
            "serving_smoke": [selected_public_row(row) for row in serving],
            "development": [
                selected_public_row(row) for row in development
            ],
            "holdout": [selected_public_row(row) for row in holdout],
            "unused_count": len(unused),
        },
        "gates": gates,
        "all_gates_pass": passed,
    }
    checkpoint(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=(
            REPO_ROOT
            / "results/nonmyopic/guesswhat_semantic_bed_source_audit/"
            "guesswhat-semantic-bed-source-audit-20260729/MANIFEST.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(output_path=args.output_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "data": result["data"],
                "splits": {
                    key: len(value) if isinstance(value, list) else value
                    for key, value in result["splits"].items()
                },
                "gates": result["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
