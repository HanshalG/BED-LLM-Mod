#!/usr/bin/env python3
"""Audit and split CUPID for an active contextual-preference BED task."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "cupid-active-preference-source-audit-1"
SOURCE_ROOT = REPO_ROOT / "external/CUPID"
SOURCE_COMMIT = "a8560cab293ae98be4fe260689d58bddf96b51ef"
SOURCE_TREE = "f035180dd7cc963ef0222e0d25a8c2e4d8d1d9c5"
DATASET_REPOSITORY = "kixlab/CUPID"
DATASET_REVISION = "f6e5fdae9b31f2b400d6ceb281a6a6760cc00309"
DATA_PATH = SOURCE_ROOT / "data/test.parquet"
DATA_SHA256 = "6d68af09f7fbe52df0a3bd621604104696d0f481f166be5c06dd65bbb089aaae"
FORMATTER_PATH = SOURCE_ROOT / "utils/formatting.py"
FORMATTER_SHA256 = "ab9597c8c0f506bc9a25fe8061f3d9b474b20c1ffaa3880ad95a1db5ac51b2e2"
EVALUATOR_PATH = SOURCE_ROOT / "evaluation/pipeline/evaluate.py"
EVALUATOR_SHA256 = "eaead752332b51761f292c5c88f7279890d8718af2ca0ce75647f63c05b36b91"
INFERRER_PATH = SOURCE_ROOT / "evaluation/modules/preference_inferrer.py"
INFERRER_SHA256 = "35187639331dd458a1cddd26b9981e56deb36d74371555e122222bc6eaa28138"
INFERRER_PROMPT_PATH = SOURCE_ROOT / "prompts/evaluation/preference_inferrer.yaml"
INFERRER_PROMPT_SHA256 = "b20374118413bd228b67a6fc3ac210766ff2415f8c3014cc61b6b432e83be472"

INSTANCE_TYPES = ("consistent", "contrastive", "changing")
SPLIT_SEED = 37400
SERVING_COUNTS = {
    "consistent": 1,
    "contrastive": 2,
    "changing": 2,
}
DEVELOPMENT_PER_TYPE = 5
HOLDOUT_PER_TYPE = 20
PRIOR_INTERACTIONS_EXPOSED = 2
EXPECTED_TOTAL_ROWS = 756
EXPECTED_ROWS_PER_TYPE = 252
EXPECTED_PRIOR_INTERACTIONS = 8


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def row_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str:
    return f"{row['persona_id']}:{row['instance_type']}"


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
        "tree": _git_value("rev-parse", "HEAD^{tree}"),
        "data_sha256": sha256_file(DATA_PATH),
        "formatter_sha256": sha256_file(FORMATTER_PATH),
        "evaluator_sha256": sha256_file(EVALUATOR_PATH),
        "inferrer_sha256": sha256_file(INFERRER_PATH),
        "inferrer_prompt_sha256": sha256_file(INFERRER_PROMPT_PATH),
    }
    expected = {
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "data_sha256": DATA_SHA256,
        "formatter_sha256": FORMATTER_SHA256,
        "evaluator_sha256": EVALUATOR_SHA256,
        "inferrer_sha256": INFERRER_SHA256,
        "inferrer_prompt_sha256": INFERRER_PROMPT_SHA256,
    }
    if observed != expected:
        raise ValueError(
            f"CUPID source changed: expected {expected}, observed {observed}"
        )
    return {
        **observed,
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
    }


def load_rows(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the CUPID parquet") from exc
    rows = parquet.read_table(path).to_pylist()
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("CUPID parquet contains a non-object row")
    return rows


def _valid_dialogue(dialogue: Any) -> bool:
    if not isinstance(dialogue, list) or not dialogue:
        return False
    for message in dialogue:
        if not isinstance(message, dict):
            return False
        if set(message) != {"content", "role"}:
            return False
        if message["role"] not in {"user", "assistant"}:
            return False
        if not isinstance(message["content"], str) or not message["content"].strip():
            return False
    return True


def eligibility_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    string_fields = (
        "persona_id",
        "instance_type",
        "current_request",
        "current_context_factor",
        "current_contextual_preference",
    )
    for field in string_fields:
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"invalid_{field}")
    if errors:
        return errors
    if row["instance_type"] not in INSTANCE_TYPES:
        errors.append("invalid_instance_type")

    checklist = row.get("current_checklist")
    if (
        not isinstance(checklist, list)
        or not checklist
        or not all(isinstance(item, str) and item.strip() for item in checklist)
    ):
        errors.append("invalid_current_checklist")

    interactions = row.get("prior_interactions")
    if not isinstance(interactions, list):
        errors.append("invalid_prior_interactions")
        return errors
    if len(interactions) != EXPECTED_PRIOR_INTERACTIONS:
        errors.append("prior_interaction_count")

    valid_interactions = True
    for interaction in interactions:
        if not isinstance(interaction, dict):
            valid_interactions = False
            continue
        if set(interaction) != {
            "context_factor",
            "contextual_preference",
            "dialogue",
        }:
            valid_interactions = False
            continue
        if (
            not isinstance(interaction["context_factor"], str)
            or not interaction["context_factor"].strip()
            or not isinstance(interaction["contextual_preference"], str)
            or not interaction["contextual_preference"].strip()
            or not _valid_dialogue(interaction["dialogue"])
        ):
            valid_interactions = False
    if not valid_interactions:
        errors.append("invalid_prior_interaction")

    valid_rows = [item for item in interactions if isinstance(item, dict)]
    matching_count = sum(
        item.get("context_factor") == row["current_context_factor"]
        for item in valid_rows
    )
    nonmatching_count = sum(
        item.get("context_factor") != row["current_context_factor"]
        for item in valid_rows
    )
    if matching_count < 1:
        errors.append("missing_matching_context")
    if nonmatching_count < PRIOR_INTERACTIONS_EXPOSED:
        errors.append("insufficient_nonmatching_contexts")
    return errors


def candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    background = []
    for interaction in row["prior_interactions"]:
        if interaction["context_factor"] == row["current_context_factor"]:
            continue
        background.append(
            {
                "context_factor": interaction["context_factor"],
                "dialogue": [
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                    for message in interaction["dialogue"]
                ],
            }
        )
        if len(background) == PRIOR_INTERACTIONS_EXPOSED:
            break
    if len(background) != PRIOR_INTERACTIONS_EXPOSED:
        raise ValueError(
            f"{row_id(row)} lacks {PRIOR_INTERACTIONS_EXPOSED} background sessions"
        )
    return {
        "current_request": row["current_request"],
        "current_context_factor": row["current_context_factor"],
        "background_interactions": background,
    }


def hidden_state_boundary_holds(row: dict[str, Any]) -> bool:
    payload = candidate_payload(row)
    if set(payload) != {
        "current_request",
        "current_context_factor",
        "background_interactions",
    }:
        return False
    for interaction in payload["background_interactions"]:
        if set(interaction) != {"context_factor", "dialogue"}:
            return False
        if interaction["context_factor"] == row["current_context_factor"]:
            return False
        if not _valid_dialogue(interaction["dialogue"]):
            return False
    return all(
        forbidden not in canonical_json(payload)
        for forbidden in (
            '"current_contextual_preference":',
            '"current_checklist":',
            '"contextual_preference":',
        )
    )


def _split_key(row: dict[str, Any]) -> tuple[str, str]:
    digest = hashlib.sha256(
        (
            f"{SPLIT_SEED}|{row['instance_type']}|"
            f"{row['persona_id']}|{row_sha256(row)}"
        ).encode("utf-8")
    ).hexdigest()
    return digest, row_id(row)


def split_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    serving: list[dict[str, Any]] = []
    development: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    unused: list[dict[str, Any]] = []
    for instance_type in INSTANCE_TYPES:
        ordered = sorted(
            (
                row
                for row in rows
                if row["instance_type"] == instance_type
            ),
            key=_split_key,
        )
        serving_count = SERVING_COUNTS[instance_type]
        development_end = serving_count + DEVELOPMENT_PER_TYPE
        holdout_end = development_end + HOLDOUT_PER_TYPE
        serving.extend(ordered[:serving_count])
        development.extend(ordered[serving_count:development_end])
        holdout.extend(ordered[development_end:holdout_end])
        unused.extend(ordered[holdout_end:])
    return serving, development, holdout, unused


def selected_public_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = candidate_payload(row)
    return {
        "id": row_id(row),
        "row_sha256": row_sha256(row),
        "instance_type": row["instance_type"],
        "prior_interaction_count": len(row["prior_interactions"]),
        "exposed_background_count": len(payload["background_interactions"]),
        "checklist_item_count": len(row["current_checklist"]),
    }


def official_formatter_control_flow() -> dict[str, bool]:
    spec = importlib.util.spec_from_file_location(
        "cupid_source_formatter",
        FORMATTER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load CUPID formatter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    synthetic = [
        {
            "context_factor": "HIDDEN_CONTEXT_SENTINEL",
            "contextual_preference": "HIDDEN_PREFERENCE_SENTINEL",
            "dialogue": [
                {"role": "user", "content": "VISIBLE_USER_SENTINEL"},
                {"role": "assistant", "content": "VISIBLE_ASSISTANT_SENTINEL"},
            ],
        }
    ]
    formatted = module.format_interaction_log(synthetic)
    return {
        "official_formatter_includes_user_dialogue": (
            "VISIBLE_USER_SENTINEL" in formatted
        ),
        "official_formatter_includes_assistant_dialogue": (
            "VISIBLE_ASSISTANT_SENTINEL" in formatted
        ),
        "official_formatter_excludes_context_metadata": (
            "HIDDEN_CONTEXT_SENTINEL" not in formatted
        ),
        "official_formatter_excludes_preference_metadata": (
            "HIDDEN_PREFERENCE_SENTINEL" not in formatted
        ),
    }


def run_audit(*, output_path: Path) -> dict[str, Any]:
    source = verify_source()
    rows = load_rows()
    errors = Counter()
    eligible = []
    for row in rows:
        row_errors = eligibility_errors(row)
        errors.update(row_errors)
        if not row_errors:
            eligible.append(row)

    ids = [row_id(row) for row in rows if "persona_id" in row and "instance_type" in row]
    serving, development, holdout, unused = split_rows(eligible)
    selected = serving + development + holdout
    selected_ids = [row_id(row) for row in selected]
    type_counts = Counter(row.get("instance_type") for row in rows)
    formatter_control_flow = official_formatter_control_flow()

    expected_serving = sum(SERVING_COUNTS.values())
    expected_development = DEVELOPMENT_PER_TYPE * len(INSTANCE_TYPES)
    expected_holdout = HOLDOUT_PER_TYPE * len(INSTANCE_TYPES)
    gates = {
        "source_hashes_match": bool(source),
        "exact_released_row_count": len(rows) == EXPECTED_TOTAL_ROWS,
        "instance_types_balanced": all(
            type_counts[instance_type] == EXPECTED_ROWS_PER_TYPE
            for instance_type in INSTANCE_TYPES
        ),
        "all_rows_eligible": len(eligible) == len(rows) and not errors,
        "source_row_ids_unique": len(ids) == len(rows) == len(set(ids)),
        "split_has_exact_sizes_and_unique_ids": (
            len(serving) == expected_serving
            and len(development) == expected_development
            and len(holdout) == expected_holdout
            and len(selected_ids) == len(set(selected_ids))
        ),
        "selected_rows_preserve_source_structure": all(
            not eligibility_errors(row) for row in selected
        ),
        "candidate_payload_respects_hidden_state_boundary": all(
            hidden_state_boundary_holds(row) for row in selected
        ),
        "official_formatter_exposes_dialogue_only": all(
            formatter_control_flow.values()
        ),
    }
    all_gates_pass = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model_calls": 0,
            "cost_usd": 0.0,
            "split_seed": SPLIT_SEED,
            "adaptation": (
                "Active contextual-preference interview; this is a released-task "
                "adaptation and not the official CUPID evaluation protocol."
            ),
            "candidate_observation": {
                "visible": [
                    "current_request",
                    "current_context_factor",
                    "two earliest nonmatching-context prior dialogues",
                ],
                "hidden": [
                    "current_contextual_preference",
                    "current_checklist",
                    "all prior contextual_preference metadata",
                    "same-context prior dialogue",
                    "remaining prior dialogues",
                ],
            },
        },
        "status": "pass" if all_gates_pass else "fail",
        "source": source,
        "data": {
            "total_rows": len(rows),
            "instance_type_counts": dict(sorted(type_counts.items())),
            "eligible_rows": len(eligible),
            "eligibility_error_counts": dict(sorted(errors.items())),
            "prior_interaction_count_distribution": dict(
                sorted(
                    Counter(
                        len(row.get("prior_interactions", [])) for row in rows
                    ).items()
                )
            ),
            "checklist_item_count_distribution": dict(
                sorted(
                    Counter(
                        len(row.get("current_checklist", [])) for row in rows
                    ).items()
                )
            ),
        },
        "splits": {
            "serving_smoke": [selected_public_row(row) for row in serving],
            "development": [selected_public_row(row) for row in development],
            "holdout": [selected_public_row(row) for row in holdout],
            "unused_count": len(unused),
        },
        "official_formatter_control_flow": formatter_control_flow,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
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
            / "results/nonmyopic/cupid_active_preference_source_audit/"
            "cupid-active-preference-source-audit-20260729/MANIFEST.json"
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
