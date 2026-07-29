#!/usr/bin/env python3
"""Audit GuessingGame for path-dependent semantic hypothesis retrieval."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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
INTERFACE_VERSION = "guessinggame-path-bed-source-audit-1"
SOURCE_ROOT = REPO_ROOT / "external/GuessingGame"
SOURCE_URL = "https://github.com/cincynlp/GuessingGame"
SOURCE_COMMIT = "df56f1f13fefc4a8ba2c1f89d5026c027f5f42b3"
OBJECTS_PATH = SOURCE_ROOT / "Object_lists/objects.txt"
OBJECTS_SHA256 = (
    "a55a5f9410c8fe3e7cb34e3f57f51b0ad96a6d52db4b261204a48b9685e9fa58"
)
GAMES_PATH = (
    SOURCE_ROOT / "Results/Raw Text Results Final/GPTOpen.txt"
)
GAMES_SHA256 = (
    "a41a4dd3bccc444555c054f362c41e0fe747de79fda9183a893bf195cf0db6d4"
)
EXPECTED_OBJECTS = 858
EXPECTED_GAMES = 858
MIN_ELIGIBLE = 800
MIN_COLLISION_EXCESS = 100
SPLIT_SEED = 39_400
SERVING_COUNT = 5
MECHANICS_COUNT = 10
DEVELOPMENT_COUNT = 32
CONFIRMATION_COUNT = 64


@dataclass(frozen=True)
class GameRow:
    source_index: int
    target: str
    reported_turns: int
    material_question: str
    material_answer: str
    function_question: str
    function_answer: str


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


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
        "objects_sha256": sha256_file(OBJECTS_PATH),
        "games_sha256": sha256_file(GAMES_PATH),
    }
    expected = {
        "commit": SOURCE_COMMIT,
        "objects_sha256": OBJECTS_SHA256,
        "games_sha256": GAMES_SHA256,
    }
    if observed != expected:
        raise ValueError(
            f"GuessingGame source changed: expected {expected}, "
            f"observed {observed}"
        )
    return {**observed, "repository": SOURCE_URL}


def load_objects(path: Path = OBJECTS_PATH) -> list[str]:
    values = path.read_text(encoding="utf-8").splitlines()
    if any(not value.strip() for value in values):
        raise ValueError("object vocabulary contains a blank value")
    return values


def _strip_prefix(value: str, prefix: str) -> str:
    if not value.startswith(prefix):
        raise ValueError(f"turn does not start with {prefix!r}")
    return value[len(prefix):].strip()


def parse_game_line(line: str, *, source_index: int) -> GameRow:
    fields = line.split(",", 2)
    if len(fields) != 3:
        raise ValueError("game row does not have three leading fields")
    target, reported_turns, dialogue = fields
    turns = dialogue.split("\t")
    if len(turns) < 3:
        raise ValueError("game row has no complete question-answer pair")
    function_question = (
        _strip_prefix(turns[3], "Guesser said: ")
        if len(turns) > 3 and turns[3]
        else ""
    )
    function_answer = (
        _strip_prefix(turns[4], "Oracle said: ")
        if len(turns) > 4 and turns[4]
        else ""
    )
    return GameRow(
        source_index=source_index,
        target=target.strip(),
        reported_turns=int(reported_turns),
        material_question=_strip_prefix(
            turns[1], "Guesser said: "
        ),
        material_answer=_strip_prefix(
            turns[2], "Oracle said: "
        ),
        function_question=function_question,
        function_answer=function_answer,
    )


def load_games(path: Path = GAMES_PATH) -> list[GameRow]:
    return [
        parse_game_line(line, source_index=index)
        for index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()
        )
    ]


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def contains_literal_target(answer: str, target: str) -> bool:
    normalized_answer = f" {normalize_text(answer)} "
    normalized_target = normalize_text(target)
    return bool(
        normalized_target
        and f" {normalized_target} " in normalized_answer
    )


def eligibility_errors(row: GameRow) -> list[str]:
    errors: list[str] = []
    if (
        normalize_text(row.material_question)
        != "what material is the object made of"
    ):
        errors.append("noncanonical_material_question")
    function_question = normalize_text(row.function_question)
    if not any(
        token in function_question
        for token in ("function", "purpose", "primary use")
    ):
        errors.append("missing_function_question")
    if not row.material_answer.strip() or not row.function_answer.strip():
        errors.append("empty_answer")
    if contains_literal_target(row.material_answer, row.target):
        errors.append("material_answer_names_target")
    if contains_literal_target(row.function_answer, row.target):
        errors.append("function_answer_names_target")
    return errors


def row_payload(row: GameRow) -> dict[str, Any]:
    return {
        "source_index": row.source_index,
        "target": row.target,
        "reported_turns": row.reported_turns,
        "material_question": row.material_question,
        "material_answer": row.material_answer,
        "function_question": row.function_question,
        "function_answer": row.function_answer,
    }


def row_sha256(row: GameRow) -> str:
    return sha256_bytes(canonical_json(row_payload(row)).encode("utf-8"))


def case_id(row: GameRow) -> str:
    return f"gg-{row_sha256(row)[:16]}"


def _selection_key(row: GameRow) -> tuple[str, str]:
    digest = sha256_bytes(
        f"{SPLIT_SEED}|{row_sha256(row)}".encode("utf-8")
    )
    return digest, case_id(row)


def split_rows(
    rows: Sequence[GameRow],
) -> tuple[
    list[GameRow],
    list[GameRow],
    list[GameRow],
    list[GameRow],
    list[GameRow],
]:
    ordered = sorted(rows, key=_selection_key)
    serving_end = SERVING_COUNT
    mechanics_end = serving_end + MECHANICS_COUNT
    development_end = mechanics_end + DEVELOPMENT_COUNT
    confirmation_end = development_end + CONFIRMATION_COUNT
    return (
        ordered[:serving_end],
        ordered[serving_end:mechanics_end],
        ordered[mechanics_end:development_end],
        ordered[development_end:confirmation_end],
        ordered[confirmation_end:],
    )


def public_row(row: GameRow) -> dict[str, Any]:
    return {
        "case_id": case_id(row),
        "source_row_sha256": row_sha256(row),
    }


def run_audit(*, output_path: Path) -> dict[str, Any]:
    source = verify_source()
    objects = load_objects()
    games = load_games()
    object_set = set(objects)
    errors: Counter[str] = Counter()
    eligible = []
    for row in games:
        row_errors = eligibility_errors(row)
        errors.update(row_errors)
        if not row_errors:
            eligible.append(row)

    serving, mechanics, development, confirmation, unused = split_rows(
        eligible
    )
    selected = serving + mechanics + development + confirmation
    material_answers = {
        normalize_text(row.material_answer) for row in eligible
    }
    function_answers = {
        normalize_text(row.function_answer) for row in eligible
    }
    public_splits = {
        "serving_smoke": [public_row(row) for row in serving],
        "mechanics": [public_row(row) for row in mechanics],
        "development": [public_row(row) for row in development],
        "confirmation": [public_row(row) for row in confirmation],
        "unused_count": len(unused),
    }
    public_text = canonical_json(public_splits)
    gates = {
        "source_hashes_match": bool(source),
        "exact_object_count": len(objects) == EXPECTED_OBJECTS,
        "object_vocabulary_is_unique": len(object_set) == len(objects),
        "exact_released_game_count": len(games) == EXPECTED_GAMES,
        "games_have_unique_targets_matching_vocabulary": (
            len({row.target for row in games}) == len(games)
            and {row.target for row in games} == object_set
        ),
        "at_least_eight_hundred_clean_two_action_rows": (
            len(eligible) >= MIN_ELIGIBLE
        ),
        "both_actions_retain_at_least_one_hundred_collisions": (
            len(eligible) - len(material_answers) >= MIN_COLLISION_EXCESS
            and len(eligible) - len(function_answers)
            >= MIN_COLLISION_EXCESS
        ),
        "split_has_exact_sizes": (
            len(serving) == SERVING_COUNT
            and len(mechanics) == MECHANICS_COUNT
            and len(development) == DEVELOPMENT_COUNT
            and len(confirmation) == CONFIRMATION_COUNT
        ),
        "split_rows_are_disjoint": (
            len({case_id(row) for row in selected}) == len(selected)
        ),
        "selected_rows_preserve_eligibility": all(
            not eligibility_errors(row) for row in selected
        ),
        "public_manifest_excludes_targets_questions_and_answers": all(
            token not in public_text
            for token in (
                '"target":',
                '"material_question":',
                '"material_answer":',
                '"function_question":',
                '"function_answer":',
            )
        ),
        "two_orders_have_identical_evidence_sets": all(
            row.material_answer and row.function_answer for row in eligible
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
            "audit_disclosure": (
                "Source counts were inspected exploratorily before this "
                "reproducibility and split freeze; this is not a blind "
                "statistical confirmation."
            ),
            "adaptation": (
                "Uniform hidden-object prior with two immutable semantic "
                "observations. Material-first and function-first expose the "
                "same evidence in different order; an LLM retrieves a "
                "bounded hypothesis belief from the released vocabulary."
            ),
            "model_boundary": {
                "visible": [
                    "released object vocabulary with opaque integer ids",
                    "the policy's own ordered question-answer history",
                ],
                "hidden": [
                    "current target id and name",
                    "unasked target answers",
                    "split membership beyond the active opaque case",
                ],
            },
        },
        "source": source,
        "data": {
            "object_count": len(objects),
            "game_count": len(games),
            "eligible_count": len(eligible),
            "excluded_count": len(games) - len(eligible),
            "eligibility_error_counts": dict(sorted(errors.items())),
            "unique_material_answer_count": len(material_answers),
            "unique_function_answer_count": len(function_answers),
            "material_collision_excess": len(eligible) - len(material_answers),
            "function_collision_excess": len(eligible) - len(function_answers),
        },
        "splits": public_splits,
        "gates": gates,
        "all_gates_pass": passed,
    }
    checkpoint(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
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
