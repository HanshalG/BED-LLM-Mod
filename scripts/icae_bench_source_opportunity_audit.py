#!/usr/bin/env python3
"""Audit the frozen ICAE-Bench mechanics split without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXPECTED_MANIFEST_SHA256 = (
    "47ab7f2fcf985433389f53873fa7fe8ccbebd88dd8390d654de93083bad84f5f"
)
EXPECTED_COMMIT = "66bbabb20a2138d066ac7d6f7ba6768b57c2f79b"
EXPECTED_TREE = "0086e71719a176a959c34c8db78879996ca8dfc1"
EXPECTED_ORACLE_CORE_SHA256 = (
    "053a3b3137e604491afb531af86579d276961eb523b0b11bd55cd93ac06f9227"
)
EXPECTED_ORACLE_PROMPT_SHA256 = (
    "316ab5f4175a6b22ab9c56edf60745c12a9d13c1dc597bd82b918f275f4a5047"
)
EXPECTED_ORACLE_SERVER_SHA256 = (
    "7d4fb8efad4234fb4870cf1212b4f05d3708d739eda4ec412fce61684a2e2026"
)
EXPECTED_TEST_ARCHIVE_SHA256 = (
    "f746c560e724d0dad12d512ead771d91de91d4b3e6051335728c305e2adff204"
)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "must",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "up",
    "use",
    "used",
    "using",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def content_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_+-]{2,}", value.lower())
        if token not in STOP_WORDS
    }


def constraint_identifier(row: dict[str, Any], index: int) -> str:
    return str(row.get("constraint_id", row.get("id", f"X{index}")))


def substantive_constraints(
    constraints: Iterable[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    for index, row in enumerate(constraints):
        identifier = constraint_identifier(row, index)
        if re.fullmatch(r"(?:C\d+|hc-\d+)", identifier, re.IGNORECASE):
            result.append((identifier, row))
    return result


def trigger_idf(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    document_count = 0
    for record in records:
        for _, constraint in substantive_constraints(
            record["oracle_data"]["hidden_constraints"]
        ):
            document_count += 1
            trigger_tokens = content_tokens(
                " ".join(constraint["trigger_keywords"])
            )
            document_frequency.update(trigger_tokens)
    return {
        token: math.log((document_count + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }


def lexical_unlock_edges(
    record: dict[str, Any],
    *,
    idf: dict[str, float],
    minimum_shared_tokens: int = 2,
    minimum_idf_score: float = 6.0,
) -> list[tuple[str, str]]:
    fuzzy_tokens = content_tokens(record["fuzzy_prd"])
    constraints = substantive_constraints(
        record["oracle_data"]["hidden_constraints"]
    )
    edges: list[tuple[str, str]] = []
    for source_id, source in constraints:
        introduced = content_tokens(source["oracle_response"]) - fuzzy_tokens
        for target_id, target in constraints:
            if source_id == target_id:
                continue
            target_tokens = (
                content_tokens(" ".join(target["trigger_keywords"]))
                - fuzzy_tokens
            )
            shared = introduced & target_tokens
            score = sum(idf.get(token, 0.0) for token in shared)
            if (
                len(shared) >= minimum_shared_tokens
                and score >= minimum_idf_score
            ):
                edges.append((source_id, target_id))
    return sorted(set(edges))


def _load_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "Frozen ICAE manifest hash mismatch: "
            f"expected {EXPECTED_MANIFEST_SHA256}, found {digest}"
        )
    manifest = json.loads(raw)
    if manifest.get("source_commit") != EXPECTED_COMMIT:
        raise ValueError("Frozen ICAE source commit mismatch")
    mechanics = manifest.get("partitions", {}).get("mechanics")
    if not isinstance(mechanics, list) or len(mechanics) != 12:
        raise ValueError("Frozen ICAE manifest must contain 12 mechanics tasks")
    return manifest


def _validate_source(repo: Path) -> None:
    if git_output(repo, "rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise ValueError("ICAE source checkout does not match frozen commit")
    if git_output(repo, "rev-parse", "HEAD^{tree}") != EXPECTED_TREE:
        raise ValueError("ICAE source checkout does not match frozen tree")
    expected_files = {
        "user_agent/user_agent.py": EXPECTED_ORACLE_CORE_SHA256,
        "user_agent/init.md": EXPECTED_ORACLE_PROMPT_SHA256,
        "user_agent/main.py": EXPECTED_ORACLE_SERVER_SHA256,
    }
    for relative, expected in expected_files.items():
        actual = sha256_bytes((repo / relative).read_bytes())
        if actual != expected:
            raise ValueError(f"Frozen ICAE source hash mismatch: {relative}")


def _test_surface_counts(test_root: Path) -> dict[str, int]:
    return {
        name: len(list((test_root / name).rglob("*.json")))
        for name in (
            "public_test_cases",
            "test_cases",
            "enhanced_test_cases",
        )
    }


def build_audit(
    repo: Path,
    manifest_path: Path,
    test_archive: Path,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    _validate_source(repo)
    if sha256_bytes(test_archive.read_bytes()) != EXPECTED_TEST_ARCHIVE_SHA256:
        raise ValueError("ICAE authoritative test archive hash mismatch")

    alias_records = json.loads((repo / "repo_alias.json").read_text())
    all_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((repo / "user_agent/prd_json").glob("*.json"))
    ]
    if len(all_records) != 480:
        raise ValueError("Expected 480 ICAE Oracle records")
    idf = trigger_idf(all_records)

    task_rows: list[dict[str, Any]] = []
    for manifest_row in manifest["partitions"]["mechanics"]:
        alias = manifest_row["alias"]
        record_path = repo / manifest_row["oracle_record"]
        if (
            sha256_bytes(record_path.read_bytes())
            != manifest_row["oracle_record_sha256"]
        ):
            raise ValueError(f"Frozen Oracle-record hash mismatch: {alias}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        constraints = substantive_constraints(
            record["oracle_data"]["hidden_constraints"]
        )
        edges = lexical_unlock_edges(record, idf=idf)

        private_key = alias_records[alias]["key"]
        tests = repo / "rcb_tests_repos" / private_key / "rcb_tests"
        surface_counts = _test_surface_counts(tests)
        task_rows.append(
            {
                "alias": alias,
                "language": manifest_row["language"],
                "substantive_constraint_count": len(constraints),
                "total_oracle_entry_count": len(
                    record["oracle_data"]["hidden_constraints"]
                ),
                "lexical_unlock_edge_count": len(edges),
                "lexical_unlock_source_count": len(
                    {source for source, _ in edges}
                ),
                "lexical_unlock_target_count": len(
                    {target for _, target in edges}
                ),
                "released_test_surfaces": surface_counts,
                "all_three_test_surfaces_nonempty": all(
                    count > 0 for count in surface_counts.values()
                ),
                "oracle_response_literal_in_fuzzy_prd": any(
                    constraint["oracle_response"].lower()
                    in record["fuzzy_prd"].lower()
                    for _, constraint in constraints
                ),
            }
        )

    oracle_core = (repo / "user_agent/user_agent.py").read_text(
        encoding="utf-8"
    )
    oracle_prompt = (repo / "user_agent/init.md").read_text(encoding="utf-8")
    oracle_server = (repo / "user_agent/main.py").read_text(encoding="utf-8")
    simulator_checks = {
        "oracle_uses_only_injected_hidden_data": all(
            token in oracle_core
            for token in (
                'oracle = record.get("oracle_data", {})',
                "# Injected oracle_data (your single source of truth)",
            )
        ),
        "complete_history_conditions_reply": all(
            token in oracle_core
            for token in (
                'self.messages.append({"role": "user", "content": question})',
                "_call_with_rr(model, self.system, self.messages)",
                'self.messages.append({"role": "assistant", "content": text})',
            )
        ),
        "unknown_questions_do_not_auto_progress": all(
            token in oracle_prompt
            for token in (
                "Unknown Question Interception",
                "fallback_response",
                "refrain from deriving an answer yourself",
            )
        ),
        "at_most_three_points_per_reply": all(
            token in oracle_prompt
            for token in (
                "up to 3 specific technical points",
                "only answer the first 3 points",
            )
        ),
        "free_form_semantic_matching_is_model_native": all(
            token in oracle_prompt
            for token in (
                "semantically and explicitly touches",
                "trigger_keywords",
            )
        )
        and "_call_with_rr(model, self.system, self.messages)" in oracle_core,
        "sessions_are_forkable_by_fresh_identifier": all(
            token in oracle_server
            for token in (
                "UserAgentSession(",
                "sessions",
                "append_id",
            )
        ),
        "query_budget_is_enforced": all(
            token in oracle_server
            for token in (
                "max_interactions",
                "max_interactions_reached",
            )
        ),
    }

    counts = {
        "mechanics_tasks": len(task_rows),
        "tasks_with_at_least_8_substantive_constraints": sum(
            row["substantive_constraint_count"] >= 8 for row in task_rows
        ),
        "tasks_with_at_least_2_unlock_targets": sum(
            row["lexical_unlock_target_count"] >= 2 for row in task_rows
        ),
        "tasks_with_all_test_surfaces": sum(
            row["all_three_test_surfaces_nonempty"] for row in task_rows
        ),
        "tasks_with_literal_oracle_response_leak": sum(
            row["oracle_response_literal_in_fuzzy_prd"] for row in task_rows
        ),
    }
    gates = {
        "source_and_manifest_hashes_match": True,
        "substantive_uncertainty_at_least_10_of_12": (
            counts["tasks_with_at_least_8_substantive_constraints"] >= 10
        ),
        "answer_introduced_followups_at_least_10_of_12": (
            counts["tasks_with_at_least_2_unlock_targets"] >= 10
        ),
        "external_executable_endpoint_all_12": (
            counts["tasks_with_all_test_surfaces"] == 12
        ),
        "no_literal_answer_leak_all_12": (
            counts["tasks_with_literal_oracle_response_leak"] == 0
        ),
        "semantic_history_conditioned_oracle": all(simulator_checks.values()),
    }
    all_pass = all(gates.values())
    return {
        "schema_version": 1,
        "audit": "icae_bench_llm_native_source_opportunity",
        "source_commit": EXPECTED_COMMIT,
        "source_tree": EXPECTED_TREE,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "test_archive_sha256": EXPECTED_TEST_ARCHIVE_SHA256,
        "model_requests": 0,
        "openrouter_cost_usd": 0.0,
        "opened_partitions": ["mechanics"],
        "sealed_partitions": ["development", "confirmation", "retained"],
        "lexical_unlock_definition": {
            "candidate_constraints": "C<number> or hc-<number> only",
            "tokens": "lowercase content tokens absent from fuzzy PRD",
            "edge": (
                "source response and target triggers share >=2 tokens "
                "with corpus-IDF score >=6"
            ),
        },
        "simulator_checks": simulator_checks,
        "counts": counts,
        "gates": gates,
        "all_pass": all_pass,
        "decision": (
            "authorize_separately_frozen_10_call_semantic_serving_smoke"
            if all_pass
            else "close_before_model_calls"
        ),
        "limitations": [
            (
                "The hidden constraint table is static; path dependence must "
                "come from history-conditioned LLM belief/query regeneration."
            ),
            (
                "Lexical unlocks establish opportunity, not causal efficacy "
                "or ranking fidelity."
            ),
            (
                "A later policy claim must use released executable tests; "
                "constraint recall alone is not an efficacy endpoint."
            ),
            (
                "Docker is not currently running locally; serving can be "
                "tested before endpoint execution, but a formal endpoint run "
                "requires the official language containers."
            ),
        ],
        "tasks": task_rows,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_atomic(
        args.output.resolve(),
        build_audit(
            args.repo.resolve(),
            args.manifest.resolve(),
            args.test_archive.resolve(),
        ),
    )


if __name__ == "__main__":
    main()
