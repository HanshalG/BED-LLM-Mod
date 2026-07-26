#!/usr/bin/env python3
"""Audit DiscoverLLM's released preference data for non-myopic BED evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Iterable


SOURCE_REPOSITORY = "https://github.com/tsook/discoverllm"
SOURCE_COMMIT = "a9eb2846f60e3681ac8d325fc57fd4e58e2bdc97"
DATASET_REPOSITORY = (
    "https://huggingface.co/datasets/"
    "kixlab/DiscoverLLM-multiturn-preferences"
)
DATASET_REVISION = "c857bbf6265bdd573938eb7eac79a7a3131fa7ca"
EXPECTED_PARQUET_SHA256 = {
    "creative_writing": (
        "e8f76a47447b442e59e0d76718228b94bba1a11d8f7f653cdf1498bb0adf7843"
    ),
    "svg_drawing": (
        "aef93ede39dbbf570bd15bda2b217d4a96dec188af77eeb5d88bcaed869a081d"
    ),
    "technical_writing": (
        "4dcc29e2cb9f21d4dabaa7d8eeaad773a433248a2be94498968e908b6858ccd0"
    ),
}
REQUIRED_COLUMNS = {
    "artifact_id",
    "turn_id",
    "assistant_index",
    "prompt",
    "completion",
    "score",
    "criteria_history",
    "conv_id",
    "source_dataset",
    "source_id",
    "source_metadata",
}
CANDIDATE_OUTCOME_COLUMNS = {
    "updated_criteria_objs",
    "full_results",
    "future_trajectory",
    "singleturn_score",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_hierarchy(
    hierarchy: list[dict[str, Any]],
    value_key: str = "aware",
) -> tuple[float, float, int]:
    """Reproduce DiscoverLLM's hierarchy score without importing its runtime."""

    total_score = 0.0
    current_score = 0.0
    total_count = 0
    for node in hierarchy:
        children = node.get("children", []) or []
        children_score = 0.0
        if children:
            descendant_score, children_score, descendant_count = (
                score_hierarchy(children, value_key)
            )
            total_score += descendant_score
            total_count += descendant_count
        if children and math.isclose(children_score, len(children)):
            current_score += 1.0
        else:
            current_score += float(node.get(value_key, 0.0))
    return (
        total_score + current_score,
        current_score,
        total_count + len(hierarchy),
    )


def criteria_score(
    criteria: list[dict[str, Any]],
    value_key: str = "aware",
) -> float:
    values = [
        score_hierarchy(criterion.get("hierarchy", []), value_key)[0]
        for criterion in criteria
    ]
    return sum(values) / len(values) if values else 0.0


def walk_hierarchy(
    hierarchy: Iterable[dict[str, Any]],
    depth: int = 1,
) -> Iterable[tuple[dict[str, Any], int]]:
    for node in hierarchy:
        yield node, depth
        yield from walk_hierarchy(node.get("children", []) or [], depth + 1)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    )
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator > 0 else None


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(
        right,
        sort_keys=True,
    )


def infer_committed_transitions(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Recover chosen-action state changes from the following turn's prompt."""

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["artifact_id"]), int(row["turn_id"]))].append(row)

    transitions: list[dict[str, Any]] = []
    diagnostics = {
        "turn_groups": len(groups),
        "turn_groups_without_observed_successor": 0,
        "committed_completion_match_failures": 0,
        "candidate_prestate_inconsistencies": 0,
    }
    for (artifact_id, turn), candidates in sorted(groups.items()):
        histories = [candidate["criteria_history"] for candidate in candidates]
        if any(not _same_json(histories[0], other) for other in histories[1:]):
            diagnostics["candidate_prestate_inconsistencies"] += 1

        successor = groups.get((artifact_id, turn + 1))
        if not successor:
            diagnostics["turn_groups_without_observed_successor"] += 1
            continue
        assistant_messages = [
            message.get("content", "")
            for message in successor[0]["prompt"]
            if message.get("role") == "assistant"
        ]
        if len(assistant_messages) < turn:
            diagnostics["committed_completion_match_failures"] += 1
            continue
        committed_text = assistant_messages[turn - 1]
        matched = [
            candidate
            for candidate in candidates
            if candidate["completion"] == committed_text
        ]
        if len(matched) != 1:
            diagnostics["committed_completion_match_failures"] += 1
            continue

        selected = matched[0]
        pre_state = selected["criteria_history"][-1]
        post_state = successor[0]["criteria_history"][-1]
        immediate_gain = criteria_score(post_state) - criteria_score(pre_state)
        transitions.append(
            {
                "artifact_id": artifact_id,
                "turn_id": turn,
                "assistant_index": int(selected["assistant_index"]),
                "released_score": float(selected["score"]),
                "immediate_awareness_gain": immediate_gain,
                "implied_token_penalty": (
                    immediate_gain - float(selected["score"])
                ),
                "selected_is_max_score": math.isclose(
                    float(selected["score"]),
                    max(float(candidate["score"]) for candidate in candidates),
                    abs_tol=1e-12,
                ),
            }
        )
    diagnostics["recoverable_committed_transitions"] = len(transitions)
    return transitions, diagnostics


def summarize_initial_trees(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_artifact: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["artifact_id"]),
            int(item["turn_id"]),
            int(item["assistant_index"]),
        ),
    ):
        by_artifact.setdefault(
            str(row["artifact_id"]),
            row["criteria_history"][0],
        )

    records: list[dict[str, float]] = []
    for criteria in by_artifact.values():
        roots = [
            root
            for criterion in criteria
            for root in criterion.get("hierarchy", [])
        ]
        nodes = list(walk_hierarchy(roots))
        records.append(
            {
                "nodes": float(len(nodes)),
                "leaves": float(
                    sum(not (node.get("children", []) or []) for node, _ in nodes)
                ),
                "maximum_depth": float(max(depth for _, depth in nodes)),
                "roots": float(len(roots)),
                "initially_hidden_nodes": float(
                    sum(float(node.get("aware", 0.0)) < 1 for node, _ in nodes)
                ),
                "branching_nodes": float(
                    sum(
                        len(node.get("children", []) or []) >= 2
                        for node, _ in nodes
                    )
                ),
            }
        )

    summary: dict[str, Any] = {"artifacts": len(by_artifact)}
    for key in records[0] if records else []:
        values = [record[key] for record in records]
        summary[key] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "positive_artifacts": sum(value > 0 for value in values),
        }
    return summary


def summarize_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["artifact_id"]), int(row["turn_id"]))].append(row)
    transitions, diagnostics = infer_committed_transitions(rows)
    penalties = [
        transition["implied_token_penalty"] for transition in transitions
    ]
    score_values = [transition["released_score"] for transition in transitions]
    gain_values = [
        transition["immediate_awareness_gain"]
        for transition in transitions
    ]
    score_ranges = [
        max(float(row["score"]) for row in group)
        - min(float(row["score"]) for row in group)
        for group in groups.values()
    ]
    tolerance = 1e-9
    return {
        "rows": len(rows),
        "artifacts": len({str(row["artifact_id"]) for row in rows}),
        "turn_groups": len(groups),
        "candidate_count_histogram": dict(
            sorted(Counter(len(group) for group in groups.values()).items())
        ),
        "turn_groups_with_positive_score_range": sum(
            value > tolerance for value in score_ranges
        ),
        "mean_candidate_score_range": (
            statistics.mean(score_ranges) if score_ranges else None
        ),
        "committed_transition_diagnostics": diagnostics,
        "selected_is_max_score": sum(
            transition["selected_is_max_score"]
            for transition in transitions
        ),
        "released_score_immediate_gain_correlation": pearson(
            score_values,
            gain_values,
        ),
        "implied_token_penalty": {
            "mean": statistics.mean(penalties) if penalties else None,
            "minimum": min(penalties) if penalties else None,
            "maximum": max(penalties) if penalties else None,
            "within_documented_zero_to_one_range": sum(
                -tolerance <= penalty <= 1.0 + tolerance
                for penalty in penalties
            ),
        },
        "initial_tree_structure": summarize_initial_trees(rows),
    }


def _load_parquet(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "discoverllm_source_audit requires duckdb to read Parquet files"
        ) from exc

    connection = duckdb.connect()
    escaped_path = str(path.resolve()).replace("'", "''")
    columns = {
        row[0]
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{escaped_path}')"
        ).fetchall()
    }
    selected = connection.execute(
        "SELECT artifact_id, turn_id, assistant_index, score, prompt, "
        "completion, criteria_history "
        f"FROM read_parquet('{escaped_path}') "
        "ORDER BY artifact_id, CAST(turn_id AS INTEGER), assistant_index"
    ).fetchall()
    rows = [
        {
            "artifact_id": artifact_id,
            "turn_id": int(turn_id),
            "assistant_index": int(assistant_index),
            "score": float(score),
            "prompt": prompt,
            "completion": completion,
            "criteria_history": json.loads(criteria_history),
        }
        for (
            artifact_id,
            turn_id,
            assistant_index,
            score,
            prompt,
            completion,
            criteria_history,
        ) in selected
    ]
    return columns, rows


def audit(
    source_root: Path,
    parquet_paths: dict[str, Path],
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if verify_hashes and revision != SOURCE_COMMIT:
        raise ValueError(
            f"DiscoverLLM revision is {revision}, expected {SOURCE_COMMIT}"
        )

    synthesis_script = (
        source_root / "scripts" / "simulate" / "run_synth.sh"
    ).read_text()
    official_window_zero = "--window-size 0" in synthesis_script

    config_results: dict[str, Any] = {}
    all_columns: set[str] | None = None
    total_transitions = 0
    compatible_transitions = 0
    for config_name, path in sorted(parquet_paths.items()):
        digest = sha256_file(path)
        expected_digest = EXPECTED_PARQUET_SHA256[config_name]
        if verify_hashes and digest != expected_digest:
            raise ValueError(
                f"{config_name} Parquet SHA-256 is {digest}, "
                f"expected {expected_digest}"
            )
        columns, rows = _load_parquet(path)
        if not REQUIRED_COLUMNS.issubset(columns):
            missing = sorted(REQUIRED_COLUMNS - columns)
            raise ValueError(f"{config_name} is missing columns: {missing}")
        all_columns = columns if all_columns is None else all_columns & columns
        summary = summarize_rows(rows)
        summary["sha256"] = digest
        config_results[config_name] = summary
        transition_count = summary[
            "committed_transition_diagnostics"
        ]["recoverable_committed_transitions"]
        total_transitions += transition_count
        compatible_transitions += summary["implied_token_penalty"][
            "within_documented_zero_to_one_range"
        ]

    common_columns = all_columns or set()
    gates = {
        "explicit_path_dependent_semantic_state_is_released": all(
            result["initial_tree_structure"]["initially_hidden_nodes"]["mean"]
            > 0
            and result["initial_tree_structure"]["maximum_depth"]["median"]
            >= 2
            for result in config_results.values()
        ),
        "multiple_scored_actions_per_turn_are_released": all(
            set(result["candidate_count_histogram"]) == {2}
            for result in config_results.values()
        ),
        "all_candidate_post_states_are_released": bool(
            "updated_criteria_objs" in common_columns
        ),
        "all_candidate_future_trajectories_are_released": bool(
            "future_trajectory" in common_columns
        ),
        "published_scores_contain_lookahead_value": (
            not official_window_zero
            and compatible_transitions < total_transitions
        ),
        "mutually_exclusive_latent_world_prior_is_released": False,
    }
    direct_replay_eligible = all(gates.values())
    return {
        "interface_version": "discoverllm-source-audit-1",
        "decision": "pass" if direct_replay_eligible else "fail",
        "decision_scope": (
            "eligibility of the public preference dataset for a direct "
            "non-myopic BED replay"
        ),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": revision,
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "synthesis_script_sha256": sha256_file(
                source_root / "scripts" / "simulate" / "run_synth.sh"
            ),
            "reward_code_sha256": sha256_file(
                source_root / "discoverllm" / "pipeline" / "rewards.py"
            ),
        },
        "release_schema": {
            "common_columns": sorted(common_columns),
            "missing_candidate_outcome_columns": sorted(
                CANDIDATE_OUTCOME_COLUMNS - common_columns
            ),
        },
        "mechanics": {
            "official_synthesis_launcher_sets_window_size_zero": (
                official_window_zero
            ),
            "recoverable_committed_transitions": total_transitions,
            "transitions_equal_to_immediate_gain_minus_documented_penalty": (
                compatible_transitions
            ),
        },
        "configs": config_results,
        "gates": gates,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--creative-writing", type=Path, required=True)
    parser.add_argument("--technical-writing", type=Path, required=True)
    parser.add_argument("--svg-drawing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-verify-hashes", action="store_true")
    args = parser.parse_args()

    result = audit(
        args.source_root,
        {
            "creative_writing": args.creative_writing,
            "technical_writing": args.technical_writing,
            "svg_drawing": args.svg_drawing,
        },
        verify_hashes=not args.no_verify_hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
