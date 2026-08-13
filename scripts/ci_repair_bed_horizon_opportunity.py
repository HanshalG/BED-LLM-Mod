#!/usr/bin/env python3
"""Audit exact depth-two opportunity in frozen CI-Repair-Bench cohorts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import ci_repair_bed_source_manifest as source


PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/CI_REPAIR_BED_SOURCE_PROTOCOL_20260811.md"
)
PROTOCOL_SHA256 = (
    "8d7ba8895eb7b118563ab5bc2883a16513b2b9adc9f6cb930bb214a46d540f5e"
)
SOURCE_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/ci_repair_bed_source_manifest/SOURCE_MANIFEST.json"
)
SOURCE_MANIFEST_SHA256 = (
    "320080cc5e78673393a74014439f235ca69cd9075d206bfd7d74a9c230536122"
)
OUTPUT_DIR = REPO_ROOT / "results/nonmyopic/ci_repair_bed_horizon_opportunity"
CONTENT_COLUMNS = ("id", "repo_name", "workflow", "logs")
ACTION_IDS = (
    "workflow_head",
    "log_head",
    "log_tail",
    "first_error",
    "last_error",
    "traceback",
    "test_signal",
    "dependency_signal",
)
SIGNALS = {
    "error": re.compile(r"error|exception|fatal|failed|failure", re.I),
    "traceback": re.compile(r"traceback|stack trace|caused by", re.I),
    "test": re.compile(r"test|assert|expected|actual|passed|failed", re.I),
    "dependency": re.compile(
        r"dependency|package|module|import|version|install|resolve", re.I
    ),
}
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
URL_RE = re.compile(r"\b(?:https?|ftp)://\S+", re.I)
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)
TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[t ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:z|[+-]\d{2}:?\d{2})?)?\b",
    re.I,
)
HEX_RE = re.compile(r"\b[0-9a-f]{7,64}\b", re.I)
HOME_PATH_RE = re.compile(
    r"(?<!\w)(?:/home/[^/\s]+|/users/[^/\s]+|/github/workspace|/runner/[^\s]*|/workspace)(?=/|\b)",
    re.I,
)
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
SPACE_RE = re.compile(r"\s+")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def response_hash(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def validate_bindings() -> None:
    for label, path, digest in (
        ("source protocol", PROTOCOL, PROTOCOL_SHA256),
        ("source manifest", SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256),
    ):
        if not path.is_file() or source.sha256_file(path) != digest:
            raise ValueError(f"CI-Repair-Bench {label} changed")
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if manifest["source"]["projected_columns"] != ["id", "repo_name"]:
        raise ValueError("source manifest did not preserve the metadata-only boundary")
    if any(
        manifest[field]
        for field in (
            "workflow_opened",
            "logs_opened",
            "diffs_opened",
            "changed_files_opened",
            "error_types_opened",
        )
    ):
        raise ValueError("source manifest claims source content was already opened")


def canonicalize_response(value: str) -> str:
    value = ANSI_RE.sub(" ", value).lower()
    value = URL_RE.sub("<url>", value)
    value = UUID_RE.sub("<uuid>", value)
    value = TIMESTAMP_RE.sub("<timestamp>", value)
    value = HEX_RE.sub("<hex>", value)
    value = HOME_PATH_RE.sub("<runner_path>", value)
    value = NUMBER_RE.sub("<number>", value)
    return SPACE_RE.sub(" ", value).strip() or "<NO_SIGNAL>"


def nonempty_lines(value: str) -> list[str]:
    return [line.strip() for line in ANSI_RE.sub("", value).splitlines() if line.strip()]


def log_lines(logs: list[dict[str, Any]] | None) -> list[str]:
    lines = []
    for record in logs or []:
        name = str(record.get("name") or "").strip()
        step = str(record.get("step_name") or record.get("setp_name") or "").strip()
        lines.append(f"[job={name}] [step={step}]")
        lines.extend(nonempty_lines(str(record.get("log") or "")))
    return lines


def centered_window(lines: list[str], index: int | None, width: int) -> list[str]:
    if index is None:
        return ["<NO_SIGNAL>"]
    before = width // 2
    start = max(0, index - before)
    return lines[start : start + width]


def first_match(lines: list[str], pattern: re.Pattern[str]) -> int | None:
    return next((index for index, line in enumerate(lines) if pattern.search(line)), None)


def last_match(lines: list[str], pattern: re.Pattern[str]) -> int | None:
    return next(
        (index for index in range(len(lines) - 1, -1, -1) if pattern.search(lines[index])),
        None,
    )


def action_responses(row: dict[str, Any]) -> dict[str, str]:
    workflow = nonempty_lines(str(row.get("workflow") or ""))
    logs = log_lines(row.get("logs"))
    first_error = first_match(logs, SIGNALS["error"])
    last_error = last_match(logs, SIGNALS["error"])
    traceback = first_match(logs, SIGNALS["traceback"])
    raw = {
        "workflow_head": workflow[:40],
        "log_head": logs[:30],
        "log_tail": logs[-30:],
        "first_error": centered_window(logs, first_error, 21),
        "last_error": centered_window(logs, last_error, 21),
        "traceback": centered_window(logs, traceback, 31),
        "test_signal": [line for line in logs if SIGNALS["test"].search(line)][:20]
        or ["<NO_SIGNAL>"],
        "dependency_signal": [
            line for line in logs if SIGNALS["dependency"].search(line)
        ][:20]
        or ["<NO_SIGNAL>"],
    }
    return {
        action_id: response_hash(canonicalize_response("\n".join(raw[action_id])))
        for action_id in ACTION_IDS
    }


def entropy(state_count: int) -> float:
    return math.log(state_count) if state_count > 0 else 0.0


def partition_information(
    response_rows: list[dict[str, str]], action_id: str, indices: tuple[int, ...]
) -> float:
    if len(indices) <= 1:
        return 0.0
    counts = Counter(response_rows[index][action_id] for index in indices)
    total = len(indices)
    posterior_entropy = sum(
        (count / total) * entropy(count) for count in counts.values()
    )
    return entropy(total) - posterior_entropy


def partition_branches(
    response_rows: list[dict[str, str]], action_id: str, indices: tuple[int, ...]
) -> list[tuple[int, ...]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[response_rows[index][action_id]].append(index)
    return [tuple(groups[key]) for key in sorted(groups)]


def two_step_score(
    response_rows: list[dict[str, str]], first_action: str
) -> tuple[float, dict[str, str]]:
    indices = tuple(range(len(response_rows)))
    root_information = partition_information(response_rows, first_action, indices)
    continuations = {}
    continuation_information = 0.0
    for branch in partition_branches(response_rows, first_action, indices):
        options = []
        for second_action in ACTION_IDS:
            if second_action == first_action:
                continue
            options.append(
                (
                    partition_information(response_rows, second_action, branch),
                    second_action,
                )
            )
        best_value = max(value for value, _ in options)
        best_action = min(
            action for value, action in options if abs(value - best_value) <= 1e-12
        )
        branch_key = response_rows[branch[0]][first_action]
        continuations[branch_key] = best_action
        continuation_information += (len(branch) / len(indices)) * best_value
    return root_information + continuation_information, continuations


def repository_metrics(repo_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    response_rows = [action_responses(row) for row in rows]
    indices = tuple(range(len(rows)))
    root_information = {
        action_id: partition_information(response_rows, action_id, indices)
        for action_id in ACTION_IDS
    }
    root_best = max(root_information.values())
    greedy = min(
        action_id
        for action_id, value in root_information.items()
        if abs(value - root_best) <= 1e-12
    )
    scores = {}
    branch_counts = {}
    for action_id in ACTION_IDS:
        score, continuations = two_step_score(response_rows, action_id)
        scores[action_id] = score
        branch_counts[action_id] = len(continuations)
    depth_best = max(scores.values())
    depth_two = min(
        action_id
        for action_id, value in scores.items()
        if abs(value - depth_best) <= 1e-12
    )
    gain = scores[depth_two] - scores[greedy]
    return {
        "repo_name": repo_name,
        "state_count": len(rows),
        "prior_entropy_nats": entropy(len(rows)),
        "nonconstant_action_count": sum(value > 1e-12 for value in root_information.values()),
        "maximum_root_information_nats": root_best,
        "maximum_root_information_fraction": root_best / entropy(len(rows)),
        "greedy_first_action": greedy,
        "greedy_two_step_information_nats": scores[greedy],
        "depth_two_first_action": depth_two,
        "depth_two_information_nats": scores[depth_two],
        "depth_two_changes_first_action": depth_two != greedy,
        "horizon_gain_nats": max(0.0, gain) if gain >= -1e-12 else gain,
        "root_response_partition_counts": branch_counts,
    }


def read_selected_content() -> dict[str, list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read the bound Parquet source") from exc
    metadata = source.read_safe_rows()
    cohorts = source.select_cohorts(metadata)
    opened_splits = ("mechanics", "opportunity")
    selected_ids = {
        row["id"] for split in opened_splits for row in cohorts[split]
    }
    table = pq.read_table(
        source.PARQUET_PATH,
        columns=list(CONTENT_COLUMNS),
        filters=[("id", "in", sorted(selected_ids))],
    )
    rows = [{key: row[key] for key in CONTENT_COLUMNS} for row in table.to_pylist()]
    loaded_ids = {str(row["id"]) for row in rows}
    if loaded_ids != selected_ids or len(rows) != len(selected_ids):
        raise ValueError("selected CI-Repair-Bench content rows changed")
    by_id = {str(row["id"]): row for row in rows}
    return {
        split: [by_id[row["id"]] for row in cohorts[split]] for split in opened_splits
    }


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    validate_bindings()
    cohorts = read_selected_content()
    metrics = {}
    for split_name, rows in cohorts.items():
        by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_repo[str(row["repo_name"])].append(row)
        metrics[split_name] = [
            repository_metrics(repo_name, by_repo[repo_name])
            for repo_name in sorted(by_repo)
        ]
    opportunity = metrics["opportunity"]
    gains = [row["horizon_gain_nats"] for row in opportunity]
    gates = {
        "exact_16_mechanics_states_opened": sum(
            row["state_count"] for row in metrics["mechanics"]
        )
        == 16,
        "exact_69_opportunity_states_opened": sum(
            row["state_count"] for row in opportunity
        )
        == 69,
        "exact_10_opportunity_repositories": len(opportunity) == 10,
        "at_least_8_have_4_nonconstant_actions": sum(
            row["nonconstant_action_count"] >= 4 for row in opportunity
        )
        >= 8,
        "at_least_8_are_not_root_saturated": sum(
            row["maximum_root_information_fraction"] < 0.95 for row in opportunity
        )
        >= 8,
        "at_least_4_change_depth_two_first_action": sum(
            row["depth_two_changes_first_action"] for row in opportunity
        )
        >= 4,
        "at_least_4_have_gain_at_least_0_05_nats": sum(
            gain >= 0.05 for gain in gains
        )
        >= 4,
        "mean_gain_at_least_0_03_nats": sum(gains) / len(gains) >= 0.03,
        "all_gains_are_finite_and_nonnegative": all(
            math.isfinite(gain) and gain >= 0.0 for gain in gains
        ),
        "no_repair_outcome_column_opened": not {
            "diff",
            "changed_files",
            "error_type",
            "sha_success",
        }.intersection(CONTENT_COLUMNS),
        "development_confirmation_reserve_unopened": set(cohorts)
        == {"mechanics", "opportunity"},
    }
    opportunity_gate_names = (
        "at_least_8_have_4_nonconstant_actions",
        "at_least_8_are_not_root_saturated",
        "at_least_4_change_depth_two_first_action",
        "at_least_4_have_gain_at_least_0_05_nats",
        "mean_gain_at_least_0_03_nats",
        "all_gains_are_finite_and_nonnegative",
    )
    gates["all_integrity_gates_pass"] = all(
        value for name, value in gates.items() if name not in opportunity_gate_names
    )
    gates["source_opportunity_pass"] = gates["all_integrity_gates_pass"] and all(
        gates[name] for name in opportunity_gate_names
    )
    result = {
        "schema_version": 1,
        "interface_version": "ci-repair-bed-horizon-opportunity-1",
        "status": "source_opportunity_pass"
        if gates["source_opportunity_pass"]
        else "source_opportunity_null",
        "authorizes": "semantic_mechanics_protocol_only"
        if gates["source_opportunity_pass"]
        else "nothing",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "content_columns_opened": list(CONTENT_COLUMNS),
        "opened_splits": ["mechanics", "opportunity"],
        "repair_outcome_columns_opened": [],
        "development_confirmation_reserve_opened": False,
        "gates": gates,
        "summary": {
            "opportunity_repository_count": len(opportunity),
            "changed_first_action_count": sum(
                row["depth_two_changes_first_action"] for row in opportunity
            ),
            "gain_at_least_0_05_count": sum(gain >= 0.05 for gain in gains),
            "mean_horizon_gain_nats": sum(gains) / len(gains),
            "maximum_horizon_gain_nats": max(gains),
        },
        "repository_metrics": metrics,
        "raw_responses_serialized": False,
        "latent_state_ids_serialized": False,
        "model_calls_made": 0,
        "cost_usd": 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "RESULT.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
