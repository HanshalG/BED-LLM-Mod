#!/usr/bin/env python3
"""Audit released LogDx-CI traces for observation-dependent tool chains."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "logdx-agent-chain-source-audit-2"
SUPERSEDED_AUDIT_SHA256 = (
    "f7f7289f33434a7e9b3a69bc7dd54abbaedbd55a92fb28be8e29090500c1d9ab"
)
SOURCE_ROOT = REPO_ROOT / "external/LogDx"
SOURCE_COMMIT = "99591c1471118c95155976346df72f520a05f100"
SOURCE_TAG_OBJECT = "8d358ae6b973320b8a5e29ce7e0eab243033d31f"
PROTOCOL_PATH = SOURCE_ROOT / "protocols/logdx-ci-v1.2.lock.json"
PROTOCOL_SHA256 = "c5e77e91267036ceb1e80df1d035caa78eb0c831ae690ab5eb0149215fb36087"
SEARCH_TOOLS_PATH = SOURCE_ROOT / "tools/log_search_tools.py"
SEARCH_TOOLS_SHA256 = "4c839cc40ad90f530fe08cb94daa020e69de72eec6b575dbcaa009bb0a71afcc"
AGENT_TOOLS_PATH = SOURCE_ROOT / "examples/diagnosis_shim_claude_agent.py"
AGENT_TOOLS_SHA256 = "25bed12c8bbfd3c6992d4efe197f4681cd2f19b7aca380da4a2e275ad3b33d42"
AGENT_PROMPT_PATH = SOURCE_ROOT / "prompts/agent_v1.md"
AGENT_PROMPT_SHA256 = "51986cc58f73058e9f4724708309257c5ccf98f06c6343618cd80b8ce240c6bd"
RUNNER_PATH = SOURCE_ROOT / "tools/run_diagnosis.py"
RUNNER_SHA256 = "d226abeff82f4468f722e1c8aec8f3278f5792fdcbda4097fcca37d1c1f43238"
EVALUATOR_PATH = SOURCE_ROOT / "tools/evaluate_diagnosis.py"
EVALUATOR_SHA256 = "ff6a51b19049b40c80ad29c7c9ca3a25fc184b01d443acbcc9c3463b12472e54"
GROUND_TRUTH_SCHEMA_PATH = SOURCE_ROOT / "schemas/ground_truth.schema.json"
GROUND_TRUTH_SCHEMA_SHA256 = (
    "ff4a339b8583125b537632c9629395438e7f7437e73032c42c4e1eff8bd15921"
)
DIAGNOSIS_EVAL_SCHEMA_PATH = SOURCE_ROOT / "schemas/diagnosis_eval.schema.json"
DIAGNOSIS_EVAL_SCHEMA_SHA256 = (
    "0b89131054e349032349ec9f9b72bb04c2b2cf49ff69ff64ebb52197f4f46b46"
)

MANIFESTS = {
    "dev": (
        SOURCE_ROOT / "cases/dev/split_manifest.json",
        "44acf2de7843d278e6f2153adbc5f54cdebcb7b416da96effb4becb87062296e",
    ),
    "holdout": (
        SOURCE_ROOT / "cases/holdout/split_manifest.json",
        "f00b9af42a5fe32ba6862e9c7e26f300560f7b8f39676f327def5d4500d14ded",
    ),
    "stress": (
        SOURCE_ROOT / "cases/stress/split_manifest.json",
        "dd856ae0ecfe9dd36f4128c0db9ab233e3195ef09a416445c508ab0c8d0cd127",
    ),
    "v2/dev": (
        SOURCE_ROOT / "cases/v2/dev/split_manifest.json",
        "37e06c718d9908eeb747714f9fccbd200da5fa424943df452ae2f2a00c63eb66",
    ),
    "v2/holdout": (
        SOURCE_ROOT / "cases/v2/holdout/split_manifest.json",
        "3fb7b4ebaed4a4f4c21706dfdb7b7ee342925eabf6587c9538c27afc73fe55d1",
    ),
    "v2/stress": (
        SOURCE_ROOT / "cases/v2/stress/split_manifest.json",
        "9c8e885463a7e9d7b094c13416f1f596782e2d35a3a0310b6586562162f121fe",
    ),
}
EXPECTED_CASE_COUNTS = {
    "dev": 5,
    "holdout": 5,
    "stress": 6,
    "v2/dev": 3,
    "v2/holdout": 10,
    "v2/stress": 6,
}

MIN_MATCHED_CASES = 25
MIN_TOOL_CASES = 12
MIN_MULTI_TOOL_CASES = 8
MIN_DEPENDENCY_CASES = 6
MIN_DEPENDENCY_IMPROVED_CASES = 5
MIN_IMPROVEMENT = 0.10
MIN_DEPENDENCY_MEAN_GAIN = 0.10
MIN_DEPENDENCY_TYPES = 2

_GENERIC_TERMS = {
    "assert",
    "error",
    "errors",
    "exception",
    "fail",
    "failed",
    "failure",
    "failures",
    "panic",
    "traceback",
    "unknown",
}
_ERROR_TERMS = {
    "assert",
    "error",
    "exception",
    "fail",
    "failed",
    "failure",
    "panic",
    "traceback",
}
_FILE_EXTENSION_RE = re.compile(
    r"\.(?:c|cc|cpp|go|h|hpp|java|js|json|jsx|md|py|rs|sh|ts|tsx|txt|yaml|yml)$",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


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
        "tag_object": _git_value("rev-parse", "v1.2"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "search_tools_sha256": sha256_file(SEARCH_TOOLS_PATH),
        "agent_tools_sha256": sha256_file(AGENT_TOOLS_PATH),
        "agent_prompt_sha256": sha256_file(AGENT_PROMPT_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "evaluator_sha256": sha256_file(EVALUATOR_PATH),
        "ground_truth_schema_sha256": sha256_file(GROUND_TRUTH_SCHEMA_PATH),
        "diagnosis_eval_schema_sha256": sha256_file(DIAGNOSIS_EVAL_SCHEMA_PATH),
        "manifest_sha256": {
            split: sha256_file(path)
            for split, (path, _) in MANIFESTS.items()
        },
    }
    expected = {
        "commit": SOURCE_COMMIT,
        "tag_object": SOURCE_TAG_OBJECT,
        "protocol_sha256": PROTOCOL_SHA256,
        "search_tools_sha256": SEARCH_TOOLS_SHA256,
        "agent_tools_sha256": AGENT_TOOLS_SHA256,
        "agent_prompt_sha256": AGENT_PROMPT_SHA256,
        "runner_sha256": RUNNER_SHA256,
        "evaluator_sha256": EVALUATOR_SHA256,
        "ground_truth_schema_sha256": GROUND_TRUTH_SCHEMA_SHA256,
        "diagnosis_eval_schema_sha256": DIAGNOSIS_EVAL_SCHEMA_SHA256,
        "manifest_sha256": {
            split: expected_sha
            for split, (_, expected_sha) in MANIFESTS.items()
        },
    }
    if observed != expected:
        raise ValueError(
            f"LogDx source changed: expected {expected}, observed {observed}"
        )
    return observed


def _case_dir(split: str, case_id: str) -> Path:
    return SOURCE_ROOT / "cases" / split / case_id


def verify_manifests() -> tuple[dict[str, Path], dict[str, Any]]:
    case_paths: dict[str, Path] = {}
    split_counts: dict[str, int] = {}
    for split, (manifest_path, _) in MANIFESTS.items():
        manifest = json.loads(manifest_path.read_text())
        rows = manifest.get("cases")
        if (
            manifest.get("split") != split
            or not isinstance(rows, list)
            or manifest.get("case_count") != EXPECTED_CASE_COUNTS[split]
            or len(rows) != EXPECTED_CASE_COUNTS[split]
        ):
            raise ValueError(f"invalid split manifest: {manifest_path}")
        split_counts[split] = len(rows)
        for row in rows:
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"invalid case id in {manifest_path}")
            if case_id in case_paths:
                raise ValueError(f"duplicate case id across splits: {case_id}")
            case_dir = _case_dir(split, case_id)
            for field, filename in (
                ("raw_log_sha256", "raw.log"),
                ("case_json_sha256", "case.json"),
                ("ground_truth_sha256", "ground_truth.json"),
            ):
                expected_sha = row.get(field)
                observed_sha = sha256_file(case_dir / filename)
                if observed_sha != expected_sha:
                    raise ValueError(
                        f"{case_id} {filename} hash mismatch: "
                        f"{observed_sha} != {expected_sha}"
                    )
            case_paths[case_id] = case_dir
    if len(case_paths) != 35:
        raise ValueError(f"expected 35 unique cases, found {len(case_paths)}")
    return case_paths, {"split_counts": split_counts, "case_count": len(case_paths)}


def verify_ground_truths(case_paths: dict[str, Path]) -> dict[str, int]:
    concrete = 0
    with_signals = 0
    with_evidence = 0
    for case_dir in case_paths.values():
        payload = json.loads((case_dir / "ground_truth.json").read_text())
        root_cause = payload.get("root_cause")
        if (
            isinstance(root_cause, dict)
            and isinstance(root_cause.get("summary"), str)
            and root_cause["summary"].strip()
            and isinstance(root_cause.get("category"), str)
            and root_cause["category"].strip()
        ):
            concrete += 1
        signals = payload.get("required_signals")
        if isinstance(signals, list) and signals:
            with_signals += 1
        evidence = payload.get("evidence_spans")
        if (
            isinstance(evidence, list)
            and evidence
            and all(
                isinstance(item, dict)
                and isinstance(item.get("start_line"), int)
                and isinstance(item.get("end_line"), int)
                and item["start_line"] <= item["end_line"]
                and isinstance(item.get("reason"), str)
                and item["reason"].strip()
                for item in evidence
            )
        ):
            with_evidence += 1
    return {
        "concrete_root_cause_count": concrete,
        "required_signal_count": with_signals,
        "evidence_span_count": with_evidence,
    }


def verify_leakage_boundary() -> dict[str, bool]:
    prompt = AGENT_PROMPT_PATH.read_text()
    shim = AGENT_TOOLS_PATH.read_text()
    runner = RUNNER_PATH.read_text()
    return {
        "prompt_names_only_raw_log_tools": all(
            f"`{name}" in prompt
            for name in ("grep", "read_file", "tail", "view_log_lines")
        ),
        "prompt_forbids_benchmark_ground_truth": (
            "Do not mention benchmark ground truth" in prompt
        ),
        "shim_rejects_ground_truth_keys": all(
            token in shim
            for token in (
                '"ground_truth"',
                '"failure_category"',
                '"required_signals"',
                "forbidden key in payload",
            )
        ),
        "shim_tools_read_only_raw_log": (
            "_load_raw_log_lines(raw_log_path)" in shim
            and "dispatch_tool(name: str, args: dict, raw_log_path: str)" in shim
        ),
        "runner_passes_only_raw_log_path": (
            'payload["raw_log_path"] = raw_log_path' in runner
        ),
    }


def load_agent_tools() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "logdx_diagnosis_shim_claude_agent",
        AGENT_TOOLS_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {AGENT_TOOLS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal_regex_alternatives(pattern: str) -> list[str]:
    candidates: list[str] = []
    for branch in pattern.split("|"):
        branch = branch.strip()
        branch = re.sub(r"\\([.^$*+?{}\[\]\\/()|])", r"\1", branch)
        branch = re.sub(r"\[[^\]]+\]", " ", branch)
        branch = re.sub(r"[\^$*+?{}()\[\]\\]", " ", branch)
        branch = " ".join(branch.split()).strip(" .,:;=-")
        if len(branch) < 4 or branch.lower() in _GENERIC_TERMS:
            continue
        candidates.append(branch)
    return list(dict.fromkeys(candidates))


def classify_literal(candidate: str) -> str:
    lowered = candidate.lower()
    if "/" in candidate or "\\" in candidate or _FILE_EXTENSION_RE.search(candidate):
        return "file_or_path"
    if any(term in lowered for term in _ERROR_TERMS):
        return "error_token"
    if (
        "::" in candidate
        or "_" in candidate
        or re.search(r"\btest[A-Za-z0-9_.:-]*\b", candidate, re.IGNORECASE)
    ):
        return "test_or_symbol"
    return "other_literal"


def _line_prefix_present(text: str, line_number: int) -> bool:
    return re.search(
        rf"(?m)^\s*{re.escape(str(line_number))}\s*:",
        text,
    ) is not None


def dependencies_for_call(
    call: dict[str, Any],
    *,
    initial_context: str,
    prior_observations: Sequence[str],
    prior_calls: Sequence[dict[str, Any]] = (),
) -> list[dict[str, str]]:
    if not prior_observations:
        return []
    prior_text = "\n".join(prior_observations)
    prior_argument_text = canonical_json(
        [prior_call.get("args", {}) for prior_call in prior_calls]
    )
    tool = call.get("tool")
    args = call.get("args")
    if not isinstance(args, dict):
        return []

    dependencies: list[dict[str, str]] = []
    if tool in {"read_file", "view_log_lines"}:
        line_fields = (
            ("center_line",)
            if tool == "view_log_lines"
            else ("start_line", "end_line")
        )
        for field in line_fields:
            value = args.get(field)
            if (
                isinstance(value, int)
                and _line_prefix_present(prior_text, value)
                and not _line_prefix_present(initial_context, value)
                and str(value) not in prior_argument_text
            ):
                dependencies.append(
                    {
                        "dependency_type": "line_number",
                        "argument_field": field,
                    }
                )

    if tool == "grep" and isinstance(args.get("pattern"), str):
        for candidate in _literal_regex_alternatives(args["pattern"]):
            if (
                candidate.casefold() in prior_text.casefold()
                and candidate.casefold() not in initial_context.casefold()
                and candidate.casefold() not in prior_argument_text.casefold()
            ):
                dependencies.append(
                    {
                        "dependency_type": classify_literal(candidate),
                        "argument_field": "pattern",
                    }
                )
    return dependencies


def replay_dependencies(
    *,
    tool_calls: Sequence[dict[str, Any]],
    initial_context: str,
    raw_log_path: Path,
    tools_module: ModuleType,
) -> tuple[list[str], list[dict[str, Any]]]:
    observations: list[str] = []
    dependencies: list[dict[str, Any]] = []
    tool_names: list[str] = []
    for call_index, call in enumerate(tool_calls):
        tool = call.get("tool")
        args = call.get("args")
        if not isinstance(tool, str) or not isinstance(args, dict):
            raise ValueError(f"malformed tool call at index {call_index}")
        tool_names.append(tool)
        for dependency in dependencies_for_call(
            call,
            initial_context=initial_context,
            prior_observations=observations,
            prior_calls=tool_calls[:call_index],
        ):
            dependencies.append(
                {
                    "call_index": call_index,
                    "tool": tool,
                    **dependency,
                }
            )
        observation = tools_module.dispatch_tool(
            tool,
            args,
            str(raw_log_path),
        )
        if not isinstance(observation, str):
            raise ValueError(f"non-string observation at index {call_index}")
        observations.append(observation)
    return tool_names, dependencies


def _methods_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError("evaluation payload has no methods list")
    return {
        method["context_method"]: method
        for method in methods
        if isinstance(method, dict)
        and isinstance(method.get("context_method"), str)
    }


def _cases_by_id(method: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = method.get("cases")
    if not isinstance(cases, list):
        raise ValueError("evaluation method has no cases list")
    return {
        case["case_id"]: case
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }


def _agent_record_path(
    evaluation_path: Path,
    context_method: str,
    case_id: str,
) -> Path:
    return (
        evaluation_path.parent
        / "diagnoses"
        / "real-agent-v1"
        / context_method
        / f"{case_id}.json"
    )


def collect_matched_rows(
    case_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    tools_module = load_agent_tools()
    rows: list[dict[str, Any]] = []
    for agent_eval_path in sorted(
        SOURCE_ROOT.glob("results/**/eval_diagnosis_real-agent-v1.json")
    ):
        debugger_eval_path = (
            agent_eval_path.parent / "eval_diagnosis_real-debugger-v2.json"
        )
        if not debugger_eval_path.exists():
            continue
        agent_eval = json.loads(agent_eval_path.read_text())
        debugger_eval = json.loads(debugger_eval_path.read_text())
        split = agent_eval.get("split")
        if split != debugger_eval.get("split"):
            raise ValueError(f"split mismatch for {agent_eval_path}")

        agent_methods = _methods_by_name(agent_eval)
        debugger_methods = _methods_by_name(debugger_eval)
        for method_name in sorted(set(agent_methods) & set(debugger_methods)):
            agent_cases = _cases_by_id(agent_methods[method_name])
            debugger_cases = _cases_by_id(debugger_methods[method_name])
            for case_id in sorted(set(agent_cases) & set(debugger_cases)):
                if case_id not in case_paths:
                    raise ValueError(f"unknown case id in evaluation: {case_id}")
                agent_case = agent_cases[case_id]
                debugger_case = debugger_cases[case_id]
                agent_score = agent_case.get("diagnosis_score_v1_1")
                debugger_score = debugger_case.get("diagnosis_score_v1_1")
                if not isinstance(agent_score, int | float) or not isinstance(
                    debugger_score, int | float
                ):
                    raise ValueError(
                        f"missing score for {split}/{method_name}/{case_id}"
                    )

                record_path = _agent_record_path(
                    agent_eval_path,
                    method_name,
                    case_id,
                )
                if not record_path.exists():
                    raise ValueError(f"missing agent record: {record_path}")
                record = json.loads(record_path.read_text())
                input_payload = record.get("input")
                agent_metadata = record.get("agent_metadata")
                if not isinstance(input_payload, dict) or not isinstance(
                    agent_metadata, dict
                ):
                    raise ValueError(f"malformed agent record: {record_path}")
                context_path_raw = input_payload.get("context_path")
                tool_calls = agent_metadata.get("tool_calls")
                if not isinstance(context_path_raw, str) or not isinstance(
                    tool_calls, list
                ):
                    raise ValueError(f"malformed agent input: {record_path}")
                context_path = SOURCE_ROOT / context_path_raw
                initial_context = context_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                tool_names, dependencies = replay_dependencies(
                    tool_calls=tool_calls,
                    initial_context=initial_context,
                    raw_log_path=case_paths[case_id] / "raw.log",
                    tools_module=tools_module,
                )
                rows.append(
                    {
                        "split": split,
                        "case_id": case_id,
                        "context_method": method_name,
                        "agent_score": float(agent_score),
                        "single_shot_score": float(debugger_score),
                        "score_gain": float(agent_score) - float(debugger_score),
                        "tool_call_count": len(tool_calls),
                        "tool_names": tool_names,
                        "dependencies": dependencies,
                    }
                )
    return rows


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)

    public_cases: list[dict[str, Any]] = []
    dependency_types: set[str] = set()
    for case_id, case_rows in sorted(by_case.items()):
        dependency_rows = [
            row for row in case_rows if row["dependencies"]
        ]
        case_dependency_types = sorted(
            {
                dependency["dependency_type"]
                for row in dependency_rows
                for dependency in row["dependencies"]
            }
        )
        dependency_types.update(case_dependency_types)
        public_cases.append(
            {
                "case_id": case_id,
                "split": sorted({row["split"] for row in case_rows}),
                "matched_context_count": len(case_rows),
                "tool_context_count": sum(
                    row["tool_call_count"] >= 1 for row in case_rows
                ),
                "multi_tool_context_count": sum(
                    row["tool_call_count"] >= 2 for row in case_rows
                ),
                "dependency_context_count": len(dependency_rows),
                "dependency_types": case_dependency_types,
                "mean_score_gain": _mean(
                    row["score_gain"] for row in case_rows
                ),
                "mean_dependency_score_gain": (
                    _mean(row["score_gain"] for row in dependency_rows)
                    if dependency_rows
                    else None
                ),
            }
        )

    matched_case_count = len(public_cases)
    tool_case_count = sum(
        case["tool_context_count"] >= 1 for case in public_cases
    )
    multi_tool_case_count = sum(
        case["multi_tool_context_count"] >= 1 for case in public_cases
    )
    dependency_cases = [
        case for case in public_cases if case["dependency_context_count"] >= 1
    ]
    improved_dependency_cases = [
        case
        for case in dependency_cases
        if case["mean_dependency_score_gain"] is not None
        and case["mean_dependency_score_gain"] >= MIN_IMPROVEMENT
    ]
    all_case_mean_gain = _mean(
        case["mean_score_gain"] for case in public_cases
    )
    dependency_case_mean_gain = _mean(
        case["mean_dependency_score_gain"]
        for case in dependency_cases
        if case["mean_dependency_score_gain"] is not None
    )

    metrics = {
        "matched_row_count": len(rows),
        "matched_case_count": matched_case_count,
        "tool_case_count": tool_case_count,
        "multi_tool_case_count": multi_tool_case_count,
        "dependency_case_count": len(dependency_cases),
        "dependency_improved_case_count": len(improved_dependency_cases),
        "all_case_mean_score_gain": all_case_mean_gain,
        "dependency_case_mean_score_gain": dependency_case_mean_gain,
        "dependency_types": sorted(dependency_types),
    }
    gates = {
        "matched_cases": matched_case_count >= MIN_MATCHED_CASES,
        "tool_cases": tool_case_count >= MIN_TOOL_CASES,
        "multi_tool_cases": multi_tool_case_count >= MIN_MULTI_TOOL_CASES,
        "dependency_cases": len(dependency_cases) >= MIN_DEPENDENCY_CASES,
        "dependency_improved_cases": (
            len(improved_dependency_cases) >= MIN_DEPENDENCY_IMPROVED_CASES
        ),
        "positive_all_case_mean_gain": all_case_mean_gain > 0.0,
        "dependency_mean_gain": (
            dependency_case_mean_gain >= MIN_DEPENDENCY_MEAN_GAIN
        ),
        "dependency_type_diversity": (
            len(dependency_types) >= MIN_DEPENDENCY_TYPES
        ),
    }
    return {
        "metrics": metrics,
        "gates": gates,
        "cases": public_cases,
    }


def run_audit() -> dict[str, Any]:
    source = verify_source()
    case_paths, manifest_summary = verify_manifests()
    ground_truth_summary = verify_ground_truths(case_paths)
    leakage_boundary = verify_leakage_boundary()
    rows = collect_matched_rows(case_paths)
    aggregate = aggregate_rows(rows)

    source_gates = {
        "source_hashes": True,
        "manifest_hashes": manifest_summary["case_count"] == 35,
        "concrete_root_causes": (
            ground_truth_summary["concrete_root_cause_count"] == 35
        ),
        "required_signals": (
            ground_truth_summary["required_signal_count"] == 35
        ),
        "evidence_spans": (
            ground_truth_summary["evidence_span_count"] == 35
        ),
        "leakage_boundary": all(leakage_boundary.values()),
    }
    gates = {
        **source_gates,
        **aggregate["gates"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "supersedes_audit_sha256": SUPERSEDED_AUDIT_SHA256,
        "source": source,
        "manifest": manifest_summary,
        "ground_truth": ground_truth_summary,
        "leakage_boundary": leakage_boundary,
        "metrics": aggregate["metrics"],
        "gates": gates,
        "status": "passed" if all(gates.values()) else "failed",
        "cases": aggregate["cases"],
        "cost_usd": 0.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "results/nonmyopic/logdx_agent_chain_source_audit_v2/AUDIT.json"
        ),
    )
    args = parser.parse_args(argv)
    payload = run_audit()
    checkpoint(args.output, payload)
    print(canonical_json({
        "status": payload["status"],
        "metrics": payload["metrics"],
        "gates": payload["gates"],
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
