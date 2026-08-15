#!/usr/bin/env python3
"""Independently replay the ChemBench LLM proposal-atlas artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.chembench_factored_mopen_oracle import sha256
from scripts.chembench_llm_proposal_atlas import (
    SCHEMA_VERSION,
    build_source_package,
    evaluate_records,
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_response_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("records")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("response bank must contain a list of records")
    return value


def verify(
    source_root: Path,
    manifest_path: Path,
    labels_path: Path,
    summary_path: Path,
    *,
    responses_path: Path | None = None,
    evaluation_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_object(manifest_path)
    labels = load_object(labels_path)
    summary = load_object(summary_path)
    if manifest.get("schema_version") != f"{SCHEMA_VERSION}-public-manifest":
        raise ValueError("proposal-atlas manifest schema mismatch")
    if labels.get("schema_version") != f"{SCHEMA_VERSION}-sealed-labels":
        raise ValueError("proposal-atlas label schema mismatch")
    if summary.get("schema_version") != f"{SCHEMA_VERSION}-source-summary":
        raise ValueError("proposal-atlas source summary schema mismatch")
    if summary.get("status") != "passed" or not summary["source_gate"]["passed"]:
        raise ValueError("proposal-atlas source summary did not pass")
    if summary["manifest"]["sha256"] != sha256(manifest_path):
        raise ValueError("proposal-atlas manifest file hash mismatch")
    if summary["labels"]["sha256"] != sha256(labels_path):
        raise ValueError("proposal-atlas labels file hash mismatch")
    implementation_commit = str(manifest["implementation_commit"])
    package = build_source_package(source_root, implementation_commit)
    if package["manifest"] != manifest:
        raise ValueError("proposal-atlas manifest does not independently reconstruct")
    if package["labels"] != labels:
        raise ValueError("proposal-atlas sealed labels do not independently reconstruct")
    checks = {
        "manifest_reconstructed": True,
        "labels_reconstructed": True,
        "source_gate_replayed": bool(manifest["source_gate"]["passed"]),
        "request_count_126": len(manifest["requests"]) == 126,
        "task_count_36": len(manifest["tasks"]) == 36,
        "model_calls_zero_at_source_gate": summary.get("model_calls") == 0,
        "cost_zero_at_source_gate": float(summary.get("cost_usd", -1.0)) == 0.0,
    }
    semantic = None
    if (responses_path is None) != (evaluation_path is None):
        raise ValueError("responses and evaluation must be supplied together")
    if responses_path is not None and evaluation_path is not None:
        records = load_response_records(responses_path)
        expected = load_object(evaluation_path)
        semantic = evaluate_records(source_root, manifest, labels, records)
        if semantic != expected:
            raise ValueError("semantic evaluation does not independently replay")
        checks.update(
            {
                "response_count_126": len(records) == 126,
                "semantic_evaluation_replayed": True,
            }
        )
    if not all(checks.values()):
        raise RuntimeError(f"proposal-atlas independent verification failed: {checks}")
    return {
        "schema_version": f"{SCHEMA_VERSION}-verification",
        "status": "passed",
        "checks": checks,
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "labels": {"path": str(labels_path), "sha256": sha256(labels_path)},
        "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "responses": (
            None
            if responses_path is None
            else {"path": str(responses_path), "sha256": sha256(responses_path)}
        ),
        "evaluation": (
            None
            if evaluation_path is None
            else {"path": str(evaluation_path), "sha256": sha256(evaluation_path)}
        ),
        "semantic_status": None if semantic is None else semantic["provisional_status"],
    }


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing verification: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.source_root,
        args.manifest,
        args.labels,
        args.summary,
        responses_path=args.responses,
        evaluation_path=args.evaluation,
    )
    write_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
