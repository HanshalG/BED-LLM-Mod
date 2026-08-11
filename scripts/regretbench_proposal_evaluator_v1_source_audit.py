#!/usr/bin/env python3
"""Audit untouched RegretBench tasks for proposal-evaluator v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import regretbench_factorized_v2_smoke as predecessor
from scripts import regretbench_deepseek_dynamic_depth2_policy as root_policy
from regretbench.schemas.cig import load_cig


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-proposal-evaluator-v1-source-audit-1"
SOURCE_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_PROPOSAL_EVALUATOR_V1_SOURCE_PROTOCOL_20260811.md"
)
SOURCE_PROTOCOL_SHA256 = (
    "9edb0a7828dfa74b660bea02fc80c50db05be4b6df571efe0911693ee41c190e"
)
SMOKE_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_PROPOSAL_EVALUATOR_V1_EXACT20_SMOKE_PROTOCOL_20260811.md"
)
SMOKE_PROTOCOL_SHA256 = (
    "c56939f2f45c1da2a5dd2dc4e6ab8fd7921cf3033febe085a4d3a0933cd5583b"
)
PREDECESSOR_SOURCE_MANIFEST = predecessor.SOURCE_MANIFEST
PREDECESSOR_SOURCE_MANIFEST_SHA256 = predecessor.SOURCE_MANIFEST_SHA256
PREDECESSOR_SOURCE_RESULT = predecessor.SOURCE_RESULT
PREDECESSOR_SOURCE_RESULT_SHA256 = predecessor.SOURCE_RESULT_SHA256
PREDECESSOR_TERMINAL_REPORT = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_FACTORIZED_V2_AUG11_TERMINAL_RESULT_20260811.md"
)
PREDECESSOR_TERMINAL_REPORT_SHA256 = (
    "085ddb06aef35821ab3b846d673037b9e38511ceaa351e4557d4eefbcaf76879"
)
PREDECESSOR_TERMINAL_AUDIT = REPO_ROOT / (
    "results/nonmyopic/regretbench_factorized_v2_smoke/TERMINAL_AUDIT.json"
)
PREDECESSOR_TERMINAL_AUDIT_SHA256 = (
    "5c72f3b085ccfb726654d33e7229ec8c9d38d9a8683d2c5bcae01897563f06f3"
)
PREDECESSOR_FAILURE = REPO_ROOT / (
    "results/nonmyopic/regretbench_factorized_v2_smoke/"
    "DAILY_FAILURE_20260811.json"
)
PREDECESSOR_FAILURE_SHA256 = (
    "8a62f55ae1866ad4bcb4d734b4f71cc24faccb275957c068d7c643526ec421a6"
)
PREDECESSOR_LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-11-regretbench-factorized-v2-smoke.json"
)
PREDECESSOR_LEDGER_SHA256 = (
    "8d8b48c4514c41cfb679908289aecc3942625611605ab3190d0e2a79fb5849cf"
)
PREDECESSOR_PAID_PROMPT_AUDIT = REPO_ROOT / (
    "results/nonmyopic/regretbench_factorized_v2_smoke/PAID_PROMPT_AUDIT.json"
)
PREDECESSOR_PAID_PROMPT_AUDIT_SHA256 = (
    "60268e7b9c5eb5547c1c32ad371f0a70406c6ed1b794706ef4ac86bee6d7febe"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "results/nonmyopic/regretbench_proposal_evaluator_v1_source_audit"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_hash(path: Path, digest: str, name: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"proposal-evaluator source binding changed: {name}")


def classify_root_payloads(
    root_rows: Sequence[Mapping[str, str]],
    paid_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(root_rows) != 4 or len(paid_audits) != 6:
        raise ValueError("unexpected predecessor task or paid-audit count")
    paid_hashes = [str(row.get("payload_sha256", "")) for row in paid_audits]
    root_hashes = [str(row["root_payload_sha256"]) for row in root_rows]
    return {
        "paid_root_hashes_equal_positions_zero_one_in_order": paid_hashes[:2]
        == root_hashes[:2],
        "positions_two_three_absent_from_all_paid_payloads": not (
            set(root_hashes[2:]) & set(paid_hashes)
        ),
        "paid_root_positions": [
            index for index, digest in enumerate(root_hashes) if digest in paid_hashes
        ],
        "untouched_positions": [
            index for index, digest in enumerate(root_hashes) if digest not in paid_hashes
        ],
    }


def build_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    bindings = {
        "source_protocol": (SOURCE_PROTOCOL, SOURCE_PROTOCOL_SHA256),
        "smoke_protocol": (SMOKE_PROTOCOL, SMOKE_PROTOCOL_SHA256),
        "predecessor_source_manifest": (
            PREDECESSOR_SOURCE_MANIFEST,
            PREDECESSOR_SOURCE_MANIFEST_SHA256,
        ),
        "predecessor_source_result": (
            PREDECESSOR_SOURCE_RESULT,
            PREDECESSOR_SOURCE_RESULT_SHA256,
        ),
        "predecessor_terminal_report": (
            PREDECESSOR_TERMINAL_REPORT,
            PREDECESSOR_TERMINAL_REPORT_SHA256,
        ),
        "predecessor_terminal_audit": (
            PREDECESSOR_TERMINAL_AUDIT,
            PREDECESSOR_TERMINAL_AUDIT_SHA256,
        ),
        "predecessor_failure": (PREDECESSOR_FAILURE, PREDECESSOR_FAILURE_SHA256),
        "predecessor_ledger": (PREDECESSOR_LEDGER, PREDECESSOR_LEDGER_SHA256),
        "predecessor_paid_prompt_audit": (
            PREDECESSOR_PAID_PROMPT_AUDIT,
            PREDECESSOR_PAID_PROMPT_AUDIT_SHA256,
        ),
    }
    for name, (path, digest) in bindings.items():
        require_hash(path, digest, name)

    source_manifest = load_object(PREDECESSOR_SOURCE_MANIFEST)
    source_result = load_object(PREDECESSOR_SOURCE_RESULT)
    terminal = load_object(PREDECESSOR_TERMINAL_AUDIT)
    failure = load_object(PREDECESSOR_FAILURE)
    ledger = load_object(PREDECESSOR_LEDGER)
    paid_projection = load_object(PREDECESSOR_PAID_PROMPT_AUDIT)
    split = source_manifest.get("splits", {}).get("mechanics", {})
    ids = list(split.get("ids", []))
    files = dict(split.get("files_sha256", {}))

    root_rows = []
    for position, cig_id in enumerate(ids):
        task_path = predecessor.REGRETBENCH_ROOT / (
            f"data/OpenDomainQA/test/{cig_id}.json"
        )
        expected_task_hash = files.get(f"test/{cig_id}.json")
        if expected_task_hash is None or sha256_file(task_path) != expected_task_hash:
            raise ValueError("frozen mechanics task file changed")
        cig = load_cig(task_path)
        _, root_audit = root_policy.messages_for(cig, [])
        root_rows.append(
            {
                "position": position,
                "task_id": cig_id,
                "task_file_sha256": expected_task_hash,
                "root_payload_sha256": root_audit["payload_sha256"],
            }
        )

    paid_audits = paid_projection.get("paid_payload_audits", [])
    classification = classify_root_payloads(root_rows, paid_audits)
    successor = root_rows[2:]
    predecessor_run = REPO_ROOT / (
        "results/nonmyopic/regretbench_factorized_v2_smoke/smoke-20260811"
    )
    forbidden_outputs = [
        predecessor_run / "RESULT.json",
        predecessor_run / "VERIFICATION.json",
        predecessor_run / "ENDPOINTS.json",
    ]
    gates = {
        "predecessor_source_passed": source_result.get("status")
        == "source_protocol_pass",
        "predecessor_failed_closed": terminal.get("status") == "failed_closed"
        and failure.get("status") == "failed_closed",
        "predecessor_authorizes_nothing": terminal.get("authorizes") == "nothing"
        and failure.get("authorizes") == "nothing",
        "predecessor_exact_six_calls_before_static": terminal.get(
            "accepted_requests"
        )
        == {"root": 2, "transition": 4, "static_likelihood": 0, "total": 6},
        "predecessor_stage_failed_closed": ledger.get("stage", {}).get("status")
        == "failed_closed",
        "predecessor_private_projection_exact": paid_projection.get("status")
        == "verified_hash_only_projection"
        and paid_projection.get("private_raw_responses_sha256")
        == "9d5d718aaa4782e30469a9d8f65341da8a0ec5ed0bc931f22f2f68e90f9726ce"
        and paid_projection.get("private_privacy_sha256")
        == "e3b0e2665d19768847a9c037a98e57e67e482e08581ca2d7e65abbb76108eb82"
        and set(paid_projection.get("private_raw_response_keys", []))
        == {"root_seeds", "roots", "transition_seeds", "transitions"}
        and paid_projection.get("accepted_requests")
        == {"root": 2, "transition": 4, "static_likelihood": 0, "total": 6},
        "all_paid_prompt_audits_passed": len(paid_audits) == 6
        and [row.get("index") for row in paid_audits] == list(range(6))
        and all(row.get("passed") is True for row in paid_audits),
        "paid_roots_are_exactly_positions_zero_one": classification[
            "paid_root_hashes_equal_positions_zero_one_in_order"
        ]
        and classification["paid_root_positions"] == [0, 1],
        "positions_two_three_never_sent": classification[
            "positions_two_three_absent_from_all_paid_payloads"
        ]
        and classification["untouched_positions"] == [2, 3],
        "successor_has_exactly_two_tasks": len(successor) == 2,
        "no_predecessor_science_outputs_exist": not any(
            path.exists() for path in forbidden_outputs
        ),
    }
    gates["all_pass"] = all(gates.values())
    status = "source_protocol_pass" if gates["all_pass"] else "source_protocol_failed"
    task_manifest = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "smoke_protocol_sha256": SMOKE_PROTOCOL_SHA256,
        "predecessor_mechanics_ids_sha256": split.get("ids_sha256"),
        "paid_predecessor_tasks": root_rows[:2],
        "successor_smoke_tasks": successor,
        "successor_task_ids_sha256": sha256_json(
            [row["task_id"] for row in successor]
        ),
        "development_or_confirmation_tasks_loaded": False,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "authorizes": (
            "proposal_evaluator_v1_exact20_smoke_only"
            if gates["all_pass"]
            else "nothing"
        ),
        "gates": gates,
        "classification": classification,
        "bindings": {
            name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": digest}
            for name, (path, digest) in bindings.items()
        },
        "task_manifest_sha256": sha256_json(task_manifest),
        "model_calls_made": 0,
        "endpoint_outcomes_opened": False,
        "development_opened": False,
        "confirmation_opened": False,
        "cost_usd": 0.0,
    }
    return task_manifest, result


def write_audit(output_dir: Path) -> dict[str, Any]:
    task_manifest, result = build_audit()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SOURCE_TASK_MANIFEST.json").write_text(
        json.dumps(task_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = write_audit(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_protocol_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
