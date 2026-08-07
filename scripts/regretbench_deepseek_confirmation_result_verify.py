#!/usr/bin/env python3
"""Independently replay a frozen RegretBench confirmation result."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


from scripts import regretbench_deepseek_result_verify as base
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/regretbench_llm_native_source_audit/"
    "SOURCE_PROTOCOL_MANIFEST.json"
)
PROTOCOL_SHA256 = (
    "7a782f02eb8c3b16d5b229cca309d02bce64df6432c5090c977d3e26d1f46498"
)
PROTOCOL_VERIFICATION_SHA256 = (
    "20a47fd4cde521327a1ea1c3c183a6efad0cd0b580aed6773e49829fd841ab3c"
)
PREREGISTRATION_SHA256 = (
    "edbbe7cd9e7ef582fc74629737e6c9261cead4486800bfc8712b0c74e90dd8bc"
)
CONFIRMATION_SPLIT_SHA256 = (
    "780a0e4e172251be2781729eeb4e591592b996dc9e1cd668b26240e3076660c9"
)
INITIAL_SEED_START = 202608209000
BRANCH_SEED_START = 202608210000
ACTUAL_FIRST_SEED_START = 202608220000
ACTUAL_FINAL_SEED_START = 202608230000
TRUTH_SEED_START = 202608240000
RANDOM_SEED_START = 202608250000
BOOTSTRAP_SEED = 202608260000
MAX_REQUESTS = 8_768
POLICY_BUDGET = 3.50


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _split_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _confirmation_cigs() -> list[Any]:
    manifest = _load(SOURCE_MANIFEST)
    ids = list(manifest["splits"]["confirmation"]["ids"])
    if len(ids) != 64 or _split_hash(ids) != CONFIRMATION_SPLIT_SHA256:
        raise ValueError("confirmation source split changed")
    cigs = [
        base.load_cig(
            base.REGRETBENCH_ROOT
            / "data/OpenDomainQA/test"
            / f"{cig_id}.json"
        )
        for cig_id in ids
    ]
    if [cig.cig_id for cig in cigs] != ids:
        raise ValueError("confirmation source order changed")
    return cigs


@contextmanager
def _confirmation_scope() -> Iterator[None]:
    cigs = _confirmation_cigs()
    replacements = {
        "INITIAL_SEED_START": INITIAL_SEED_START,
        "BRANCH_SEED_START": BRANCH_SEED_START,
        "ACTUAL_FIRST_SEED_START": ACTUAL_FIRST_SEED_START,
        "ACTUAL_FINAL_SEED_START": ACTUAL_FINAL_SEED_START,
        "TRUTH_SEED_START": TRUTH_SEED_START,
        "RANDOM_SEED_START": RANDOM_SEED_START,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "MAX_PRIMARY_REQUESTS": MAX_REQUESTS,
        "POLICY_BUDGET": POLICY_BUDGET,
    }
    originals = {name: getattr(base, name) for name in replacements}
    original_loader = base._stage_cigs

    def scoped_loader(stage: str) -> list[Any]:
        if stage == "development":
            return list(cigs)
        return original_loader(stage)

    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        base._stage_cigs = scoped_loader
        yield
    finally:
        base._stage_cigs = original_loader
        for name, value in originals.items():
            setattr(base, name, value)


def verify_confirmation(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    with _confirmation_scope():
        replay = base.verify_policy(run_dir)
    result = _load(run_dir / "RESULT.json")
    protocol = result.get("protocol") or {}
    tasks = result.get("tasks") or []
    source_ids = [cig.cig_id for cig in _confirmation_cigs()]
    usage = result.get("usage") or {}
    primary = usage.get("deepseek_primary") or {}
    naive_luna = usage.get("naive_luna") or {}
    naive_endpoint = usage.get("deepseek_naive_endpoint") or {}
    expected_requests = 8_256 + len(
        _load(run_dir / "private" / "RAW_ACTUAL.json")["first_manifest"]
    ) + len(_load(run_dir / "private" / "RAW_ACTUAL.json")["final_manifest"])

    confirmation_checks = {
        "base_raw_replay_verified": replay.get("status") == "verified",
        "confirmation_interface_exact": result.get("interface_version")
        == "regretbench-deepseek-dynamic-depth2-confirmation-1",
        "confirmation_protocol_bound": (
            protocol.get("stage") == "confirmation"
            and protocol.get("source_split") == "confirmation"
            and protocol.get("confirmation_protocol_sha256") == PROTOCOL_SHA256
            and protocol.get("confirmation_protocol_verification_sha256")
            == PROTOCOL_VERIFICATION_SHA256
            and protocol.get("confirmation_preregistration_sha256")
            == PREREGISTRATION_SHA256
            and protocol.get("confirmation_split_sha256")
            == CONFIRMATION_SPLIT_SHA256
        ),
        "exact_confirmation_seed_metadata": (
            protocol.get("initial_seed_start") == INITIAL_SEED_START
            and protocol.get("branch_seed_start") == BRANCH_SEED_START
            and protocol.get("actual_first_seed_start")
            == ACTUAL_FIRST_SEED_START
            and protocol.get("actual_final_seed_start")
            == ACTUAL_FINAL_SEED_START
            and protocol.get("truth_seed_start") == TRUTH_SEED_START
            and protocol.get("random_seed_start") == RANDOM_SEED_START
            and protocol.get("bootstrap_seed") == BOOTSTRAP_SEED
        ),
        "exact_untouched_confirmation_cohort": (
            len(tasks) == 64
            and [str(task.get("task_id")) for task in tasks] == source_ids
        ),
        "literal_verified_development_authorization": (
            protocol.get("development_authorization", {}).get("status")
            == "authorized"
            and protocol.get("development_authorization", {}).get(
                "development_status"
            )
            == "passed"
            and protocol.get("development_authorization", {}).get(
                "independent_replay_status"
            )
            == "verified"
        ),
        "no_optional_baseline": (
            protocol.get("optional_baselines") == []
            and all("naive_thinking" not in task.get("policies", {}) for task in tasks)
            and result.get("naive_baseline", {}).get("status")
            == "not_in_confirmation_protocol"
            and naive_luna.get("adapter_requests") == 0
            and naive_endpoint.get("adapter_requests") == 0
        ),
        "exact_primary_request_accounting": (
            8_256 <= expected_requests <= MAX_REQUESTS
            and primary.get("adapter_requests") == expected_requests
            and primary.get("http_attempts") == expected_requests
            and protocol.get("maximum_requests") == MAX_REQUESTS
        ),
        "confirmation_flags_exact": (
            result.get("confirmation_opened") is True
            and result.get("development_opened") is False
            and protocol.get("confirmation_opened") is True
        ),
        "authorization_matches_status": result.get("authorizes")
        == ("confirmed_result_only" if result.get("status") == "passed" else "nothing"),
    }
    mismatches = list(replay.get("mismatches") or [])
    mismatches.extend(
        f"$.confirmation_checks.{name}"
        for name, passed in confirmation_checks.items()
        if not passed
    )
    artifacts = {
        str(path.relative_to(run_dir)): base.sha256_file(path)
        for path in sorted(run_dir.rglob("*.json"))
        if path.name != "VERIFICATION.json"
    }
    all_pass = not mismatches
    return {
        "schema_version": 1,
        "interface_version": "regretbench-confirmation-result-verification-1",
        "status": "verified" if all_pass else "verification_failed",
        "kind": "confirmation_policy",
        "result_status": result.get("status"),
        "checks": {
            "raw_artifacts_reparsed": True,
            "truth_controls_replayed": True,
            "scientific_endpoint_recomputed": True,
            "confirmation_contract_replayed": True,
            "reported_result_matches_replay": all_pass,
            **confirmation_checks,
        },
        "mismatches": mismatches,
        "artifact_sha256": artifacts,
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify_confirmation(args.run_dir)
    checkpoint(args.run_dir / "VERIFICATION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
