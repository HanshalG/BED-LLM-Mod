#!/usr/bin/env python3
"""Independently replay the sealed RegretBench SMC confirmation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from scripts import regretbench_deepseek_result_verify as independent
from scripts import regretbench_deepseek_smc_dynamic_depth2_verify as policy_verify
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-smc-confirmation-verify-1"
PRODUCER_INTERFACE = "regretbench-deepseek-smc-confirmation-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
PROTOCOL = independent.REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_SMC_CONFIRMATION_PREREGISTRATION_20260807.md"
)
PROTOCOL_SHA256 = (
    "9d413f2871e301ac4262b23a5e51cf195f34abc481e5ada9e6418736d88bd02b"
)
CONFIRMATION_SPLIT_SHA256 = (
    "780a0e4e172251be2781729eeb4e591592b996dc9e1cd668b26240e3076660c9"
)
PARENT_SEED_START = 202608405000
ANNOTATION_SEED_START = 202608410000
BRANCH_SEED_START = 202608420000
TRUTH_SEED_START = 202608430000
ACTUAL_FIRST_SEED_START = 202608440000
ACTUAL_FINAL_SEED_START = 202608450000
BOOTSTRAP_SEED = 202608460000
RANDOM_SEED_START = 202608470000


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_protocol_binding() -> None:
    if not PROTOCOL.is_file() or _sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("SMC confirmation protocol changed")


def _load_confirmation_cigs() -> list[Any]:
    manifest = _load(independent.SOURCE_MANIFEST)
    split = manifest["splits"]["confirmation"]
    ids = list(split["ids"])
    if len(ids) != 64 or split.get("ids_sha256") != CONFIRMATION_SPLIT_SHA256:
        raise ValueError("SMC confirmation split changed")
    cigs = [
        independent.load_cig(
            independent.REGRETBENCH_ROOT
            / "data/OpenDomainQA/test"
            / f"{cig_id}.json"
        )
        for cig_id in ids
    ]
    if [cig.cig_id for cig in cigs] != ids:
        raise ValueError("SMC confirmation order changed")
    return cigs


@contextmanager
def verification_scope() -> Iterator[None]:
    cigs = _load_confirmation_cigs()
    original_loader = independent._stage_cigs
    replacements = {
        "ANNOTATION_SEED_START": ANNOTATION_SEED_START,
        "BRANCH_SEED_START": BRANCH_SEED_START,
        "TRUTH_SEED_START": TRUTH_SEED_START,
        "RANDOM_SEED_START": RANDOM_SEED_START,
        "ACTUAL_FIRST_SEED_START": ACTUAL_FIRST_SEED_START,
        "ACTUAL_FINAL_SEED_START": ACTUAL_FINAL_SEED_START,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
    }
    originals = {name: getattr(policy_verify, name) for name in replacements}

    def scoped_loader(stage: str) -> list[Any]:
        return list(cigs) if stage == "development" else original_loader(stage)

    try:
        independent._stage_cigs = scoped_loader
        for name, value in replacements.items():
            setattr(policy_verify, name, value)
        yield
    finally:
        independent._stage_cigs = original_loader
        for name, value in originals.items():
            setattr(policy_verify, name, value)


def verify_parent_bank(parent_dir: Path) -> dict[str, Any]:
    _validate_protocol_binding()
    result = _load(parent_dir / "RESULT.json")
    raw = _load(parent_dir / "private/RAW_RESPONSES.json")
    controls = _load(parent_dir / "private/CONTROLS.json")
    cigs = _load_confirmation_cigs()
    responses = raw.get("root") or []
    seeds = raw.get("seeds") or []
    parents = [policy_verify._parse_parent(value) for value in responses]
    control_rows = controls.get("roots") or []
    expected_seeds = [PARENT_SEED_START + index for index in range(64)]
    usage = result.get("usage") or {}
    checks = {
        "exact_confirmation_cohort": result.get("task_ids")
        == [cig.cig_id for cig in cigs],
        "exact_parent_seed_schedule": seeds == expected_seeds,
        "exact_64_independently_parsed_parents": len(parents) == 64,
        "minimal_controls_match_raw_parents": len(control_rows) == 64
        and all(
            row.get("task_id") == cig.cig_id
            and row.get("question") == parent["questions"][0]
            and row.get("raw_parent_sha256")
            == hashlib.sha256(value.encode()).hexdigest()
            for row, cig, parent, value in zip(
                control_rows, cigs, parents, responses, strict=True
            )
        ),
        "no_branch_responses": raw.get("branches") == [],
        "exact_parent_transport_accounting": usage.get("adapter_requests") == 64
        and usage.get("http_attempts") == 64
        and usage.get("retry_count") == 0
        and usage.get("provider_error_retries") == 0
        and usage.get("adapter_reasoning_tokens") == 0
        and usage.get("forced_exits") == 0,
        "exact_parent_model_contract": result.get("protocol", {}).get("model")
        == MODEL_ID
        and result.get("protocol", {}).get("reasoning")
        == "disabled_excluded"
        and result.get("protocol", {}).get("expected_requests") == 64,
        "hidden_truth_not_stored_in_controls": all(
            set(row) == {"task_id", "question", "raw_parent_sha256"}
            for row in control_rows
        ),
        "reported_parent_mechanics_pass": result.get("status") == "passed"
        and result.get("mechanics_gates", {}).get("all_pass") is True,
        "protocol_binding_exact": result.get("protocol", {}).get(
            "protocol_sha256"
        )
        == PROTOCOL_SHA256,
    }
    return {
        "status": "verified" if all(checks.values()) else "verification_failed",
        "checks": checks,
        "mismatches": [key for key, passed in checks.items() if not passed],
        "model_calls": 0,
        "cost_usd": 0.0,
        "artifact_sha256": {
            str(path.relative_to(parent_dir)): _sha256(path)
            for path in sorted(parent_dir.rglob("*.json"))
            if path.name != "VERIFICATION.json"
        },
    }


def verify(run_dir: Path, *, parent_dir: Path) -> dict[str, Any]:
    parent = verify_parent_bank(parent_dir)
    if parent["status"] != "verified":
        raise ValueError("SMC confirmation parent bank replay failed")
    with verification_scope():
        replay = policy_verify.verify(run_dir, primary_dir=parent_dir)
    result = _load(run_dir / "RESULT.json")
    protocol = result.get("protocol") or {}
    extra_checks = {
        "policy_replay_verified": replay.get("status") == "verified"
        and replay.get("mismatches") == [],
        "confirmation_interface_exact": result.get("interface_version")
        == PRODUCER_INTERFACE,
        "confirmation_protocol_exact": protocol.get(
            "confirmation_protocol_sha256"
        )
        == PROTOCOL_SHA256,
        "confirmation_split_exact": protocol.get("confirmation_split_sha256")
        == CONFIRMATION_SPLIT_SHA256,
        "parent_bank_bound": protocol.get("parent_bank_result_sha256")
        == _sha256(parent_dir / "RESULT.json"),
        "seed_schedule_exact": all(
            protocol.get(key) == value
            for key, value in {
                "parent_seed_start": PARENT_SEED_START,
                "annotation_seed_start": ANNOTATION_SEED_START,
                "branch_seed_start": BRANCH_SEED_START,
                "truth_seed_start": TRUTH_SEED_START,
                "actual_first_seed_start": ACTUAL_FIRST_SEED_START,
                "actual_final_seed_start": ACTUAL_FINAL_SEED_START,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "random_seed_start": RANDOM_SEED_START,
            }.items()
        ),
        "no_optional_baseline": protocol.get("optional_baselines") == []
        and result.get("naive_baseline", {}).get("status") == "unavailable"
        and result.get("usage", {}).get("naive_luna", {}).get(
            "adapter_requests"
        )
        == 0
        and result.get("usage", {}).get("deepseek_naive_endpoint", {}).get(
            "adapter_requests"
        )
        == 0,
        "authorization_matches_status": result.get("authorizes")
        == ("confirmed_result_only" if result.get("status") == "passed" else "nothing"),
    }
    mismatches = list(replay.get("mismatches") or [])
    mismatches.extend(key for key, passed in extra_checks.items() if not passed)
    artifacts = {
        key: value
        for key, value in (replay.get("artifact_sha256") or {}).items()
        if key != "FROZEN_REPORT.json"
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "verified" if not mismatches else "verification_failed",
        "kind": "smc_dynamic_depth2_confirmation",
        "result_status": result.get("status"),
        "authorizes": result.get("authorizes") if not mismatches else "nothing",
        "model_calls": 0,
        "cost_usd": 0.0,
        "parent_bank_verification": parent,
        "policy_replay_checks": replay.get("checks"),
        "confirmation_checks": extra_checks,
        "mismatches": mismatches,
        "artifact_sha256": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir.resolve(), parent_dir=args.parent_dir.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        checkpoint(args.output.resolve(), result)
    else:
        print(rendered, end="")
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
