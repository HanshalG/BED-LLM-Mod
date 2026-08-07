#!/usr/bin/env python3
"""Run the frozen RegretBench dynamic depth-two confirmation."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


from scripts import regretbench_deepseek_dynamic_depth2_policy as policy
from scripts import regretbench_deepseek_support_recovery as recovery
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-dynamic-depth2-confirmation-1"
PROTOCOL = recovery.REPO_ROOT / (
    "results/nonmyopic/regretbench_deepseek_dynamic_depth2_confirmation/"
    "PROTOCOL_MANIFEST.json"
)
PROTOCOL_SHA256 = (
    "7a782f02eb8c3b16d5b229cca309d02bce64df6432c5090c977d3e26d1f46498"
)
PROTOCOL_VERIFICATION = recovery.REPO_ROOT / (
    "results/nonmyopic/regretbench_deepseek_dynamic_depth2_confirmation/"
    "PROTOCOL_VERIFICATION.json"
)
PROTOCOL_VERIFICATION_SHA256 = (
    "20a47fd4cde521327a1ea1c3c183a6efad0cd0b580aed6773e49829fd841ab3c"
)
PREREGISTRATION = recovery.REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_DEEPSEEK_DYNAMIC_DEPTH2_CONFIRMATION_PREREGISTRATION.md"
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
BOOTSTRAP_SAMPLES = 20_000
PLANNING_REQUESTS = 8_256
MAX_ACTUAL_REQUESTS = 512
MAX_REQUESTS = 8_768
RUN_BUDGET_USD = 3.50
PROJECTED_COST_USD = 3.10


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_protocol_binding() -> dict[str, Any]:
    policy.validate_protocol_binding()
    if recovery.sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("confirmation protocol manifest changed")
    if recovery.sha256_file(PROTOCOL_VERIFICATION) != PROTOCOL_VERIFICATION_SHA256:
        raise ValueError("confirmation protocol verification changed")
    if recovery.sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("confirmation preregistration changed")
    protocol = _load(PROTOCOL)
    verification = _load(PROTOCOL_VERIFICATION)
    if (
        protocol.get("status") != "frozen_before_development_responses"
        or protocol.get("source", {}).get("confirmation_ids_sha256")
        != CONFIRMATION_SPLIT_SHA256
        or protocol.get("instrument", {}).get("optional_baselines") != []
        or protocol.get("requests", {}).get("exact_planning") != PLANNING_REQUESTS
        or protocol.get("requests", {}).get("maximum_total") != MAX_REQUESTS
        or protocol.get("budget", {}).get("run_cap_usd") != RUN_BUDGET_USD
        or verification.get("status") != "verified_frozen_protocol"
        or verification.get("gates", {}).get("all_pass") is not True
        or verification.get("model_calls_made") != 0
    ):
        raise ValueError("confirmation protocol is not a verified frozen protocol")
    return protocol


def load_confirmation_cigs() -> list[Any]:
    manifest = recovery.validate_source_bindings()
    split = manifest["splits"]["confirmation"]
    ids = list(split["ids"])
    if len(ids) != 64 or split.get("ids_sha256") != CONFIRMATION_SPLIT_SHA256:
        raise ValueError("confirmation split binding changed")
    cigs = [
        recovery.load_cig(
            recovery.REGRETBENCH_ROOT
            / "data/OpenDomainQA/test"
            / f"{cig_id}.json"
        )
        for cig_id in ids
    ]
    if [cig.cig_id for cig in cigs] != ids:
        raise ValueError("confirmation CIG order changed")
    return cigs


@contextmanager
def confirmation_scope() -> Iterator[None]:
    """Apply only the prospectively frozen cohort and seed substitutions."""

    cigs = load_confirmation_cigs()
    original_loader = recovery.load_stage_cigs
    replacements = {
        "INITIAL_SEED_START": INITIAL_SEED_START,
        "BRANCH_SEED_START": BRANCH_SEED_START,
        "ACTUAL_FIRST_SEED_START": ACTUAL_FIRST_SEED_START,
        "ACTUAL_FINAL_SEED_START": ACTUAL_FINAL_SEED_START,
        "TRUTH_SEED_START": TRUTH_SEED_START,
        "RANDOM_SEED_START": RANDOM_SEED_START,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "BOOTSTRAP_SAMPLES": BOOTSTRAP_SAMPLES,
        "MAX_PRIMARY_DEEPSEEK_REQUESTS": MAX_REQUESTS,
        "MAX_DEEPSEEK_REQUESTS": MAX_REQUESTS,
        "MAX_REQUESTS": MAX_REQUESTS,
        "RUN_BUDGET_USD": RUN_BUDGET_USD,
        "PROJECTED_COST_USD": PROJECTED_COST_USD,
    }
    originals = {name: getattr(policy, name) for name in replacements}

    def scoped_loader(stage: str) -> list[Any]:
        if stage == "development":
            return list(cigs)
        return original_loader(stage)

    try:
        for name, value in replacements.items():
            setattr(policy, name, value)
        recovery.load_stage_cigs = scoped_loader
        yield
    finally:
        recovery.load_stage_cigs = original_loader
        for name, value in originals.items():
            setattr(policy, name, value)


def _confirmation_protocol(
    base: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(base)
    value.update(
        {
            "stage": "confirmation",
            "source_split": "confirmation",
            "confirmation_opened": True,
            "confirmation_protocol_sha256": PROTOCOL_SHA256,
            "confirmation_protocol_verification_sha256": (
                PROTOCOL_VERIFICATION_SHA256
            ),
            "confirmation_preregistration_sha256": PREREGISTRATION_SHA256,
            "confirmation_split_sha256": CONFIRMATION_SPLIT_SHA256,
            "development_authorization": dict(authorization),
            "initial_seed_start": INITIAL_SEED_START,
            "branch_seed_start": BRANCH_SEED_START,
            "actual_first_seed_start": ACTUAL_FIRST_SEED_START,
            "actual_final_seed_start": ACTUAL_FINAL_SEED_START,
            "truth_seed_start": TRUTH_SEED_START,
            "random_seed_start": RANDOM_SEED_START,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "optional_baselines": [],
            "naive_is_descriptive_only": False,
            "maximum_naive_endpoint_requests": 0,
            "maximum_naive_requests": 0,
            "maximum_primary_deepseek_requests": MAX_REQUESTS,
            "maximum_deepseek_requests": MAX_REQUESTS,
            "maximum_requests": MAX_REQUESTS,
        }
    )
    return value


def run_confirmation(
    *,
    output_dir: Path,
    run_id: str,
    support_smoke_result: Path,
    support_development_result: Path,
    policy_smoke_result: Path,
    development_authorization: Mapping[str, Any],
    adapter: policy.StructuredAdapter,
    daily_budget_status: Mapping[str, Any] | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    validate_protocol_binding()
    if (
        development_authorization.get("status") != "authorized"
        or development_authorization.get("development_status") != "passed"
        or development_authorization.get("independent_replay_status")
        != "verified"
    ):
        raise ValueError("confirmation lacks a literal verified development pass")
    with confirmation_scope():
        result = policy.run_development(
            output_dir=output_dir,
            run_id=run_id,
            support_smoke_result=support_smoke_result,
            support_development_result=support_development_result,
            policy_smoke_result=policy_smoke_result,
            naive_smoke_result=Path("disabled-confirmation-baseline"),
            adapter=adapter,
            naive_baseline_enabled=False,
            daily_budget_status=daily_budget_status,
            bootstrap_samples=bootstrap_samples,
        )
    result["schema_version"] = SCHEMA_VERSION
    result["interface_version"] = INTERFACE_VERSION
    result["protocol"] = _confirmation_protocol(
        result.get("protocol", {}), development_authorization
    )
    result["authorizes"] = "confirmed_result_only" if result["status"] == "passed" else "nothing"
    result["development_opened"] = False
    result["confirmation_opened"] = True
    result["naive_baseline"] = {
        "status": "not_in_confirmation_protocol",
        "enabled": False,
        "attempted": False,
        "can_affect_primary_status": False,
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def expected_branch_seed(task: int, hypothesis: int, draw: int) -> int:
    return BRANCH_SEED_START + task * 16 + hypothesis * 2 + draw


def expected_seed_sets() -> dict[str, set[int]]:
    return {
        "initial": {INITIAL_SEED_START + index for index in range(64)},
        "branch": {
            expected_branch_seed(task, hypothesis, draw)
            for task in range(64)
            for hypothesis in range(8)
            for draw in range(2)
        },
        "actual_first": {ACTUAL_FIRST_SEED_START + index for index in range(64)},
        "actual_final": {ACTUAL_FINAL_SEED_START + index for index in range(64)},
        "truth": {TRUTH_SEED_START + index for index in range(64)},
        "random": {RANDOM_SEED_START + index for index in range(64)},
        "bootstrap": {BOOTSTRAP_SEED},
    }


def selected_task_ids(tasks: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(task["task_id"]) for task in tasks]
