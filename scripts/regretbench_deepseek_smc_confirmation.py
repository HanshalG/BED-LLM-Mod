#!/usr/bin/env python3
"""Zero-call orchestration core for the sealed RegretBench SMC confirmation."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from scripts import regretbench_deepseek_smc_dynamic_depth2_experiment as experiment
from scripts import regretbench_deepseek_smc_dynamic_depth2_policy as core
from scripts import regretbench_deepseek_support_recovery as primary
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-smc-confirmation-1"
PROTOCOL = primary.REPO_ROOT / (
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
PARENT_REQUESTS = 64
PLANNING_REQUESTS = 8_256
MAX_REALIZED_REQUESTS = 512
MAX_POLICY_REQUESTS = PLANNING_REQUESTS + MAX_REALIZED_REQUESTS
MAX_TOTAL_REQUESTS = PARENT_REQUESTS + MAX_POLICY_REQUESTS
PARENT_BUDGET_USD = 0.20
POLICY_BUDGET_USD = 3.50
RUN_BUDGET_USD = 3.70


def validate_protocol_binding() -> None:
    core.validate_protocol_binding()
    if not PROTOCOL.is_file() or primary.sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("SMC confirmation protocol changed")


def load_confirmation_cigs() -> list[Any]:
    manifest = primary.validate_source_bindings()
    split = manifest["splits"]["confirmation"]
    ids = list(split["ids"])
    if len(ids) != 64 or split.get("ids_sha256") != CONFIRMATION_SPLIT_SHA256:
        raise ValueError("SMC confirmation split changed")
    cigs = [
        primary.load_cig(
            primary.REGRETBENCH_ROOT
            / "data/OpenDomainQA/test"
            / f"{cig_id}.json"
        )
        for cig_id in ids
    ]
    if [cig.cig_id for cig in cigs] != ids:
        raise ValueError("SMC confirmation order changed")
    return cigs


def build_parent_bank(
    *,
    output_dir: Path,
    adapter: experiment.StructuredAdapter,
    daily_budget_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate only the untouched cohort's banked parent populations."""

    validate_protocol_binding()
    cigs = load_confirmation_cigs()
    messages = []
    privacy = []
    for cig in cigs:
        request, audit = primary.messages_for(cig, [])
        messages.append(request)
        privacy.append(audit)
    seeds = [PARENT_SEED_START + index for index in range(PARENT_REQUESTS)]
    raw = primary._call_batch(adapter, messages, seeds)
    supports = [primary.parse_support(value) for value in raw]
    usage = experiment.summarize_usage(adapter.usage_snapshot())
    gates = {
        "exact_64_parent_responses": len(raw) == PARENT_REQUESTS,
        "exact_64_accepted_requests": usage["adapter_requests"]
        == PARENT_REQUESTS,
        "exact_64_http_attempts": usage["http_attempts"] == PARENT_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_supports_strict_with_eight_slots_and_four_questions": all(
            support["diagnostic"]["codec_mode"] == "strict_json"
            and len(support["hypotheses"]) == core.PARTICLES
            and support["diagnostic"]["question_count"] == core.QUESTIONS
            for support in supports
        ),
        "all_privacy_audits_pass": len(privacy) == PARENT_REQUESTS
        and all(row["passed"] for row in privacy),
        "within_parent_budget": float(usage["run_cost_usd"])
        <= PARENT_BUDGET_USD + 1e-12,
    }
    gates["all_pass"] = all(gates.values())
    status = "passed" if gates["all_pass"] else "mechanics_failed"
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": "smc_confirmation_parent_bank",
        "status": status,
        "authorizes": "smc_confirmation_policy_only" if status == "passed" else "nothing",
        "protocol": {
            "protocol_sha256": PROTOCOL_SHA256,
            "confirmation_split_sha256": CONFIRMATION_SPLIT_SHA256,
            "model": primary.MODEL_ID,
            "reasoning": "disabled_excluded",
            "temperature": primary.TEMPERATURE,
            "max_output_tokens": primary.MAX_TOKENS,
            "parent_seed_start": PARENT_SEED_START,
            "expected_requests": PARENT_REQUESTS,
            "hidden_truth_accessed": False,
        },
        "daily_budget_status": dict(daily_budget_status or {}),
        "usage": usage,
        "mechanics_gates": gates,
        "task_ids": [cig.cig_id for cig in cigs],
    }
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    checkpoint(
        private / "RAW_RESPONSES.json",
        {"stage": "confirmation", "seeds": seeds, "root": raw, "branches": []},
    )
    checkpoint(
        private / "CONTROLS.json",
        {
            "stage": "confirmation",
            "roots": [
                {
                    "task_id": cig.cig_id,
                    "question": support["questions"][0],
                    "raw_parent_sha256": hashlib.sha256(value.encode()).hexdigest(),
                }
                for cig, support, value in zip(cigs, supports, raw, strict=True)
            ],
            "privacy": privacy,
        },
    )
    checkpoint(output_dir / "RESULT.json", result)
    return result


@contextmanager
def confirmation_scope() -> Iterator[None]:
    """Bind the unchanged SMC implementation to confirmation data and seeds."""

    cigs = load_confirmation_cigs()
    original_loader = primary.load_stage_cigs
    original_choose = experiment.scorer.choose_roots
    replacements = {
        "DEVELOPMENT_ANNOTATION_SEED_START": ANNOTATION_SEED_START,
        "DEVELOPMENT_BRANCH_SEED_START": BRANCH_SEED_START,
        "TRUTH_SEED_START": TRUTH_SEED_START,
        "ACTUAL_FIRST_SEED_START": ACTUAL_FIRST_SEED_START,
        "ACTUAL_FINAL_SEED_START": ACTUAL_FINAL_SEED_START,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
    }
    originals = {name: getattr(experiment, name) for name in replacements}

    def scoped_loader(stage: str) -> list[Any]:
        return list(cigs) if stage == "development" else original_loader(stage)

    def scoped_choose(*args: Any, **kwargs: Any) -> dict[str, int]:
        seed = int(kwargs["random_seed"])
        kwargs["random_seed"] = RANDOM_SEED_START + (seed - 202608330000)
        return original_choose(*args, **kwargs)

    try:
        primary.load_stage_cigs = scoped_loader
        experiment.scorer.choose_roots = scoped_choose
        for name, value in replacements.items():
            setattr(experiment, name, value)
        yield
    finally:
        primary.load_stage_cigs = original_loader
        experiment.scorer.choose_roots = original_choose
        for name, value in originals.items():
            setattr(experiment, name, value)


def validate_development_authorization(value: Mapping[str, Any]) -> None:
    if (
        value.get("status") != "authorized"
        or value.get("development_status") != "passed"
        or value.get("independent_replay_status") != "verified"
        or value.get("report_tier")
        != "smc_provisional_development_signal_confirmation_required"
    ):
        raise ValueError("SMC confirmation lacks a literal verified development pass")


def run_confirmation(
    *,
    output_dir: Path,
    parent_dir: Path,
    adapter: experiment.StructuredAdapter,
    development_authorization: Mapping[str, Any],
    daily_budget_status: Mapping[str, Any] | None = None,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    validate_protocol_binding()
    validate_development_authorization(development_authorization)
    parent = json.loads((parent_dir / "RESULT.json").read_text(encoding="utf-8"))
    if (
        parent.get("status") != "passed"
        or parent.get("authorizes") != "smc_confirmation_policy_only"
        or parent.get("protocol", {}).get("protocol_sha256") != PROTOCOL_SHA256
    ):
        raise ValueError("verified SMC confirmation parent bank is required")
    with confirmation_scope():
        tree = experiment.build_development_planning_tree(
            output_dir=output_dir,
            adapter=adapter,
            primary_development_dir=parent_dir,
        )
        realized = experiment.run_realized_primary(
            output_dir=output_dir,
            adapter=adapter,
            tree=tree,
            bootstrap_samples=bootstrap_samples,
        )
        result = experiment.finalize_development_result(
            output_dir=output_dir,
            primary_result=realized,
            policy_smoke={"status": "inherited_verified_development_instrument"},
            naive_smoke={"status": "not_in_confirmation_protocol"},
            naive_result=None,
            naive_error={"error": "baseline excluded by confirmation protocol"},
            primary_privacy=[*tree["privacy"], *realized["actual_privacy"]],
            daily_budget_status=daily_budget_status,
            bootstrap_samples=bootstrap_samples,
        )
    result["schema_version"] = SCHEMA_VERSION
    result["interface_version"] = INTERFACE_VERSION
    result["kind"] = "smc_dynamic_depth2_confirmation"
    result["authorizes"] = "confirmed_result_only" if result["status"] == "passed" else "nothing"
    result["protocol"].update(
        {
            "stage": "confirmation",
            "confirmation_opened": True,
            "confirmation_protocol_sha256": PROTOCOL_SHA256,
            "confirmation_split_sha256": CONFIRMATION_SPLIT_SHA256,
            "parent_seed_start": PARENT_SEED_START,
            "annotation_seed_start": ANNOTATION_SEED_START,
            "branch_seed_start": BRANCH_SEED_START,
            "truth_seed_start": TRUTH_SEED_START,
            "actual_first_seed_start": ACTUAL_FIRST_SEED_START,
            "actual_final_seed_start": ACTUAL_FINAL_SEED_START,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "random_seed_start": RANDOM_SEED_START,
            "parent_bank_result_sha256": primary.sha256_file(parent_dir / "RESULT.json"),
            "development_authorization": dict(development_authorization),
            "maximum_parent_requests": PARENT_REQUESTS,
            "maximum_policy_requests": MAX_POLICY_REQUESTS,
            "maximum_total_requests": MAX_TOTAL_REQUESTS,
            "optional_baselines": [],
        }
    )
    result["parent_bank"] = {
        "status": parent["status"],
        "usage": parent["usage"],
        "mechanics_gates": parent["mechanics_gates"],
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def expected_seed_sets() -> dict[str, set[int]]:
    return {
        "parent": {PARENT_SEED_START + index for index in range(64)},
        "annotation": {ANNOTATION_SEED_START + index for index in range(64)},
        "branch": {
            BRANCH_SEED_START + task * 16 + particle * 2 + draw
            for task in range(64)
            for particle in range(core.PARTICLES)
            for draw in range(experiment.BRANCH_DRAWS)
        },
        "truth": {TRUTH_SEED_START + index for index in range(64)},
        "actual_first": {ACTUAL_FIRST_SEED_START + index for index in range(64)},
        "actual_final": {ACTUAL_FINAL_SEED_START + index for index in range(64)},
        "random": {RANDOM_SEED_START + index for index in range(64)},
        "bootstrap": {BOOTSTRAP_SEED},
    }
