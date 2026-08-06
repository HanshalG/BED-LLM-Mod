#!/usr/bin/env python3
"""Run the fresh DeepSeek diversity-V2 reliability and stress gates."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_budget_model_aug7_execute as aug7
from scripts import number_game_budget_model_reliability128 as base
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-deepseek-diversity-v2-aug7-1"
MODEL = "deepseek/deepseek-v4-flash-0731"
TIMEZONE = "Europe/London"
EXPECTED_DATE = "2026-08-07"
PREREGISTRATION = REPO_ROOT / (
    "results/nonmyopic/"
    "NUMBER_GAME_DEEPSEEK_DIVERSITY_V2_AUG7_PREREGISTRATION.md"
)
PREREGISTRATION_SHA256 = (
    "3ccabbb81c13e37a0a0f48f39785ecc9535317f299f8ec7fdc70f5828e219cd0"
)
SOURCE_TREES = base.SOURCE_TREES
SOURCE_TREES_SHA256 = base.SOURCE_TREES_SHA256
OLD_LUNA_RESULT = aug7.RELIABILITY_ROOT / (
    "number-game-budget-model-reliability128-luna-20260807/RESULT.json"
)
OLD_LUNA_RESULT_SHA256 = (
    "2949ab2faac0a2e4597cec95f2d70626a724ed753a8bb12c8f189aa149026915"
)
OLD_DEEPSEEK_RESULT = aug7.RELIABILITY_ROOT / (
    "number-game-budget-model-reliability128-deepseek0731-20260807/RESULT.json"
)
OLD_DEEPSEEK_RESULT_SHA256 = (
    "c5843bbedc1780200717705281dd2cdb6a6f927492acf5cf9b2b0b0937403a1b"
)
DAILY_LEDGER = aug7.DAILY_LEDGER
INITIAL_LEDGER_SHA256 = (
    "15022d299a9ba09c7e57ce7e1097347cdf47e4650866beffd25de6ebb6f66ea7"
)
OUTPUT_ROOT = REPO_ROOT / (
    "results/nonmyopic/number_game_deepseek_diversity_v2_aug7"
)
RELIABILITY_DIR = OUTPUT_ROOT / (
    "number-game-deepseek-diversity-v2-reliability128-20260807"
)
STRESS_DIR = OUTPUT_ROOT / (
    "number-game-deepseek-diversity-v2-stress7168-20260807"
)
WRAPPER_DIR = OUTPUT_ROOT / (
    "number-game-deepseek-diversity-v2-wrapper-20260807"
)

CASE_SELECTION_SEED = 1_080_900
GATE_MODEL_SEEDS = tuple(range(1_080_901, 1_080_909))
STRESS_MODEL_SEEDS = tuple(range(1_081_201, 1_081_233))
GATE_MANIFEST_SHA256 = (
    "5252f992f2065b8f2cf983bd5721d7e07b8fdbd7d0a3c16006353ec027df4bc9"
)
STRESS_MANIFEST_SHA256 = (
    "9ef820703ebc3e046b2323e120248b59cb7aa152443e8d1645e0230b602af613"
)
GATE_REQUESTS = 128
GATE_GROUPS = 8
GATE_INITIAL = 8
GATE_ONE = 40
GATE_TWO = 80
GATE_MAX_FORMAT_RETRIES = 3
GATE_MAX_TRANSPORT_RETRIES = 4
GATE_MAX_FORCED_EXITS = 3
GATE_BUDGET_USD = 0.10
STRESS_REQUESTS = 7_168
STRESS_GROUPS = 32
STRESS_INITIAL = 32
STRESS_FRESH_CONDITIONED = 570
STRESS_REPEATED_CONDITIONED = 6_566
STRESS_MAX_FORMAT_RETRIES = 16
STRESS_MAX_TRANSPORT_RETRIES = 8
STRESS_MAX_FORCED_EXITS = 16
STRESS_BUDGET_USD = 2.10
TOTAL_AUTHORIZATION_USD = GATE_BUDGET_USD + STRESS_BUDGET_USD
CONCURRENCY = 64
MIN_INITIAL_VALID = 16
MIN_CONDITIONED_VALID = 4
MIN_MEAN_CONDITIONED_VALID = 8.0

_GRAMMAR = """Return predicates over integers n in 0..100.
Each expression must be one Python expression returning bool.
Allowed: and/or/not, integer arithmetic + - * // %, comparisons, n, and:
divisible(n, k), is_square(n), is_power_of_two(n), is_prime(n),
digit_sum(n), ends_with(n, digit).
Do not use lambdas, comprehensions, containers, indexing, attributes, imports,
or lookup tables. Avoid singleton and near-singleton memorization."""

_DIVERSITY = """Return exactly 24 objects with exactly the fields name and expression.
Use six families in this order, four hypotheses per family:
1. modular/divisibility; 2. digit-based; 3. intervals/bounds;
4. primes/squares/powers; 5. complements/disjunctions; 6. compositions.
All 24 predicates must induce different membership sets over every integer
0..100. Syntactic rewrites of the same set are duplicates and do not count.
Do not make the list 24 refinements or rewrites of one observed number."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _history_key(history: Sequence[tuple[int, bool]]) -> str:
    encoded = ";".join(f"{n}:{int(label)}" for n, label in history)
    return hashlib.sha256(
        f"{CASE_SELECTION_SEED}|{encoded}".encode("ascii")
    ).hexdigest()


def _messages(
    observations: Sequence[tuple[int, bool]],
) -> list[dict[str, str]]:
    system = {
        "role": "system",
        "content": (
            "Generate diverse executable Bayesian concept hypotheses. "
            "Return only strict JSON matching the supplied schema."
        ),
    }
    if observations:
        observed = "\n".join(
            f"- n={number} MUST evaluate to {label}."
            for number, label in observations
        )
        task = (
            "The Number Game observations are hard executable constraints:\n"
            f"{observed}\n"
            "Evaluate every proposed expression at every observed n before "
            "returning it. A positive label requires True and a negative label "
            "requires False. Any contradiction makes the item unusable."
        )
    else:
        task = "No Number Game labels have been observed."
    return [
        system,
        {
            "role": "user",
            "content": f"{task}\n\n{_DIVERSITY}\n\n{_GRAMMAR}",
        },
    ]


def _source_histories(
    source_path: Path = SOURCE_TREES,
) -> tuple[set[tuple[tuple[int, bool], ...]], set[tuple[tuple[int, bool], ...]]]:
    if _sha256(source_path) != SOURCE_TREES_SHA256:
        raise ValueError("source TREES hash changed")
    trees = _load(source_path).get("trees") or []
    if len(trees) != 32:
        raise ValueError("source must contain exactly 32 trees")
    one = {
        base._parse_first_history(key)
        for tree in trees
        for key in (tree.get("first_branches") or {})
    }
    two = {
        base._parse_second_history(key)
        for tree in trees
        for key in (tree.get("second_branches") or {})
    }
    return one, two


def _old_gate_histories(
    source_path: Path = SOURCE_TREES,
) -> set[tuple[tuple[int, bool], ...]]:
    return {
        case["observations"]
        for case in base.build_cases(source_path)
        if case["observations"]
    }


def _new_gate_histories(
    source_path: Path = SOURCE_TREES,
) -> tuple[list[tuple[tuple[int, bool], ...]], list[tuple[tuple[int, bool], ...]]]:
    one, two = _source_histories(source_path)
    old = _old_gate_histories(source_path)
    selected_one = sorted(one - old, key=_history_key)[:GATE_ONE]
    selected_two = sorted(two - old, key=_history_key)[:GATE_TWO]
    if len(selected_one) != GATE_ONE or len(selected_two) != GATE_TWO:
        raise ValueError("insufficient fresh V2 reliability histories")
    return selected_one, selected_two


def _manifest(cases: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    public = [
        {
            key: (
                [[n, label] for n, label in case[key]]
                if key == "observations"
                else case[key]
            )
            for key in fields
        }
        for case in cases
    ]
    encoded = json.dumps(
        public, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_gate_cases(
    source_path: Path = SOURCE_TREES,
) -> list[dict[str, Any]]:
    one, two = _new_gate_histories(source_path)
    cases = []
    for seed_group in range(GATE_GROUPS):
        histories = [
            tuple(),
            *one[seed_group * 5 : (seed_group + 1) * 5],
            *two[seed_group * 10 : (seed_group + 1) * 10],
        ]
        for history in histories:
            cases.append(
                {
                    "case_index": len(cases),
                    "seed_group": seed_group,
                    "observations": history,
                    "messages": _messages(history),
                }
            )
    fields = ("case_index", "seed_group", "observations")
    if len(cases) != GATE_REQUESTS or _manifest(cases, fields) != GATE_MANIFEST_SHA256:
        raise AssertionError("V2 reliability case manifest changed")
    return cases


def build_stress_cases(
    source_path: Path = SOURCE_TREES,
) -> list[dict[str, Any]]:
    one, two = _source_histories(source_path)
    old = _old_gate_histories(source_path)
    new_one, new_two = _new_gate_histories(source_path)
    excluded = old | set(new_one) | set(new_two)
    fresh = sorted((one | two) - excluded, key=_history_key)
    if len(fresh) != STRESS_FRESH_CONDITIONED:
        raise ValueError("fresh V2 stress history inventory changed")

    records = [
        {
            "seed_group": group,
            "observations": tuple(),
            "history_role": "initial",
            "history_index": None,
            "replicate": 0,
        }
        for group in range(STRESS_GROUPS)
    ]
    for index, history in enumerate(fresh):
        records.append(
            {
                "seed_group": index % STRESS_GROUPS,
                "observations": history,
                "history_role": "fresh_conditioned",
                "history_index": index,
                "replicate": 0,
            }
        )
    quotas = [
        223
        - sum(index % STRESS_GROUPS == group for index in range(len(fresh)))
        for group in range(STRESS_GROUPS)
    ]
    used_groups = {
        index: {index % STRESS_GROUPS} for index in range(len(fresh))
    }
    for repeat_index in range(STRESS_REPEATED_CONDITIONED):
        index = repeat_index % len(fresh)
        replicate = repeat_index // len(fresh) + 1
        candidates = [
            group
            for group in range(STRESS_GROUPS)
            if group not in used_groups[index] and quotas[group] > 0
        ]
        if not candidates:
            raise AssertionError("could not balance V2 stress repeats")
        start = (index * 7 + replicate * 11) % STRESS_GROUPS
        group = min(
            candidates,
            key=lambda candidate: (
                -quotas[candidate],
                (candidate - start) % STRESS_GROUPS,
            ),
        )
        quotas[group] -= 1
        used_groups[index].add(group)
        records.append(
            {
                "seed_group": group,
                "observations": fresh[index],
                "history_role": "fresh_conditioned_repeat",
                "history_index": index,
                "replicate": replicate,
            }
        )
    if any(quotas):
        raise AssertionError("V2 stress group quotas were not exhausted")
    for case_index, record in enumerate(records):
        record["case_index"] = case_index
        record["messages"] = _messages(record["observations"])
    fields = (
        "case_index",
        "seed_group",
        "observations",
        "history_role",
        "history_index",
        "replicate",
    )
    if (
        len(records) != STRESS_REQUESTS
        or _manifest(records, fields) != STRESS_MANIFEST_SHA256
        or any(
            sum(record["seed_group"] == group for record in records) != 224
            for group in range(STRESS_GROUPS)
        )
    ):
        raise AssertionError("V2 stress case manifest changed")
    return records


def _protocol(*, stage: str) -> dict[str, Any]:
    gate = stage == "reliability128"
    return {
        "interface_version": INTERFACE_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "stage": stage,
        "model": MODEL,
        "model_seeds": list(GATE_MODEL_SEEDS if gate else STRESS_MODEL_SEEDS),
        "case_manifest_sha256": (
            GATE_MANIFEST_SHA256 if gate else STRESS_MANIFEST_SHA256
        ),
        "temperature": base.TEMPERATURE,
        "reasoning": False,
        "max_output_tokens": base.MAX_TOKENS,
        "aggregate_concurrency": CONCURRENCY,
        "initial_requests": GATE_REQUESTS if gate else STRESS_REQUESTS,
        "run_budget_usd": GATE_BUDGET_USD if gate else STRESS_BUDGET_USD,
        "efficacy_accessed": False,
    }


def _usage(adapters: Sequence[Any]) -> dict[str, Any]:
    return base.aggregate_usage(adapters)


def _records_from_responses(
    *,
    cases: Sequence[dict[str, Any]],
    initial_responses: Sequence[str],
    retry_responses: Sequence[dict[str, Any]],
    max_format_retries: int,
) -> tuple[list[dict[str, Any]], int, int]:
    if len(initial_responses) != len(cases):
        raise ValueError("initial response count changed")
    parsed = []
    failed_indices = []
    for case, response in zip(cases, initial_responses, strict=True):
        support, diagnostic, error = base._safe_parse(
            response, observations=case["observations"]
        )
        parsed.append([support, diagnostic, error])
        if error is not None:
            failed_indices.append(case["case_index"])
    expected_retry_indices = (
        failed_indices
        if 0 < len(failed_indices) <= max_format_retries
        else []
    )
    observed_retry_indices = []
    for retry in retry_responses:
        if (
            not isinstance(retry, dict)
            or set(retry) != {"case_index", "response"}
            or not isinstance(retry["case_index"], int)
            or not isinstance(retry["response"], str)
        ):
            raise ValueError("retry response shape changed")
        index = retry["case_index"]
        observed_retry_indices.append(index)
        parsed[index] = list(
            base._safe_parse(
                retry["response"], observations=cases[index]["observations"]
            )
        )
    if observed_retry_indices != expected_retry_indices:
        raise ValueError("format retry indices changed")

    records = []
    retry_set = set(observed_retry_indices)
    for case, (support, diagnostic, final_error) in zip(
        cases, parsed, strict=True
    ):
        record = {
            "case_index": case["case_index"],
            "seed_group": case["seed_group"],
            "observations": [list(item) for item in case["observations"]],
            "initial_parse_failed": case["case_index"] in failed_indices,
            "format_retried": case["case_index"] in retry_set,
            "final_parse_error": final_error,
            "valid_unique_count": len(support or []),
            "diagnostics": diagnostic,
            "extension_sha256s": [
                hypothesis.public_dict()["extension_sha256"]
                for hypothesis in (support or [])
            ],
        }
        for key in ("history_role", "history_index", "replicate"):
            if key in case:
                record[key] = case[key]
        records.append(record)
    return records, len(failed_indices), len(observed_retry_indices)


def _gates(
    *,
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    expected_requests: int,
    expected_initial: int,
    expected_conditioned: int,
    max_format_retries: int,
    max_transport_retries: int,
    max_forced_exits: int,
    budget_usd: float,
    initial_parse_failures: int,
    format_retry_requests: int,
) -> dict[str, bool]:
    initial = [record for record in records if not record["observations"]]
    conditioned = [record for record in records if record["observations"]]
    conditioned_mean = sum(
        record["valid_unique_count"] for record in conditioned
    ) / len(conditioned)
    expected_accepted = expected_requests + format_retry_requests
    gates = {
        "exact_final_case_records": len(records) == expected_requests,
        "initial_parse_failures_within_cap": (
            initial_parse_failures <= max_format_retries
        ),
        "all_parse_failures_retried_once": (
            format_retry_requests == initial_parse_failures
        ),
        "all_final_responses_strictly_parsed": all(
            record["final_parse_error"] is None for record in records
        ),
        "accepted_request_accounting_exact": (
            usage["adapter_requests"] == expected_accepted
        ),
        "http_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "transport_retries_within_cap": (
            usage["retry_count"] <= max_transport_retries
        ),
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "forced_exits_within_cap": usage["forced_exits"] <= max_forced_exits,
        "every_parsed_response_has_24_schema_items": all(
            record["diagnostics"] is not None
            and record["diagnostics"].get("raw_count") == 24
            for record in records
        ),
        "all_initial_supports_have_at_least_16_valid": (
            len(initial) == expected_initial
            and all(
                record["valid_unique_count"] >= MIN_INITIAL_VALID
                for record in initial
            )
        ),
        "all_conditioned_supports_have_at_least_4_valid": (
            len(conditioned) == expected_conditioned
            and all(
                record["valid_unique_count"] >= MIN_CONDITIONED_VALID
                for record in conditioned
            )
        ),
        "mean_conditioned_support_at_least_8_valid": (
            conditioned_mean >= MIN_MEAN_CONDITIONED_VALID
        ),
        "within_stage_budget": usage["run_cost_usd"] <= budget_usd,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _run_stage(
    *,
    output_dir: Path,
    run_id: str,
    stage: str,
    cases: Sequence[dict[str, Any]],
    model_seeds: Sequence[int],
    max_format_retries: int,
    max_transport_retries: int,
    max_forced_exits: int,
    budget_usd: float,
    adapters: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"stage output is not pristine: {output_dir}")
    output_dir.mkdir(parents=True)
    private_dir = output_dir / "private"
    private_dir.mkdir()
    adapters = list(adapters or [])
    if not adapters:
        adapters = [
            base._adapter(
                model=MODEL,
                run_id=run_id,
                output_dir=output_dir,
                request_seed=seed,
                concurrency=CONCURRENCY // len(model_seeds),
                projected_cost=budget_usd,
                run_budget_usd=budget_usd,
            )
            for seed in model_seeds
        ]
    if len(adapters) != len(model_seeds):
        raise ValueError("adapter seed-group count changed")
    grouped = [
        [case for case in cases if case["seed_group"] == group]
        for group in range(len(model_seeds))
    ]
    initial_by_index = base._call_seed_groups(
        adapters=adapters, grouped_cases=grouped
    )
    if len(initial_by_index) != len(cases):
        raise ValueError("adapter omitted initial responses")
    initial = [initial_by_index[index] for index in range(len(cases))]
    failed_indices = []
    for case, response in zip(cases, initial, strict=True):
        _, _, error = base._safe_parse(
            response, observations=case["observations"]
        )
        if error is not None:
            failed_indices.append(case["case_index"])
    retries = []
    if 0 < len(failed_indices) <= max_format_retries:
        retry_groups = [
            [
                cases[index]
                for index in failed_indices
                if cases[index]["seed_group"] == group
            ]
            for group in range(len(model_seeds))
        ]
        retry_by_index = base._call_seed_groups(
            adapters=adapters, grouped_cases=retry_groups
        )
        retries = [
            {"case_index": index, "response": retry_by_index[index]}
            for index in failed_indices
        ]
    raw = {"initial_responses": initial, "retry_responses": retries}
    raw_path = private_dir / "RAW_RESPONSES.json"
    checkpoint(raw_path, raw)
    records, initial_failures, retry_count = _records_from_responses(
        cases=cases,
        initial_responses=initial,
        retry_responses=retries,
        max_format_retries=max_format_retries,
    )
    usage = _usage(adapters)
    expected_initial = GATE_INITIAL if stage == "reliability128" else STRESS_INITIAL
    gates = _gates(
        records=records,
        usage=usage,
        expected_requests=len(cases),
        expected_initial=expected_initial,
        expected_conditioned=len(cases) - expected_initial,
        max_format_retries=max_format_retries,
        max_transport_retries=max_transport_retries,
        max_forced_exits=max_forced_exits,
        budget_usd=budget_usd,
        initial_parse_failures=initial_failures,
        format_retry_requests=retry_count,
    )
    conditioned = [r["valid_unique_count"] for r in records if r["observations"]]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gated_null",
        "decision": (
            "eligible_for_conditional_stress7168"
            if stage == "reliability128" and gates["all_pass"]
            else "eligible_for_separately_preregistered_policy_development"
            if stage == "stress7168" and gates["all_pass"]
            else "close_v2_interface_at_current_stage"
        ),
        "protocol": _protocol(stage=stage),
        "usage": usage,
        "initial_parse_failures": initial_failures,
        "format_retry_requests": retry_count,
        "conditioned_valid_summary": {
            "minimum": min(conditioned),
            "mean": sum(conditioned) / len(conditioned),
            "maximum": max(conditioned),
        },
        "gates": gates,
        "cases": records,
        "raw_responses_sha256": _sha256(raw_path),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def replay_stage(
    *, result_path: Path, cases: Sequence[dict[str, Any]], stage: str
) -> dict[str, Any]:
    raw_path = result_path.parent / "private/RAW_RESPONSES.json"
    result = _load(result_path)
    raw = _load(raw_path)
    gate = stage == "reliability128"
    max_format = GATE_MAX_FORMAT_RETRIES if gate else STRESS_MAX_FORMAT_RETRIES
    records, failures, retries = _records_from_responses(
        cases=cases,
        initial_responses=raw["initial_responses"],
        retry_responses=raw["retry_responses"],
        max_format_retries=max_format,
    )
    usage = result.get("usage") or {}
    gates = _gates(
        records=records,
        usage=usage,
        expected_requests=GATE_REQUESTS if gate else STRESS_REQUESTS,
        expected_initial=GATE_INITIAL if gate else STRESS_INITIAL,
        expected_conditioned=(
            GATE_REQUESTS - GATE_INITIAL if gate else STRESS_REQUESTS - STRESS_INITIAL
        ),
        max_format_retries=max_format,
        max_transport_retries=(
            GATE_MAX_TRANSPORT_RETRIES if gate else STRESS_MAX_TRANSPORT_RETRIES
        ),
        max_forced_exits=GATE_MAX_FORCED_EXITS if gate else STRESS_MAX_FORCED_EXITS,
        budget_usd=GATE_BUDGET_USD if gate else STRESS_BUDGET_USD,
        initial_parse_failures=failures,
        format_retry_requests=retries,
    )
    conditioned = [r["valid_unique_count"] for r in records if r["observations"]]
    expected = {
        "protocol": _protocol(stage=stage),
        "initial_parse_failures": failures,
        "format_retry_requests": retries,
        "conditioned_valid_summary": {
            "minimum": min(conditioned),
            "mean": sum(conditioned) / len(conditioned),
            "maximum": max(conditioned),
        },
        "gates": gates,
        "cases": records,
        "raw_responses_sha256": _sha256(raw_path),
        "status": "passed" if gates["all_pass"] else "gated_null",
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{stage} result does not replay exactly")
    return {
        "verified": True,
        "stage": stage,
        "status": result["status"],
        "requests": int(usage["adapter_requests"]),
        "cost_usd": float(usage["run_cost_usd"]),
        "result_sha256": _sha256(result_path),
        "raw_responses_sha256": _sha256(raw_path),
    }


def _authorization_entries() -> list[dict[str, Any]]:
    return [
        {
            "stage": "reliability128",
            "model": MODEL,
            "status": "authorized_pending",
            "maximum_cost_usd": GATE_BUDGET_USD,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "efficacy_used_for_authorization": False,
        },
        {
            "stage": "stress7168",
            "model": MODEL,
            "status": "waiting_for_reliability_pass",
            "maximum_cost_usd": STRESS_BUDGET_USD,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "efficacy_used_for_authorization": False,
        },
    ]


def _validate_frozen_inputs() -> None:
    expected = {
        PREREGISTRATION: PREREGISTRATION_SHA256,
        SOURCE_TREES: SOURCE_TREES_SHA256,
        OLD_LUNA_RESULT: OLD_LUNA_RESULT_SHA256,
        OLD_DEEPSEEK_RESULT: OLD_DEEPSEEK_RESULT_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"frozen V2 input changed: {path}")
    build_gate_cases()
    build_stress_cases()


def _authorize(ledger_path: Path = DAILY_LEDGER) -> dict[str, Any]:
    ledger = _load(ledger_path)
    existing = ledger.get("deepseek_diversity_v2_blocks")
    if existing:
        expected = {item["stage"]: item for item in _authorization_entries()}
        observed = {item.get("stage"): item for item in existing}
        if set(observed) != set(expected) or any(
            item.get("model") != MODEL
            or item.get("preregistration_sha256") != PREREGISTRATION_SHA256
            or float(item.get("maximum_cost_usd", -1.0))
            != expected[stage]["maximum_cost_usd"]
            for stage, item in observed.items()
        ):
            raise RuntimeError("another V2 authorization exists")
        return ledger
    if _sha256(ledger_path) != INITIAL_LEDGER_SHA256:
        raise ValueError("pre-V2 Aug 7 ledger changed")
    remaining = float(ledger["reconciliation"]["remaining_daily_allowance_usd"])
    if remaining + 1e-12 < TOTAL_AUTHORIZATION_USD:
        raise RuntimeError("remaining Aug 7 allowance cannot fund V2")
    for path in (RELIABILITY_DIR, STRESS_DIR, WRAPPER_DIR):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"V2 paid path is not pristine: {path}")
    updated = json.loads(json.dumps(ledger))
    updated["deepseek_diversity_v2_blocks"] = _authorization_entries()
    updated["additional_paid_blocks_authorized"] = True
    updated["deepseek_diversity_v2_preregistration_sha256"] = (
        PREREGISTRATION_SHA256
    )
    checkpoint(ledger_path, updated)
    return updated


def _reconcile(
    *,
    ledger: dict[str, Any],
    stage: str,
    measured_cost_usd: float,
    live_after: dict[str, float],
    status: str,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    previous = float(updated["recorded_actual_spend_usd"])
    posted = max(0.0, float(live_after["total_usage_usd"]) - opening)
    local = previous + measured_cost_usd
    recorded = max(posted, local)
    updated["recorded_actual_spend_usd"] = recorded
    for item in updated["deepseek_diversity_v2_blocks"]:
        if item["stage"] == stage:
            item["status"] = status
            item["actual_cost_usd"] = max(
                float(item.get("actual_cost_usd", 0.0)), measured_cost_usd
            )
        elif (
            stage == "reliability128"
            and status.startswith("failed_closed")
            and item["stage"] == "stress7168"
        ):
            item["status"] = "not_authorized_reliability_failure"
            item["actual_cost_usd"] = 0.0
    updated["additional_paid_blocks_authorized"] = any(
        item["status"] in {"authorized_pending", "waiting_for_reliability_pass"}
        for item in updated["deepseek_diversity_v2_blocks"]
    )
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live_after["total_credits_usd"]),
        "live_total_usage_usd": float(live_after["total_usage_usd"]),
        "live_balance_usd": float(live_after["balance_usd"]),
        "posted_spend_since_opening_usd": posted,
        "locally_measured_spend_usd": local,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": max(0.0, 5.0 - recorded),
    }
    return updated


def _execute_paid_stage(
    *,
    stage: str,
    output_dir: Path,
    run_id: str,
    cases: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    max_format_retries: int,
    max_transport_retries: int,
    max_forced_exits: int,
    budget_usd: float,
    ledger_path: Path,
    live_reader: Callable[[], dict[str, float]],
    now: datetime,
) -> dict[str, Any]:
    result_path = output_dir / "RESULT.json"
    if result_path.is_file():
        return replay_stage(result_path=result_path, cases=cases, stage=stage)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"partial V2 stage cannot repeat: {output_dir}")
    ledger = _load(ledger_path)
    matches = [
        item
        for item in ledger["deepseek_diversity_v2_blocks"]
        if item["stage"] == stage
    ]
    if len(matches) != 1 or matches[0]["status"] not in {
        "authorized_pending",
        "waiting_for_reliability_pass",
    }:
        raise RuntimeError(f"{stage} is not authorized")
    live_before = live_reader()
    require_budget(
        ledger,
        projected_cost_usd=budget_usd,
        total_usage_usd=float(live_before["total_usage_usd"]),
        now=now,
    )
    if float(live_before["balance_usd"]) + 1e-12 < budget_usd:
        raise RuntimeError(f"OpenRouter balance cannot fund {stage}")
    try:
        result = _run_stage(
            output_dir=output_dir,
            run_id=run_id,
            stage=stage,
            cases=cases,
            model_seeds=seeds,
            max_format_retries=max_format_retries,
            max_transport_retries=max_transport_retries,
            max_forced_exits=max_forced_exits,
            budget_usd=budget_usd,
        )
    except Exception:
        try:
            live_failure = live_reader()
            checkpoint(
                ledger_path,
                _reconcile(
                    ledger=ledger,
                    stage=stage,
                    measured_cost_usd=0.0,
                    live_after=live_failure,
                    status="failed_closed_posted_spend_reconciled",
                ),
            )
        except Exception:
            pass
        raise
    local = _reconcile(
        ledger=ledger,
        stage=stage,
        measured_cost_usd=float(result["usage"]["run_cost_usd"]),
        live_after=live_before,
        status=result["status"],
    )
    checkpoint(ledger_path, local)
    checkpoint(
        ledger_path,
        _reconcile(
            ledger=local,
            stage=stage,
            measured_cost_usd=0.0,
            live_after=live_reader(),
            status=result["status"],
        ),
    )
    return replay_stage(result_path=result_path, cases=cases, stage=stage)


def preflight(
    *,
    ledger_path: Path = DAILY_LEDGER,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    now: datetime | None = None,
) -> dict[str, Any]:
    local_now = (now or datetime.now(tz=ZoneInfo(TIMEZONE))).astimezone(
        ZoneInfo(TIMEZONE)
    )
    if local_now.date().isoformat() != EXPECTED_DATE:
        raise RuntimeError("V2 is restricted to Aug 7")
    _validate_frozen_inputs()
    ledger = _load(ledger_path)
    if ledger.get("deepseek_diversity_v2_blocks"):
        raise RuntimeError("V2 preflight requires unopened authorization")
    if _sha256(ledger_path) != INITIAL_LEDGER_SHA256:
        raise ValueError("pre-V2 Aug 7 ledger changed")
    for path in (RELIABILITY_DIR, STRESS_DIR, WRAPPER_DIR):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"V2 paid path is not pristine: {path}")
    live = live_reader()
    require_budget(
        ledger,
        projected_cost_usd=TOTAL_AUTHORIZATION_USD,
        total_usage_usd=float(live["total_usage_usd"]),
        now=local_now,
    )
    if float(live["balance_usd"]) + 1e-12 < TOTAL_AUTHORIZATION_USD:
        raise RuntimeError("OpenRouter balance cannot fund full V2 authorization")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "gate_manifest_sha256": GATE_MANIFEST_SHA256,
        "stress_manifest_sha256": STRESS_MANIFEST_SHA256,
        "recorded_actual_spend_usd": ledger["recorded_actual_spend_usd"],
        "remaining_daily_allowance_usd": ledger["reconciliation"][
            "remaining_daily_allowance_usd"
        ],
        "maximum_new_cost_usd": TOTAL_AUTHORIZATION_USD,
        "post_authorization_slack_usd": (
            float(ledger["reconciliation"]["remaining_daily_allowance_usd"])
            - TOTAL_AUTHORIZATION_USD
        ),
        "live_credits": live,
        "model_calls_made": 0,
        "files_written": 0,
        "efficacy_accessed": False,
    }


def execute(
    *,
    ledger_path: Path = DAILY_LEDGER,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    now: datetime | None = None,
) -> dict[str, Any]:
    local_now = (now or datetime.now(tz=ZoneInfo(TIMEZONE))).astimezone(
        ZoneInfo(TIMEZONE)
    )
    if local_now.date().isoformat() != EXPECTED_DATE:
        raise RuntimeError("V2 executor is restricted to Aug 7")
    wrapper_path = WRAPPER_DIR / "RESULT.json"
    if wrapper_path.is_file():
        return _load(wrapper_path)
    if WRAPPER_DIR.exists() and any(WRAPPER_DIR.iterdir()):
        raise FileExistsError("partial V2 wrapper cannot repeat")
    _validate_frozen_inputs()
    _authorize(ledger_path)
    gate_cases = build_gate_cases()
    gate = _execute_paid_stage(
        stage="reliability128",
        output_dir=RELIABILITY_DIR,
        run_id="number-game-deepseek-diversity-v2-reliability128-20260807",
        cases=gate_cases,
        seeds=GATE_MODEL_SEEDS,
        max_format_retries=GATE_MAX_FORMAT_RETRIES,
        max_transport_retries=GATE_MAX_TRANSPORT_RETRIES,
        max_forced_exits=GATE_MAX_FORCED_EXITS,
        budget_usd=GATE_BUDGET_USD,
        ledger_path=ledger_path,
        live_reader=live_reader,
        now=local_now,
    )
    stress = None
    if gate["status"] == "passed":
        stress = _execute_paid_stage(
            stage="stress7168",
            output_dir=STRESS_DIR,
            run_id="number-game-deepseek-diversity-v2-stress7168-20260807",
            cases=build_stress_cases(),
            seeds=STRESS_MODEL_SEEDS,
            max_format_retries=STRESS_MAX_FORMAT_RETRIES,
            max_transport_retries=STRESS_MAX_TRANSPORT_RETRIES,
            max_forced_exits=STRESS_MAX_FORCED_EXITS,
            budget_usd=STRESS_BUDGET_USD,
            ledger_path=ledger_path,
            live_reader=live_reader,
            now=local_now,
        )
    else:
        ledger = _load(ledger_path)
        for item in ledger["deepseek_diversity_v2_blocks"]:
            if item["stage"] == "stress7168":
                item["status"] = "not_authorized_reliability_null"
                item["actual_cost_usd"] = 0.0
        ledger["additional_paid_blocks_authorized"] = False
        checkpoint(ledger_path, ledger)
        STRESS_DIR.mkdir(parents=True, exist_ok=True)
        stress_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "gated_null",
            "decision": "not_authorized_reliability_null",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "stage": "stress7168",
                "model_calls": 0,
                "run_budget_usd": STRESS_BUDGET_USD,
            },
            "usage": {"adapter_requests": 0, "run_cost_usd": 0.0},
            "reliability_result_sha256": gate["result_sha256"],
        }
        checkpoint(STRESS_DIR / "RESULT.json", stress_payload)
        stress = {
            "verified": True,
            "stage": "stress7168",
            "status": "gated_null",
            "requests": 0,
            "cost_usd": 0.0,
            "result_sha256": _sha256(STRESS_DIR / "RESULT.json"),
        }
    ledger = _load(ledger_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "reliability_verification": gate,
        "stress_verification": stress,
        "recorded_actual_spend_usd": ledger["recorded_actual_spend_usd"],
        "remaining_daily_allowance_usd": ledger["reconciliation"][
            "remaining_daily_allowance_usd"
        ],
        "efficacy_accessed": False,
        "authorizes_aug8_diversity": False,
    }
    WRAPPER_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint(wrapper_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = preflight() if args.preflight else execute()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
