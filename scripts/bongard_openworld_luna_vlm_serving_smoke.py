#!/usr/bin/env python3
"""Run the exact-10 Luna multimodal semantic-belief serving gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.number_game_predictive_risk_replication import SeededStructuredAdapter
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-vlm-serving-smoke-2"
MODEL_ID = "openai/gpt-5.6-luna"
MODEL_SEED = 2_026_081_001
EARLIEST_DATE = "2026-08-10"
TIMEZONE = "Europe/London"
EXPECTED_REQUESTS = 10
CONCURRENCY = 10
MAX_TOKENS = 3_200
TEMPERATURE = 0.0
PROJECTED_COST_USD = 0.10
RUN_BUDGET_USD = 0.25
MIN_BRANCH_PREDICTION_MAE = 0.05


class StructuredModel(Protocol):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    task: bed.VisualTask
    history: tuple[tuple[str, bool], ...]
    kind: str
    branch_candidate_id: str | None = None
    branch_label: bool | None = None


class LunaVisionAdapter(SeededStructuredAdapter):
    """Remove unsupported sampling knobs while preserving strict JSON."""

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=disable_reasoning,
            response_format=response_format,
        )
        for key in ("temperature", "top_p", "top_k", "n"):
            payload.pop(key, None)
        payload["reasoning"] = {"enabled": False, "exclude": True}
        if response_format is not None:
            payload["provider"] = {"require_parameters": False}
        return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_smoke_cases(tasks: Sequence[bed.VisualTask]) -> list[SmokeCase]:
    selected = sorted(tasks, key=lambda task: task.task_id)[:2]
    if len(selected) != 2:
        raise ValueError("serving smoke requires at least two mechanics tasks")
    cases = []
    for task in selected:
        cases.append(
            SmokeCase(
                case_id=f"{task.task_id}-root",
                task=task,
                history=task.initial_history,
                kind="root",
            )
        )
    for task in selected:
        for candidate_id in task.candidate_ids[:2]:
            for label in (False, True):
                cases.append(
                    SmokeCase(
                        case_id=(
                            f"{task.task_id}-{candidate_id}-"
                            f"{'positive' if label else 'negative'}"
                        ),
                        task=task,
                        history=tuple(
                            sorted((*task.initial_history, (candidate_id, label)))
                        ),
                        kind="branch",
                        branch_candidate_id=candidate_id,
                        branch_label=label,
                    )
                )
    if len(cases) != EXPECTED_REQUESTS:
        raise AssertionError("serving smoke case count changed")
    return cases


def branch_sensitivity(
    negative: bed.SemanticBelief,
    positive: bed.SemanticBelief,
    *,
    candidate_id: str,
) -> dict[str, Any]:
    observed_ids = {
        image_id for image_id, _ in (*negative.history, *positive.history)
    }
    unobserved_ids = [
        image_id
        for image_id in negative.image_ids
        if image_id not in observed_ids and image_id != candidate_id
    ]
    if not unobserved_ids:
        raise ValueError("branch sensitivity has no still-unobserved images")
    prediction_mae = sum(
        abs(
            bed.predictive_probability(negative, image_id)
            - bed.predictive_probability(positive, image_id)
        )
        for image_id in unobserved_ids
    ) / len(unobserved_ids)
    negative_rules = {
        bed.canonical_rule(item.rule) for item in negative.hypotheses
    }
    positive_rules = {
        bed.canonical_rule(item.rule) for item in positive.hypotheses
    }
    union = negative_rules | positive_rules
    rule_jaccard = len(negative_rules & positive_rules) / len(union)
    material = (
        prediction_mae >= MIN_BRANCH_PREDICTION_MAE
        or rule_jaccard <= 0.8
    )
    return {
        "candidate_id": candidate_id,
        "unobserved_image_count": len(unobserved_ids),
        "unobserved_prediction_mae": prediction_mae,
        "rule_jaccard": rule_jaccard,
        "material": material,
    }


def serving_metrics(
    cases: Sequence[SmokeCase],
    beliefs: Sequence[bed.SemanticBelief],
) -> dict[str, Any]:
    by_key = {
        (case.task.task_id, case.branch_candidate_id, case.branch_label): belief
        for case, belief in zip(cases, beliefs, strict=True)
        if case.kind == "branch"
    }
    sensitivities = []
    for task in sorted({case.task.task_id: case.task for case in cases}.values(), key=lambda x: x.task_id):
        for candidate_id in task.candidate_ids[:2]:
            sensitivities.append(
                branch_sensitivity(
                    by_key[(task.task_id, candidate_id, False)],
                    by_key[(task.task_id, candidate_id, True)],
                    candidate_id=candidate_id,
                )
            )
    root_rows = []
    for case, belief in zip(cases, beliefs, strict=True):
        if case.kind != "root":
            continue
        scores = bed.candidate_eigs(belief, case.task.candidate_ids)
        root_rows.append(
            {
                "task_id": case.task.task_id,
                "max_candidate_eig": max(scores.values()),
                "unique_candidate_eig_count_1e8": len(
                    {round(value, 8) for value in scores.values()}
                ),
                "candidate_eigs": scores,
            }
        )
    return {
        "mean_observed_history_fit_log_loss": sum(
            bed.observed_history_fit_log_loss(belief) for belief in beliefs
        )
        / len(beliefs),
        "max_observed_history_fit_log_loss": max(
            bed.observed_history_fit_log_loss(belief) for belief in beliefs
        ),
        "branch_sensitivities": sensitivities,
        "roots": root_rows,
    }


def serving_gates(
    *,
    cases: Sequence[SmokeCase],
    beliefs: Sequence[bed.SemanticBelief],
    metrics: Mapping[str, Any],
    prompt_errors: Sequence[Sequence[str]],
    usage: Mapping[str, Any],
) -> dict[str, bool]:
    exact_rules = all(
        len(belief.hypotheses) == bed.NUM_HYPOTHESES
        and len({bed.canonical_rule(item.rule) for item in belief.hypotheses})
        == bed.NUM_HYPOTHESES
        for belief in beliefs
    )
    finite_history_weights = all(
        all(math.isfinite(weight) and weight > 0 for weight in belief.history_weights)
        and math.isfinite(bed.entropy(belief.history_weights))
        for belief in beliefs
    )
    gates = {
        "exact_10_accepted_requests": usage.get("adapter_requests") == EXPECTED_REQUESTS,
        "exact_10_http_attempts": usage.get("http_attempts") == EXPECTED_REQUESTS,
        "zero_retries": usage.get("retry_count") == 0,
        "zero_provider_error_retries": usage.get("provider_error_retries", 0) == 0,
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "all_10_strict_schemas_parse": len(beliefs) == len(cases) == EXPECTED_REQUESTS,
        "all_responses_have_exact_unique_hypothesis_support": exact_rules,
        "all_history_conditioned_weights_have_finite_positive_mass_and_entropy": (
            finite_history_weights
        ),
        "all_four_simulated_branch_pairs_change_unobserved_beliefs": all(
            row["material"] for row in metrics["branch_sensitivities"]
        ),
        "both_roots_have_nonzero_nonidentical_candidate_eig": all(
            row["max_candidate_eig"] > 1e-8
            and row["unique_candidate_eig_count_1e8"] >= 2
            for row in metrics["roots"]
        ),
        "all_multimodal_prompts_hide_bound_source_truth": not any(prompt_errors),
        "cost_at_most_0_25": float(usage.get("run_cost_usd", math.inf)) <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _adapter(*, output_dir: Path, run_id: str) -> LunaVisionAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=500.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return LunaVisionAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=1_000_000),
        config,
        request_seed=MODEL_SEED,
    )


def run_smoke(
    *,
    output_dir: Path,
    run_id: str,
    tasks: Sequence[bed.VisualTask] | None = None,
    adapter: StructuredModel | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "private/RAW_RESPONSES.json"
    cases = build_smoke_cases(tasks or bed.load_mechanics_tasks())
    messages = [bed.build_belief_messages(case.task, case.history) for case in cases]
    prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, message)
        for case, message in zip(cases, messages, strict=True)
    ]
    if any(prompt_errors):
        raise ValueError(f"hidden-state prompt audit failed: {prompt_errors}")
    model = adapter or _adapter(output_dir=output_dir, run_id=run_id)
    responses = model.chat_complete_messages_batched_structured(
        messages,
        temperature=TEMPERATURE,
        block_size=EXPECTED_REQUESTS,
        response_format=bed.belief_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    checkpoint(
        raw_path,
        {
            "case_ids": [case.case_id for case in cases],
            "responses": responses,
            "actual_candidate_labels_accessed": False,
            "endpoint_labels_accessed": False,
        },
    )
    beliefs = [
        bed.parse_belief_response(
            response,
            image_ids=case.task.image_ids,
            history=case.history,
        )
        for case, response in zip(cases, responses, strict=True)
    ]
    usage = summarize_usage(model.usage_snapshot())
    metrics = serving_metrics(cases, beliefs)
    gates = serving_gates(
        cases=cases,
        beliefs=beliefs,
        metrics=metrics,
        prompt_errors=prompt_errors,
        usage=usage,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gated_null",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_VLM_MECHANICS_PREREGISTRATION.md"
            ),
            "semantic_validity_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_SEMANTIC_VALIDITY_AMENDMENT.md"
            ),
            "model": MODEL_ID,
            "model_seed": MODEL_SEED,
            "reasoning": False,
            "expected_requests": EXPECTED_REQUESTS,
            "run_budget_usd": RUN_BUDGET_USD,
            "actual_candidate_labels_accessed": False,
            "endpoint_labels_accessed": False,
            "scientific_endpoint_accessed": False,
        },
        "usage": usage,
        "metrics": metrics,
        "gates": gates,
        "cases": [
            {
                "case_id": case.case_id,
                "task_id": case.task.task_id,
                "kind": case.kind,
                "history_size": len(case.history),
                "request_text_sha256": hashlib.sha256(
                    bed.request_text(message).encode()
                ).hexdigest(),
                "belief": bed.public_belief_summary(belief),
            }
            for case, message, belief in zip(cases, messages, beliefs, strict=True)
        ],
        "raw_responses_sha256": sha256_file(raw_path),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def _validate_ledger(ledger: Mapping[str, Any], now: datetime | None) -> None:
    timezone = ZoneInfo(TIMEZONE)
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    if local_now.date().isoformat() < EARLIEST_DATE:
        raise RuntimeError(f"Bongard VLM spend is forbidden before {EARLIEST_DATE}")
    if ledger.get("date") != local_now.date().isoformat():
        raise RuntimeError("daily ledger date is not current")
    if ledger.get("timezone") != TIMEZONE:
        raise RuntimeError("daily ledger timezone must be Europe/London")
    if float(ledger.get("daily_cap_usd", 0.0)) != 5.0:
        raise RuntimeError("daily ledger must enforce the exact $5 cap")
    if "bongard_luna_vlm_serving_smoke" in ledger:
        raise RuntimeError("Bongard Luna serving smoke is already recorded")


def initialize_daily_ledger(
    *,
    path: Path,
    live: Mapping[str, float],
    now: datetime | None = None,
) -> dict[str, Any]:
    timezone = ZoneInfo(TIMEZONE)
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        _validate_ledger(ledger, local_now)
        return ledger
    if local_now.date().isoformat() < EARLIEST_DATE:
        raise RuntimeError(f"Bongard VLM spend is forbidden before {EARLIEST_DATE}")
    if float(live["balance_usd"]) + 1e-12 < RUN_BUDGET_USD:
        raise RuntimeError("OpenRouter balance is below the smoke run cap")
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "date": local_now.date().isoformat(),
        "timezone": TIMEZONE,
        "daily_cap_usd": 5.0,
        "opening_total_credits_usd": float(live["total_credits_usd"]),
        "opening_total_usage_usd": float(live["total_usage_usd"]),
        "opening_balance_usd": float(live["balance_usd"]),
        "opening_frozen_at_london": local_now.isoformat(),
        "recorded_actual_spend_usd": 0.0,
        "account_wide_usage_counts_against_cap": True,
        "unspent_allowance_does_not_roll_over": True,
        "first_authorized_block": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "expected_requests": EXPECTED_REQUESTS,
            "maximum_cost_usd": RUN_BUDGET_USD,
            "status": "authorized_pending",
        },
        "additional_paid_blocks_authorized": False,
    }
    checkpoint(path, ledger)
    return ledger


def reconcile_ledger(
    *,
    ledger: Mapping[str, Any],
    measured_cost_usd: float,
    live_after: Mapping[str, float],
    status: str,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    previous = float(updated.get("recorded_actual_spend_usd", 0.0))
    posted = max(0.0, float(live_after["total_usage_usd"]) - opening)
    local = previous + measured_cost_usd
    recorded = max(posted, local)
    updated["recorded_actual_spend_usd"] = recorded
    previous_block_cost = float(
        (updated.get("bongard_luna_vlm_serving_smoke") or {}).get(
            "actual_cost_usd", 0.0
        )
    )
    block_cost = max(previous_block_cost, measured_cost_usd)
    updated["bongard_luna_vlm_serving_smoke"] = {
        "status": status,
        "actual_cost_usd": block_cost,
        "maximum_cost_usd": RUN_BUDGET_USD,
        "interface_version": INTERFACE_VERSION,
        "model": MODEL_ID,
    }
    authorization = updated.get("first_authorized_block") or {}
    if authorization.get("interface_version") == INTERFACE_VERSION:
        authorization["status"] = status
        authorization["actual_cost_usd"] = max(
            float(authorization.get("actual_cost_usd", 0.0)),
            block_cost,
        )
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live_after["total_credits_usd"]),
        "live_total_usage_usd": float(live_after["total_usage_usd"]),
        "live_balance_usd": float(live_after["balance_usd"]),
        "posted_spend_since_opening_usd": posted,
        "locally_measured_spend_usd": local,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": max(
            0.0, float(updated["daily_cap_usd"]) - recorded
        ),
    }
    return updated


def execute_smoke(
    *,
    output_dir: Path,
    run_id: str,
    ledger_path: Path,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    smoke_runner: Callable[..., dict[str, Any]] = run_smoke,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    live = live_reader()
    ledger = initialize_daily_ledger(path=ledger_path, live=live, now=now)
    _validate_ledger(ledger, now)
    require_budget(
        ledger,
        projected_cost_usd=PROJECTED_COST_USD,
        total_usage_usd=live["total_usage_usd"],
        now=now,
    )
    if live["balance_usd"] + 1e-12 < RUN_BUDGET_USD:
        raise RuntimeError("OpenRouter balance is below the smoke run cap")
    try:
        result = smoke_runner(output_dir=output_dir, run_id=run_id)
    except Exception as exc:
        reconciliation_error = None
        try:
            reconciled = reconcile_ledger(
                ledger=ledger,
                measured_cost_usd=0.0,
                live_after=live_reader(),
                status="failed_closed_posted_spend_reconciled",
            )
            checkpoint(ledger_path, reconciled)
        except Exception as reconciliation_exc:
            reconciliation_error = (
                f"{type(reconciliation_exc).__name__}: {reconciliation_exc}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint(
            output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ledger_reconciliation_error": reconciliation_error,
            },
        )
        raise
    measured_cost = float(result["usage"]["run_cost_usd"])
    locally_reconciled = reconcile_ledger(
        ledger=ledger,
        measured_cost_usd=measured_cost,
        live_after=live,
        status=result["status"],
    )
    checkpoint(ledger_path, locally_reconciled)
    live_after = live_reader()
    final_ledger = reconcile_ledger(
        ledger=locally_reconciled,
        measured_cost_usd=0.0,
        live_after=live_after,
        status=result["status"],
    )
    checkpoint(ledger_path, final_ledger)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    args = parser.parse_args()
    result = execute_smoke(
        output_dir=args.output_dir,
        run_id=args.run_id,
        ledger_path=args.daily_ledger,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "usage": result["usage"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
