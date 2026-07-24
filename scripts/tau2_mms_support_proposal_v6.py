#!/usr/bin/env python3
"""Test LLM semantic support proposal with exact Tau2 MMS likelihoods."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.tau2_mms_prerequisite_ranking_gate import (
    ROOT_ACTIONS,
    STATUS_FIELDS,
    STATUS_VALUES,
    UNLOCKED_ACTION,
    _load_tau2_tasks,
    _task_index,
    best_two_step_information,
    exact_action_values,
    exact_observations_for_backbone,
    information_gain,
)


SCHEMA_VERSION = 6
SELECTION_SEED = 24322
HYPOTHESIS_COUNT = 6
PERMISSION_WORLDS = (
    "break_app_sms_permission",
    "break_app_storage_permission",
    "break_app_both_permissions",
)
WORLD_PERMISSION_STATES = {
    "break_app_sms_permission": ("faulty", "normal"),
    "break_app_storage_permission": ("normal", "faulty"),
    "break_app_both_permissions": ("faulty", "faulty"),
}
OBSERVED_ACTIONS = tuple(
    action for action in ROOT_ACTIONS if action != "installed_apps"
)

SMOKE_PROMPTS = (
    "MMS still fails after the standard phone and network checks below.",
    "Picture messaging remains unavailable despite the completed diagnostics.",
)
FORMAL_PROMPTS = (
    "The user still cannot send MMS after ordinary connectivity checks.",
    "Group and picture messages continue to fail after device diagnostics.",
    "MMS remains broken after the listed network and handset checks.",
    "The messaging failure persists despite normal direct diagnostic results.",
    "Picture-message delivery still fails after routine phone checks.",
    "The completed device checks have not explained the MMS outage.",
    "MMS is still unavailable after direct network diagnostics.",
    "The phone cannot send picture messages despite the recorded checks.",
    "Standard diagnostics are complete, but multimedia messaging still fails.",
    "The customer still cannot use MMS after the direct tests below.",
    "The handset's picture messaging remains unresolved after routine checks.",
    "Multimedia messages continue to fail after the completed diagnostics.",
)
CONFIRMATION_PROMPTS = (
    "MMS remains unavailable after the recorded handset checks.",
    "Picture messaging still fails following standard diagnostics.",
    "The direct phone tests are complete, but MMS is unresolved.",
    "Multimedia messaging cannot be used after routine network checks.",
    "The customer still cannot send picture messages after diagnostics.",
    "MMS delivery remains broken despite the completed phone checks.",
    "Routine connectivity tests did not resolve the multimedia-message issue.",
    "The handset still cannot send MMS after the listed observations.",
    "Picture and group messages remain unavailable after direct checks.",
    "The messaging problem persists after ordinary handset diagnostics.",
    "MMS is unresolved even though the direct tests are complete.",
    "The customer reports continued picture-message failure after checks.",
    "Multimedia messages still cannot be sent after routine diagnostics.",
    "The completed network checks did not identify the remaining MMS fault.",
    "The phone's picture messaging is still unavailable after testing.",
    "MMS continues to fail following the recorded direct observations.",
    "The direct device diagnostics are complete, but messaging is unresolved.",
    "Picture-message service remains broken after standard checks.",
    "The handset still cannot use MMS after ordinary diagnostics.",
    "The remaining multimedia-message fault was not found by direct tests.",
    "MMS remains nonfunctional after the listed connectivity checks.",
    "The standard phone observations did not resolve picture messaging.",
    "The customer still has an MMS outage after routine diagnostics.",
    "Multimedia messaging remains unavailable following direct testing.",
)


def stage_prompts(stage: str) -> tuple[str, ...]:
    if stage == "serving_smoke":
        return SMOKE_PROMPTS
    if stage == "formal":
        return FORMAL_PROMPTS
    if stage == "confirmation":
        return CONFIRMATION_PROMPTS
    raise ValueError("stage must be serving_smoke, formal, or confirmation")


def common_observed_history(
    observations: dict[str, dict[str, str]],
) -> dict[str, str]:
    history: dict[str, str] = {}
    for action in OBSERVED_ACTIONS:
        values = {observations[world][action] for world in observations}
        if len(values) != 1:
            raise ValueError(f"{action} is not common across permission worlds")
        history[action] = next(iter(values))
    return history


def support_messages(
    ticket: str,
    observed_history: dict[str, str],
) -> list[dict[str, str]]:
    schema = {
        "hypotheses": [
            {
                "id": f"h{index + 1}",
                "description": "short concrete remaining MMS failure mechanism",
                **{field: "normal|faulty|unknown" for field in STATUS_FIELDS},
            }
            for index in range(HYPOTHESIS_COUNT)
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "Generate a diverse semantic differential from observed tool "
                "results. Return strict JSON only, with no reasoning text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Ticket: {ticket}\n"
                "The following diagnostic actions have already been executed. "
                "Treat their literal results as fixed evidence:\n"
                f"{json.dumps(observed_history, separators=(',', ':'))}\n"
                "Generate exactly six distinct remaining MMS failure hypotheses. "
                "Each row must describe its own concrete predicted status for "
                "network mode, Wi-Fi Calling, MMSC/APN, messaging-app SMS "
                "permission, and messaging-app storage permission. Use unknown "
                "only when the evidence genuinely leaves that field unresolved. "
                "Include diverse app-permission explanations when direct phone "
                "and network checks do not distinguish them. Do not identify a "
                "true hypothesis, choose a next action, calculate a score, or "
                "invent an observation that was not supplied. Return this exact "
                "schema in the supplied order:\n"
                + json.dumps(schema, separators=(",", ":"))
            ),
        },
    ]


def parse_support(text: str) -> list[dict[str, str]]:
    payload = _parse_json_object(text)
    rows = payload.get("hypotheses")
    if not isinstance(rows, list) or len(rows) != HYPOTHESIS_COUNT:
        raise ValueError(f"hypotheses must contain exactly {HYPOTHESIS_COUNT} rows")
    parsed: list[dict[str, str]] = []
    descriptions: set[str] = set()
    signatures: set[tuple[str, ...]] = set()
    for index, row in enumerate(rows):
        expected_id = f"h{index + 1}"
        if not isinstance(row, dict) or row.get("id") != expected_id:
            raise ValueError("hypothesis IDs or order changed")
        description = " ".join(str(row.get("description", "")).split())
        if not description or description.casefold() in descriptions:
            raise ValueError("hypothesis descriptions must be distinct")
        descriptions.add(description.casefold())
        parsed_row = {"id": expected_id, "description": description}
        for field in STATUS_FIELDS:
            value = str(row.get(field, "")).strip().lower()
            if value not in STATUS_VALUES:
                raise ValueError(f"invalid {field} status for {expected_id}")
            parsed_row[field] = value
        signature = tuple(parsed_row[field] for field in STATUS_FIELDS)
        if signature in signatures:
            raise ValueError("hypothesis status signatures must be distinct")
        signatures.add(signature)
        parsed.append(parsed_row)
    return parsed


def compatible_permission_worlds(
    hypothesis: dict[str, str],
) -> tuple[str, ...]:
    if any(
        hypothesis[field] == "faulty"
        for field in ("network_mode", "wifi_calling", "mmsc_apn")
    ):
        return ()
    sms = hypothesis["sms_permission"]
    storage = hypothesis["storage_permission"]
    if sms == storage == "unknown":
        return ()
    if "faulty" not in (sms, storage):
        return ()
    worlds = []
    for world in PERMISSION_WORLDS:
        expected_sms, expected_storage = WORLD_PERMISSION_STATES[world]
        if sms not in ("unknown", expected_sms):
            continue
        if storage not in ("unknown", expected_storage):
            continue
        worlds.append(world)
    return tuple(worlds)


def proposed_world_support(
    hypotheses: Sequence[dict[str, str]],
) -> tuple[list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    covered: set[str] = set()
    for hypothesis in hypotheses:
        worlds = compatible_permission_worlds(hypothesis)
        rows.append({**hypothesis, "compatible_worlds": list(worlds)})
        covered.update(worlds)
    support = [world for world in PERMISSION_WORLDS if world in covered]
    return support, rows


def _selected_action(
    values: dict[str, dict[str, Any]],
    score_key: str,
) -> str:
    return max(
        values,
        key=lambda action: (float(values[action][score_key]), action),
    )


def _legal_followups(root_action: str) -> list[str]:
    legal = [action for action in ROOT_ACTIONS if action != root_action]
    if root_action == "installed_apps":
        legal.append(str(UNLOCKED_ACTION["id"]))
    return legal


def _single_branch_followup(
    observations: dict[str, dict[str, str]],
    values: dict[str, dict[str, Any]],
    root_action: str,
) -> str:
    root_outcomes = {
        observations[world][root_action] for world in observations
    }
    if len(root_outcomes) != 1:
        raise ValueError("confirmation root has multiple outcomes")
    return str(
        values[root_action]["exact_d2_branch_actions"][
            next(iter(root_outcomes))
        ]
    )


def _greedy_followup(
    observations: dict[str, dict[str, str]],
    root_action: str,
) -> str:
    legal = _legal_followups(root_action)
    return max(
        legal,
        key=lambda action: (
            information_gain(
                [observations[world][action] for world in observations]
            ),
            action,
        ),
    )


def sequence_information(
    observations: dict[str, dict[str, str]],
    root_action: str,
    followup_action: str,
) -> float:
    return information_gain(
        [
            json.dumps(
                (
                    observations[world][root_action],
                    observations[world][followup_action],
                ),
                separators=(",", ":"),
            )
            for world in observations
        ]
    )


def _truth_log_posterior(
    observations: dict[str, dict[str, str]],
    truth: str,
    root_action: str,
    followup_action: str,
) -> float:
    signature = (
        observations[truth][root_action],
        observations[truth][followup_action],
    )
    compatible = sum(
        (
            observations[world][root_action],
            observations[world][followup_action],
        )
        == signature
        for world in observations
    )
    return -math.log(compatible)


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    draws: int = 5000,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.05, 0.95])
    return float(low), float(high)


def _usage(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(snapshot["adapter_cost_usd"]),
        "model": snapshot,
    }


def _write_raw(path: Path | None, stage: str, raw: Sequence[str]) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "stage": stage,
                "responses": list(raw),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _ranking_summary(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    expected = len(stage_prompts(stage))
    full_coverage = sum(record["world_coverage"] == 3 for record in records)
    at_least_three_valid = sum(
        record["valid_permission_hypothesis_count"] >= 3 for record in records
    )
    d1_setup = sum(record["d1_selected_action"] == "installed_apps" for record in records)
    d2_setup = sum(record["d2_selected_action"] == "installed_apps" for record in records)
    setup_permissions = sum(
        record["setup_followup_action"] == "messaging_permissions"
        for record in records
    )
    summary: dict[str, Any] = {
        "num_prompt_variants": len(records),
        "full_three_world_coverage_count": full_coverage,
        "at_least_three_valid_hypotheses_count": at_least_three_valid,
        "d1_setup_selected_count": d1_setup,
        "d2_setup_selected_count": d2_setup,
        "setup_uses_permissions_count": setup_permissions,
        "mean_valid_permission_hypotheses": float(
            np.mean(
                [record["valid_permission_hypothesis_count"] for record in records]
            )
        ),
    }
    base = {
        "all_prompt_variants_completed": len(records) == expected,
        "exact_physical_request_count": int(usage["physical_requests"]) == expected,
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_scores_finite": all(
            np.isfinite(action["exact_d2_information"])
            for record in records
            for action in record["actions"]
        ),
    }
    if stage == "serving_smoke":
        gates = {
            **base,
            "full_three_world_coverage_both": full_coverage == 2,
            "at_least_three_valid_hypotheses_both": at_least_three_valid == 2,
            "d1_never_selects_setup": d1_setup == 0,
            "d2_selects_setup_both": d2_setup == 2,
            "setup_uses_permissions_both": setup_permissions == 2,
        }
    else:
        gates = {
            **base,
            "full_three_world_coverage_at_least_10": full_coverage >= 10,
            "at_least_three_valid_hypotheses_at_least_10": at_least_three_valid
            >= 10,
            "d1_never_selects_setup": d1_setup == 0,
            "d2_selects_setup_at_least_10": d2_setup >= 10,
            "setup_uses_permissions_at_least_10": setup_permissions >= 10,
        }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def _confirmation_summary(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    d1_minus_d2 = [
        record["d1_final_entropy_nats"] - record["d2_final_entropy_nats"]
        for record in records
    ]
    random_minus_d2 = [
        record["random_final_entropy_nats"] - record["d2_final_entropy_nats"]
        for record in records
    ]
    d1_truth = [
        record["d2_truth_log_posterior"] - record["d1_truth_log_posterior"]
        for record in records
    ]
    random_truth = [
        record["d2_truth_log_posterior"]
        - record["random_truth_log_posterior"]
        for record in records
    ]
    d1_ci = _bootstrap_mean_ci(d1_minus_d2, seed=SELECTION_SEED + 101)
    random_ci = _bootstrap_mean_ci(random_minus_d2, seed=SELECTION_SEED + 102)
    d1_truth_ci = _bootstrap_mean_ci(d1_truth, seed=SELECTION_SEED + 103)
    random_truth_ci = _bootstrap_mean_ci(
        random_truth, seed=SELECTION_SEED + 104
    )
    full_coverage = sum(record["world_coverage"] == 3 for record in records)
    truth_coverage = sum(record["truth_in_proposed_support"] for record in records)
    d2_setup = sum(record["d2_root_action"] == "installed_apps" for record in records)
    d2_wins_d1 = sum(value > 1.0e-12 for value in d1_minus_d2)
    d2_wins_random = sum(value > 1.0e-12 for value in random_minus_d2)
    summary = {
        "num_trajectories": len(records),
        "full_three_world_coverage_count": full_coverage,
        "truth_coverage_count": truth_coverage,
        "d2_setup_selected_count": d2_setup,
        "d2_wins_d1_count": d2_wins_d1,
        "d2_wins_random_count": d2_wins_random,
        "mean_d1_minus_d2_final_entropy_nats": float(np.mean(d1_minus_d2)),
        "d1_minus_d2_final_entropy_90ci": list(d1_ci),
        "mean_random_minus_d2_final_entropy_nats": float(
            np.mean(random_minus_d2)
        ),
        "random_minus_d2_final_entropy_90ci": list(random_ci),
        "mean_d2_minus_d1_truth_log_posterior": float(np.mean(d1_truth)),
        "d2_minus_d1_truth_log_posterior_90ci": list(d1_truth_ci),
        "mean_d2_minus_random_truth_log_posterior": float(
            np.mean(random_truth)
        ),
        "d2_minus_random_truth_log_posterior_90ci": list(random_truth_ci),
        "mean_d1_minus_d2_entropy_auc_nats": float(
            np.mean(
                [
                    record["d1_entropy_auc_nats"]
                    - record["d2_entropy_auc_nats"]
                    for record in records
                ]
            )
        ),
    }
    gates = {
        "all_confirmation_prompts_completed": len(records)
        == len(CONFIRMATION_PROMPTS),
        "exact_physical_request_count": int(usage["physical_requests"])
        == len(CONFIRMATION_PROMPTS),
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "full_support_coverage_at_least_20": full_coverage >= 20,
        "truth_coverage_at_least_22": truth_coverage >= 22,
        "d2_selects_setup_at_least_20": d2_setup >= 20,
        "d2_wins_d1_at_least_20": d2_wins_d1 >= 20,
        "d2_wins_random_at_least_16": d2_wins_random >= 16,
        "d1_entropy_gain_at_least_0_80": float(np.mean(d1_minus_d2)) >= 0.80,
        "d1_entropy_gain_ci_positive": d1_ci[0] > 0.0,
        "random_entropy_gain_at_least_0_40": float(np.mean(random_minus_d2))
        >= 0.40,
        "random_entropy_gain_ci_positive": random_ci[0] > 0.0,
        "d1_truth_gain_ci_positive": d1_truth_ci[0] > 0.0,
        "random_truth_gain_ci_positive": random_truth_ci[0] > 0.0,
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def run_gate(
    config: Config,
    *,
    t3_dir: str | Path,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    prompts = stage_prompts(stage)
    t3_path = Path(t3_dir)
    observations = exact_observations_for_backbone(
        t3_path,
        _task_index(_load_tau2_tasks(t3_path)),
        (),
    )
    observations = {
        world: observations[world] for world in PERMISSION_WORLDS
    }
    observed_history = common_observed_history(observations)

    if len(config.model_pairs) != 1:
        raise ValueError("Tau2 MMS V6 requires exactly one model pair")
    model = build_model_adapter(config.model_pairs[0].questioner, config)
    raw = model.chat_complete_messages_batched(
        [support_messages(prompt, observed_history) for prompt in prompts],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    _write_raw(raw_checkpoint_path, stage, raw)
    if len(raw) != len(prompts):
        raise ValueError("Tau2 MMS V6 response count changed")
    hypotheses_many = [parse_support(response) for response in raw]

    truth_schedule = [
        PERMISSION_WORLDS[index % len(PERMISSION_WORLDS)]
        for index in range(len(prompts))
    ]
    random.Random(SELECTION_SEED).shuffle(truth_schedule)
    records: list[dict[str, Any]] = []
    for prompt_index, (prompt, hypotheses) in enumerate(
        zip(prompts, hypotheses_many, strict=True)
    ):
        support, annotated = proposed_world_support(hypotheses)
        if len(support) < 2:
            raise ValueError("proposed permission support has fewer than two worlds")
        support_observations = {world: observations[world] for world in support}
        values = exact_action_values(support_observations)
        d1_root = _selected_action(values, "exact_d1_information")
        d2_root = _selected_action(values, "exact_d2_information")
        setup_followup = _single_branch_followup(
            support_observations, values, "installed_apps"
        )
        actions = [
            {"action_id": action, **values[action]} for action in ROOT_ACTIONS
        ]
        record: dict[str, Any] = {
            "prompt_variant": prompt_index,
            "ticket": prompt,
            "observed_history": observed_history,
            "generated_hypotheses": annotated,
            "valid_permission_hypothesis_count": sum(
                bool(row["compatible_worlds"]) for row in annotated
            ),
            "proposed_world_support": support,
            "world_coverage": len(support),
            "d1_selected_action": d1_root,
            "d2_selected_action": d2_root,
            "setup_followup_action": setup_followup,
            "actions": actions,
        }
        if stage == "confirmation":
            d1_followup = _greedy_followup(support_observations, d1_root)
            d2_followup = _single_branch_followup(
                support_observations, values, d2_root
            )
            rng = random.Random(SELECTION_SEED + 1000 + prompt_index)
            random_root = rng.choice(list(ROOT_ACTIONS))
            random_followup = rng.choice(_legal_followups(random_root))
            truth = truth_schedule[prompt_index]
            initial_entropy = math.log(len(observations))

            def endpoint(root: str, followup: str) -> tuple[float, float, float]:
                root_information = information_gain(
                    [observations[world][root] for world in observations]
                )
                total_information = sequence_information(
                    observations, root, followup
                )
                entropy_after_root = initial_entropy - root_information
                final_entropy = initial_entropy - total_information
                entropy_auc = (
                    0.5 * initial_entropy
                    + entropy_after_root
                    + 0.5 * final_entropy
                )
                truth_log = _truth_log_posterior(
                    observations, truth, root, followup
                )
                return final_entropy, entropy_auc, truth_log

            d1_endpoint = endpoint(d1_root, d1_followup)
            d2_endpoint = endpoint(d2_root, d2_followup)
            random_endpoint = endpoint(random_root, random_followup)
            record.update(
                {
                    "truth_world_id": truth,
                    "truth_in_proposed_support": truth in support,
                    "d1_root_action": d1_root,
                    "d1_followup_action": d1_followup,
                    "d2_root_action": d2_root,
                    "d2_followup_action": d2_followup,
                    "random_root_action": random_root,
                    "random_followup_action": random_followup,
                    "d1_final_entropy_nats": d1_endpoint[0],
                    "d1_entropy_auc_nats": d1_endpoint[1],
                    "d1_truth_log_posterior": d1_endpoint[2],
                    "d2_final_entropy_nats": d2_endpoint[0],
                    "d2_entropy_auc_nats": d2_endpoint[1],
                    "d2_truth_log_posterior": d2_endpoint[2],
                    "random_final_entropy_nats": random_endpoint[0],
                    "random_entropy_auc_nats": random_endpoint[1],
                    "random_truth_log_posterior": random_endpoint[2],
                }
            )
        records.append(record)

    usage = _usage(model)
    summary = (
        _confirmation_summary(records, usage)
        if stage == "confirmation"
        else _ranking_summary(records, usage, stage=stage)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "distinct_support_proposal_after_v5": True,
            "source_world_count": 1984,
            "observed_history_action_count": len(OBSERVED_ACTIONS),
            "remaining_official_worlds": list(PERMISSION_WORLDS),
            "llm_generates_semantic_hypothesis_support": True,
            "exact_history_compatibility_filter": True,
            "official_tau2_likelihoods_and_planner": True,
            "official_future_outputs_hidden_from_llm": True,
            "llm_receives_no_scores_or_policy_choices": True,
            "one_llm_request_per_prompt_variant": True,
            "no_reasoning": True,
            "expected_physical_requests": len(prompts),
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--t3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "formal", "confirmation"),
        required=True,
    )
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.05
        config.openrouter_run_budget_usd = 0.50
    elif args.stage == "formal":
        config.openrouter_projected_cost_usd = 0.20
        config.openrouter_run_budget_usd = 1.00
    else:
        config.openrouter_projected_cost_usd = 0.40
        config.openrouter_run_budget_usd = 1.50
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = {
        "serving_smoke": "SERVING_SMOKE.json",
        "formal": "GATE.json",
        "confirmation": "CONFIRMATION.json",
    }[args.stage]
    failure_name = {
        "serving_smoke": "SERVING_SMOKE_FAILURE.json",
        "formal": "GATE_FAILURE.json",
        "confirmation": "CONFIRMATION_FAILURE.json",
    }[args.stage]
    try:
        payload = run_gate(
            config,
            t3_dir=args.t3_dir,
            stage=args.stage,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_responses_path": str(raw_path),
        }
        (args.output_dir / failure_name).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
