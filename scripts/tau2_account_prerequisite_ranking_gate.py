#!/usr/bin/env python3
"""Rank a zero-EIG Tau2 customer lookup with semantic LLM rollouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.tau2_mms_prerequisite_ranking_gate import (
    T3_COMMIT,
    information_gain,
    spearman,
)


HYPOTHESIS_COUNT = 8
SELECTION_SEED = 24316

WORLD_SPECS = (
    (
        "available_roaming_enabled_device_off",
        "mobile_data_issue",
        ("airplane_mode_on", "user_abroad_roaming_enabled_off"),
    ),
    (
        "available_roaming_disabled_device_on",
        "mobile_data_issue",
        ("airplane_mode_on", "user_abroad_roaming_disabled_on"),
    ),
    (
        "available_roaming_disabled_device_off",
        "mobile_data_issue",
        ("airplane_mode_on", "user_abroad_roaming_disabled_off"),
    ),
    (
        "exhausted_roaming_enabled_device_off",
        "mobile_data_issue",
        (
            "airplane_mode_on",
            "data_usage_exceeded",
            "user_abroad_roaming_enabled_off",
        ),
    ),
    (
        "exhausted_roaming_disabled_device_on",
        "mobile_data_issue",
        (
            "airplane_mode_on",
            "data_usage_exceeded",
            "user_abroad_roaming_disabled_on",
        ),
    ),
    (
        "exhausted_roaming_disabled_device_off",
        "mobile_data_issue",
        (
            "airplane_mode_on",
            "data_usage_exceeded",
            "user_abroad_roaming_disabled_off",
        ),
    ),
)

SMOKE_PROMPTS = (
    "The customer's cellular service and mobile data are unavailable.",
    "The phone cannot obtain usable carrier service or mobile internet.",
)

FORMAL_PROMPTS = (
    "The handset has no usable cellular service and mobile data does not work.",
    "Calls and mobile internet cannot reach the carrier network.",
    "The phone shows no usable carrier connection and data is unavailable.",
    "Cellular connectivity has stopped, including mobile internet access.",
    "The customer cannot establish service through the mobile carrier.",
    "The device currently has neither usable service nor mobile data.",
    "Carrier connectivity is unavailable and the phone cannot use mobile internet.",
    "The handset cannot connect for cellular service or data.",
    "The customer's line is not providing usable cellular connectivity.",
    "Mobile carrier service is unavailable on the phone.",
    "The device has lost carrier service and cannot access mobile data.",
    "No usable cellular connection is available to the customer.",
)

ROOT_ACTIONS = {
    "status_bar": {
        "side": "user",
        "method": "check_status_bar",
        "description": "Read the phone's current status bar.",
    },
    "network_status": {
        "side": "user",
        "method": "check_network_status",
        "description": "Read the complete device network status.",
    },
    "speed_test": {
        "side": "user",
        "method": "run_speed_test",
        "description": "Run a mobile internet speed test.",
    },
    "payment_request": {
        "side": "user",
        "method": "check_payment_request",
        "description": "Check whether the phone shows a carrier payment request.",
    },
    "sim_status": {
        "side": "user",
        "method": "check_sim_status",
        "description": "Read the SIM card status.",
    },
    "customer_lookup": {
        "side": "assistant",
        "method": "get_customer_by_phone",
        "description": (
            "Look up the customer by the known phone number. This returns identity "
            "and owned line IDs, but not the hidden line state."
        ),
    },
}

UNLOCKED_ACTIONS = {
    "line_details": (
        "Use the newly observed line ID to read its complete carrier line record, "
        "including status, roaming setting, plan, and usage fields."
    ),
    "data_usage": (
        "Use the newly observed customer and line IDs to read data allowance usage."
    ),
    "customer_bills": (
        "Use the newly observed customer ID to read recent carrier bills."
    ),
}

STATUS_FIELDS = (
    "data_allowance",
    "account_roaming",
    "device_roaming",
)
STATUS_VALUES = {
    "data_allowance": {"available", "exhausted", "unknown"},
    "account_roaming": {"enabled", "disabled", "unknown"},
    "device_roaming": {"on", "off", "unknown"},
}

OFFICIAL_SIGNATURES = (
    ("available", "enabled", "off"),
    ("available", "disabled", "on"),
    ("available", "disabled", "off"),
    ("exhausted", "enabled", "off"),
    ("exhausted", "disabled", "on"),
    ("exhausted", "disabled", "off"),
)
DEFAULT_SIGNATURE = ("available", "enabled", "off")


def selected_prompts(stage: str) -> tuple[str, ...]:
    if stage == "serving_smoke":
        return SMOKE_PROMPTS
    if stage == "formal":
        return FORMAL_PROMPTS
    raise ValueError("stage must be serving_smoke or formal")


def hypothesis_messages(ticket: str) -> list[dict[str, str]]:
    schema = {
        "hypotheses": [
            {
                "id": f"h{index + 1}",
                "description": "short concrete carrier-account hypothesis",
                "data_allowance": "available|exhausted|unknown",
                "account_roaming": "enabled|disabled|unknown",
                "device_roaming": "on|off|unknown",
            }
            for index in range(HYPOTHESIS_COUNT)
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "Maintain a target-blind semantic hypothesis support for carrier "
                "account diagnosis. Return strict JSON only, with no reasoning text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Ticket: {ticket} Airplane mode is known to be on and cannot be "
                "changed during this diagnostic window. It is shared by every "
                "possible world, so diagnose the additional carrier-account or "
                "roaming state: data allowance, account roaming, and device roaming. "
                "The customer's phone number is known, but customer "
                "and line IDs have not been retrieved. Generate exactly eight "
                "distinct plausible hypotheses with materially different diagnostic "
                "predictions. Multiple abnormal fields may coexist. Do not identify "
                "a hypothesis as true. Return this exact schema:\n"
                + json.dumps(schema, separators=(",", ":"))
            ),
        },
    ]


def parse_hypotheses(text: str) -> list[dict[str, str]]:
    rows = _parse_json_object(text).get("hypotheses")
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
            raise ValueError("hypothesis descriptions must be nonempty and distinct")
        descriptions.add(description.casefold())
        parsed_row = {"id": expected_id, "description": description}
        for field in STATUS_FIELDS:
            value = str(row.get(field, "")).strip().lower()
            if value not in STATUS_VALUES[field]:
                raise ValueError(f"invalid {field} status for {expected_id}")
            parsed_row[field] = value
        signature = tuple(parsed_row[field] for field in STATUS_FIELDS)
        if signature in signatures:
            raise ValueError("hypothesis status signatures must be distinct")
        signatures.add(signature)
        parsed.append(parsed_row)
    return parsed


def rollout_messages(
    hypotheses: Sequence[dict[str, str]],
    root_action: str,
    ticket: str,
) -> list[dict[str, str]]:
    legal = {
        action: ROOT_ACTIONS[action]["description"]
        for action in ROOT_ACTIONS
        if action != root_action
    }
    if root_action == "customer_lookup":
        legal.update(UNLOCKED_ACTIONS)
    schema = {
        "root_predictions": [
            {"hypothesis_id": row["id"], "outcome": "short category"}
            for row in hypotheses
        ],
        "branches": [
            {
                "root_outcome": "one exact category used above",
                "followup_action": "one exact legal action ID",
                "followup_predictions": [
                    {
                        "hypothesis_id": "each branch hypothesis exactly once",
                        "outcome": "short category",
                    }
                ],
            }
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Simulate a two-step carrier diagnostic policy over semantic "
                "hypotheses. Return strict JSON only, with no reasoning text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Ticket: {ticket} Airplane mode is on in every world and cannot "
                "be changed. Predict the root diagnostic result under every "
                "hypothesis, reusing exactly the same short category for "
                "observationally identical results. Create one branch per distinct "
                "root category. Choose one branch-specific follow-up by copying its "
                "ID exactly from legal_followups, then predict its result for every "
                "hypothesis in that branch. The follow-up can depend on the root "
                "observation. Customer and line IDs are unknown before "
                "customer_lookup; line_details, data_usage, and customer_bills are "
                "therefore legal only after customer_lookup. customer_lookup itself "
                "returns the same customer identity and line IDs in every possible "
                "world; hidden line state is returned only by later reads. Do not "
                "change or add hypotheses.\n"
                f"Hypotheses: {json.dumps(list(hypotheses), separators=(',', ':'))}\n"
                f"Root action: {root_action}: "
                f"{ROOT_ACTIONS[root_action]['description']}\n"
                f"legal_followups: {json.dumps(legal, separators=(',', ':'))}\n"
                "Return this exact schema:\n"
                + json.dumps(schema, separators=(",", ":"))
            ),
        },
    ]


def parse_rollout(
    text: str,
    *,
    hypotheses: Sequence[dict[str, str]],
    root_action: str,
) -> dict[str, Any]:
    payload = _parse_json_object(text)
    hypothesis_ids = [row["id"] for row in hypotheses]
    rows = payload.get("root_predictions")
    if not isinstance(rows, list) or len(rows) != len(hypothesis_ids):
        raise ValueError("root_predictions lost a hypothesis")
    root_by_id: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("root prediction must be an object")
        hypothesis_id = str(row.get("hypothesis_id", ""))
        outcome = " ".join(str(row.get("outcome", "")).split())
        if (
            hypothesis_id not in hypothesis_ids
            or hypothesis_id in root_by_id
            or not outcome
        ):
            raise ValueError("invalid root prediction")
        root_by_id[hypothesis_id] = outcome
    if list(root_by_id) != hypothesis_ids:
        raise ValueError("root prediction IDs or order changed")

    legal = set(ROOT_ACTIONS) - {root_action}
    if root_action == "customer_lookup":
        legal.update(UNLOCKED_ACTIONS)
    root_outcomes = set(root_by_id.values())
    branches = payload.get("branches")
    if not isinstance(branches, list) or len(branches) != len(root_outcomes):
        raise ValueError("branches do not match distinct root outcomes")
    branch_by_outcome: dict[str, dict[str, Any]] = {}
    followup_outcome_by_id: dict[str, str] = {}
    followup_action_by_id: dict[str, str] = {}
    for branch in branches:
        if not isinstance(branch, dict):
            raise ValueError("branch must be an object")
        root_outcome = " ".join(str(branch.get("root_outcome", "")).split())
        followup_action = str(branch.get("followup_action", "")).strip()
        if (
            root_outcome not in root_outcomes
            or root_outcome in branch_by_outcome
            or followup_action not in legal
        ):
            raise ValueError("invalid rollout branch or follow-up action")
        expected_ids = [
            hypothesis_id
            for hypothesis_id in hypothesis_ids
            if root_by_id[hypothesis_id] == root_outcome
        ]
        predictions = branch.get("followup_predictions")
        if not isinstance(predictions, list) or len(predictions) != len(
            expected_ids
        ):
            raise ValueError("follow-up predictions lost a branch hypothesis")
        parsed_predictions: list[dict[str, str]] = []
        for prediction in predictions:
            if not isinstance(prediction, dict):
                raise ValueError("follow-up prediction must be an object")
            hypothesis_id = str(prediction.get("hypothesis_id", ""))
            outcome = " ".join(str(prediction.get("outcome", "")).split())
            if (
                hypothesis_id not in expected_ids
                or hypothesis_id in followup_outcome_by_id
                or not outcome
            ):
                raise ValueError("invalid follow-up prediction")
            followup_outcome_by_id[hypothesis_id] = outcome
            followup_action_by_id[hypothesis_id] = followup_action
            parsed_predictions.append(
                {"hypothesis_id": hypothesis_id, "outcome": outcome}
            )
        if [
            prediction["hypothesis_id"]
            for prediction in parsed_predictions
        ] != expected_ids:
            raise ValueError("follow-up prediction IDs or order changed")
        branch_by_outcome[root_outcome] = {
            "root_outcome": root_outcome,
            "followup_action": followup_action,
            "followup_predictions": parsed_predictions,
        }
    if set(branch_by_outcome) != root_outcomes:
        raise ValueError("rollout omitted a root outcome")

    d1_partition = [root_by_id[value] for value in hypothesis_ids]
    d2_partition = [
        json.dumps(
            (
                root_by_id[value],
                followup_action_by_id[value],
                followup_outcome_by_id[value],
            ),
            separators=(",", ":"),
        )
        for value in hypothesis_ids
    ]
    return {
        "root_predictions": rows,
        "branches": [
            branch_by_outcome[value]
            for value in dict.fromkeys(d1_partition)
        ],
        "predicted_d1_information": information_gain(d1_partition),
        "predicted_d2_information": information_gain(d2_partition),
    }


def official_signature_coverage(hypotheses: Sequence[dict[str, str]]) -> int:
    def covers(
        row: dict[str, str],
        signature: tuple[str, ...],
    ) -> bool:
        known_fields = 0
        for field, official, default in zip(
            STATUS_FIELDS,
            signature,
            DEFAULT_SIGNATURE,
            strict=True,
        ):
            generated = row[field]
            known_fields += generated != "unknown"
            if official != default:
                if generated != official:
                    return False
            elif generated not in {official, "unknown"}:
                return False
        return known_fields >= 2

    return sum(
        any(covers(row, signature) for row in hypotheses)
        for signature in OFFICIAL_SIGNATURES
    )


def _check_t3_commit(t3_dir: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(t3_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != T3_COMMIT:
        raise ValueError("external T3 checkout does not match the frozen commit")


def _load_tau2(t3_dir: Path) -> tuple[list[Any], Any]:
    _check_t3_commit(t3_dir)
    package_path = str(t3_dir / "verl")
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    from search_r1.tau2_adapter.loader.registry import get_env_constructor
    from search_r1.tau2_adapter.loader.tasks import get_tasks

    return get_tasks("telecom", task_split_name=None), get_env_constructor


def _find_task(
    tasks: Sequence[Any],
    issue: str,
    faults: Sequence[str],
) -> Any:
    target = set(faults)
    for task in tasks:
        task_id = str(task.id)
        if not task_id.startswith(f"[{issue}]"):
            continue
        fault_text = task_id.split("]", 1)[1].split("[PERSONA:", 1)[0]
        if set(fault_text.split("|")) == target:
            return task
    raise ValueError(f"missing Tau2 task for {issue}: {sorted(target)}")


def exact_observations(t3_dir: Path) -> dict[str, dict[str, str]]:
    tasks, get_env_constructor = _load_tau2(t3_dir)
    observations: dict[str, dict[str, str]] = {}
    for world_id, issue, faults in WORLD_SPECS:
        task = _find_task(tasks, issue, faults)
        environment = get_env_constructor("telecom")(solo_mode=False)
        environment.set_state(
            initialization_data=task.initial_state.initialization_data,
            initialization_actions=task.initial_state.initialization_actions,
            message_history=[],
        )
        row: dict[str, str] = {}
        for action_id, action in ROOT_ACTIONS.items():
            try:
                toolkit = (
                    environment.user_tools
                    if action["side"] == "user"
                    else environment.tools
                )
                function = getattr(toolkit, str(action["method"]))
                value = (
                    function()
                    if action["side"] == "user"
                    else function("555-123-2002")
                )
            except Exception as exc:
                value = f"ERROR:{type(exc).__name__}:{exc}"
            row[action_id] = str(value)
        environment.tools.get_customer_by_phone("555-123-2002")
        unlocked_calls = {
            "line_details": lambda: environment.tools.get_details_by_id(
                "L1002"
            ),
            "data_usage": lambda: environment.tools.get_data_usage(
                "C1001", "L1002"
            ),
            "customer_bills": lambda: environment.tools.get_bills_for_customer(
                "C1001"
            ),
        }
        for action_id, function in unlocked_calls.items():
            try:
                value = function()
            except Exception as exc:
                value = f"ERROR:{type(exc).__name__}:{exc}"
            row[action_id] = str(value)
        observations[world_id] = row
    return observations


def best_two_step_information(
    observations: dict[str, dict[str, str]],
    root_action: str,
) -> tuple[float, dict[str, str]]:
    worlds = list(observations)
    root_groups: dict[str, list[str]] = defaultdict(list)
    for world in worlds:
        root_groups[observations[world][root_action]].append(world)
    expected_terminal_entropy = 0.0
    branch_actions: dict[str, str] = {}
    for root_outcome, branch_worlds in root_groups.items():
        legal = [action for action in ROOT_ACTIONS if action != root_action]
        if root_action == "customer_lookup":
            legal.extend(UNLOCKED_ACTIONS)
        best_gain = -1.0
        best_action = legal[0]
        for action in legal:
            gain = information_gain(
                [observations[world][action] for world in branch_worlds]
            )
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_action = action
        branch_actions[root_outcome] = best_action
        expected_terminal_entropy += (
            len(branch_worlds) / len(worlds)
        ) * (math.log(len(branch_worlds)) - best_gain)
    return math.log(len(worlds)) - expected_terminal_entropy, branch_actions


def exact_action_values(
    observations: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    worlds = list(observations)
    values: dict[str, dict[str, Any]] = {}
    for action in ROOT_ACTIONS:
        d2_value, branch_actions = best_two_step_information(
            observations, action
        )
        values[action] = {
            "exact_d1_information": information_gain(
                [observations[world][action] for world in worlds]
            ),
            "exact_d2_information": d2_value,
            "exact_d2_branch_actions": branch_actions,
        }
    return values


def _usage(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(snapshot["adapter_cost_usd"]),
        "model": snapshot,
    }


def _write_raw(path: Path | None, stage: str, raw: dict[str, Any]) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {"schema_version": 1, "stage": stage, "responses": raw},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    expected_variants = len(selected_prompts(stage))
    expected_requests = expected_variants * (1 + len(ROOT_ACTIONS))
    setup_uses_line_details = sum(
        all(
            branch["followup_action"] == "line_details"
            for branch in record["actions_by_id"]["customer_lookup"]["branches"]
        )
        for record in records
    )
    base_gates = {
        "all_prompt_variants_completed": len(records) == expected_variants,
        "exact_physical_request_count": (
            int(usage["physical_requests"]) == expected_requests
        ),
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_scores_finite": all(
            np.isfinite(action["predicted_d2_information"])
            for record in records
            for action in record["actions"]
        ),
    }
    mean_coverage = float(
        np.mean([record["official_signature_coverage"] for record in records])
    )
    if stage == "serving_smoke":
        gates = {
            **base_gates,
            "mean_signature_coverage_at_least_4": mean_coverage >= 4.0,
            "setup_uses_line_details_both": setup_uses_line_details == 2,
        }
        gates["all_pass"] = all(gates.values())
        return {
            "num_prompt_variants": len(records),
            "mean_official_signature_coverage": mean_coverage,
            "setup_uses_line_details_count": setup_uses_line_details,
            "gates": gates,
        }

    predicted: list[float] = []
    exact: list[float] = []
    d1_regrets: list[float] = []
    d2_regrets: list[float] = []
    d1_setup_count = 0
    d2_setup_count = 0
    d2_beats_d1 = 0
    for record in records:
        actions = record["actions"]
        predicted.extend(
            float(action["predicted_d2_information"]) for action in actions
        )
        exact.extend(float(action["exact_d2_information"]) for action in actions)
        oracle = max(float(action["exact_d2_information"]) for action in actions)
        d1 = max(
            actions,
            key=lambda action: (
                float(action["predicted_d1_information"]),
                action["action_id"],
            ),
        )
        d2 = max(
            actions,
            key=lambda action: (
                float(action["predicted_d2_information"]),
                action["action_id"],
            ),
        )
        d1_regret = oracle - float(d1["exact_d2_information"])
        d2_regret = oracle - float(d2["exact_d2_information"])
        d1_regrets.append(d1_regret)
        d2_regrets.append(d2_regret)
        d1_setup_count += d1["action_id"] == "customer_lookup"
        d2_setup_count += d2["action_id"] == "customer_lookup"
        d2_beats_d1 += d2_regret + 1e-12 < d1_regret
    correlation = spearman(predicted, exact)
    mean_d1_regret = float(np.mean(d1_regrets))
    mean_d2_regret = float(np.mean(d2_regrets))
    summary = {
        "num_prompt_variants": len(records),
        "mean_official_signature_coverage": mean_coverage,
        "predicted_d2_spearman_vs_exact_d2": correlation,
        "d1_setup_selected_count": d1_setup_count,
        "d2_setup_selected_count": d2_setup_count,
        "setup_uses_line_details_count": setup_uses_line_details,
        "d2_beats_d1_count": d2_beats_d1,
        "mean_d1_top1_regret_nats": mean_d1_regret,
        "mean_d2_top1_regret_nats": mean_d2_regret,
        "mean_regret_improvement_nats": (
            mean_d1_regret - mean_d2_regret
        ),
    }
    gates = {
        **base_gates,
        "mean_signature_coverage_at_least_5": mean_coverage >= 5.0,
        "setup_uses_line_details_at_least_10": setup_uses_line_details >= 10,
        "d2_spearman_at_least_0_25": (
            correlation is not None and correlation >= 0.25
        ),
        "d2_setup_selected_at_least_8": d2_setup_count >= 8,
        "d1_never_selects_setup": d1_setup_count == 0,
        "d2_regret_improves_by_0_50": (
            mean_d1_regret - mean_d2_regret >= 0.50
        ),
        "d2_beats_d1_at_least_8": d2_beats_d1 >= 8,
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
    t3_path = Path(t3_dir)
    prompts = selected_prompts(stage)
    _check_t3_commit(t3_path)
    if len(config.model_pairs) != 1:
        raise ValueError("Tau2 account gate requires exactly one model pair")
    model = build_model_adapter(config.model_pairs[0].questioner, config)
    raw: dict[str, Any] = {}

    hypothesis_raw = model.chat_complete_messages_batched(
        [hypothesis_messages(prompt) for prompt in prompts],
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["hypotheses"] = hypothesis_raw
    _write_raw(raw_checkpoint_path, stage, raw)
    hypotheses_many = [parse_hypotheses(value) for value in hypothesis_raw]

    flat_keys: list[tuple[int, str]] = []
    rollout_prompts: list[list[dict[str, str]]] = []
    for prompt_index, (prompt, hypotheses) in enumerate(
        zip(prompts, hypotheses_many, strict=True)
    ):
        for action in ROOT_ACTIONS:
            flat_keys.append((prompt_index, action))
            rollout_prompts.append(
                rollout_messages(hypotheses, action, prompt)
            )
    rollout_raw = model.chat_complete_messages_batched(
        rollout_prompts,
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["rollouts"] = rollout_raw
    _write_raw(raw_checkpoint_path, stage, raw)

    parsed: dict[tuple[int, str], dict[str, Any]] = {}
    for response, (prompt_index, action) in zip(
        rollout_raw, flat_keys, strict=True
    ):
        parsed[prompt_index, action] = parse_rollout(
            response,
            hypotheses=hypotheses_many[prompt_index],
            root_action=action,
        )

    exact_values = exact_action_values(exact_observations(t3_path))
    records: list[dict[str, Any]] = []
    for prompt_index, (prompt, hypotheses) in enumerate(
        zip(prompts, hypotheses_many, strict=True)
    ):
        actions: list[dict[str, Any]] = []
        actions_by_id: dict[str, dict[str, Any]] = {}
        for action in ROOT_ACTIONS:
            row = {
                "action_id": action,
                **parsed[prompt_index, action],
                **exact_values[action],
            }
            actions.append(row)
            actions_by_id[action] = row
        records.append(
            {
                "prompt_variant": prompt_index,
                "ticket": prompt,
                "hypotheses": hypotheses,
                "official_signature_coverage": official_signature_coverage(
                    hypotheses
                ),
                "actions": actions,
                "actions_by_id": actions_by_id,
            }
        )
    usage = _usage(model)
    summary = summarize(records, usage, stage=stage)
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "t3_commit": T3_COMMIT,
            "world_specs": [
                {
                    "world_id": world_id,
                    "issue": issue,
                    "faults": list(faults),
                }
                for world_id, issue, faults in WORLD_SPECS
            ],
            "prompt_variants": list(prompts),
            "prompt_variants_share_one_physical_support_family": True,
            "hypothesis_count": HYPOTHESIS_COUNT,
            "root_actions": list(ROOT_ACTIONS),
            "official_task_states_hidden_from_llm": True,
            "official_simulator_used_only_for_realized_scoring": True,
            "llm_generates_semantic_support": True,
            "llm_predicts_two_step_likelihood_partitions": True,
            "no_reasoning": True,
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
        choices=("serving_smoke", "formal"),
        required=True,
    )
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.05
        config.openrouter_run_budget_usd = 0.50
    else:
        config.openrouter_projected_cost_usd = 0.20
        config.openrouter_run_budget_usd = 2.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "GATE.json"
    )
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "GATE_FAILURE.json"
    )
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
            "schema_version": 1,
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
    print(
        json.dumps(
            {"status": payload["status"], **payload["summary"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
