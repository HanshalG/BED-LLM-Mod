#!/usr/bin/env python3
"""Rank Tau2 MMS diagnostics with explicit LLM semantic rollouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.movielens_profile_dynamics_gate import _parse_json_object


T3_COMMIT = "492f31fa05d2065c750a72d5e798385af282fa5d"
SELECTION_SEED = 24315
HYPOTHESIS_COUNT = 8

FAULT_VARIANTS = (
    "bad_network_preference",
    "bad_wifi_calling",
    "break_apn_mms_setting",
    "break_app_sms_permission",
    "break_app_storage_permission",
    "break_app_both_permissions",
)

SMOKE_BACKBONES: tuple[tuple[str, ...], ...] = (
    (),
    ("airplane_mode_on",),
)
FORMAL_BACKBONES: tuple[tuple[str, ...], ...] = (
    ("data_mode_off",),
    ("data_usage_exceeded",),
    ("unseat_sim_card",),
    ("user_abroad_roaming_disabled_off",),
    ("user_abroad_roaming_disabled_on",),
    ("user_abroad_roaming_enabled_off",),
    ("airplane_mode_on", "data_mode_off"),
    ("airplane_mode_on", "data_usage_exceeded"),
    ("data_mode_off", "unseat_sim_card"),
    ("data_usage_exceeded", "user_abroad_roaming_disabled_off"),
    ("unseat_sim_card", "user_abroad_roaming_disabled_on"),
    ("airplane_mode_on", "data_mode_off", "unseat_sim_card"),
)

BACKGROUND_TEXT = {
    "airplane_mode_on": "Airplane mode is currently on.",
    "data_mode_off": "Mobile data is currently switched off.",
    "data_usage_exceeded": "The line has exhausted its included data allowance.",
    "unseat_sim_card": "The SIM card is currently unseated.",
    "user_abroad_roaming_disabled_off": (
        "The user is abroad; device roaming is off and account roaming is disabled."
    ),
    "user_abroad_roaming_disabled_on": (
        "The user is abroad; device roaming is on and account roaming is disabled."
    ),
    "user_abroad_roaming_enabled_off": (
        "The user is abroad; device roaming is off and account roaming is enabled."
    ),
}

ROOT_ACTIONS = {
    "status_bar": {
        "method": "check_status_bar",
        "description": "Read the phone status bar.",
    },
    "network_status": {
        "method": "check_network_status",
        "description": "Read the complete cellular and Wi-Fi network status.",
    },
    "network_mode": {
        "method": "check_network_mode_preference",
        "description": "Read the preferred cellular network mode.",
    },
    "apn_settings": {
        "method": "check_apn_settings",
        "description": "Read the current APN and MMSC settings.",
    },
    "wifi_calling": {
        "method": "check_wifi_calling_status",
        "description": "Read whether Wi-Fi Calling is enabled.",
    },
    "speed_test": {
        "method": "run_speed_test",
        "description": "Run a mobile internet speed test.",
    },
    "mms_probe": {
        "method": "can_send_mms",
        "description": "Attempt to send an MMS message.",
    },
    "installed_apps": {
        "method": "check_installed_apps",
        "description": "List the apps installed on the phone.",
    },
}

UNLOCKED_ACTION = {
    "id": "messaging_permissions",
    "description": (
        "Read the permissions granted to the installed app named messaging."
    ),
}

STATUS_FIELDS = (
    "network_mode",
    "wifi_calling",
    "mmsc_apn",
    "sms_permission",
    "storage_permission",
)
STATUS_VALUES = {"normal", "faulty", "unknown"}


def selected_backbones(stage: str) -> tuple[tuple[str, ...], ...]:
    if stage == "serving_smoke":
        return SMOKE_BACKBONES
    if stage == "formal":
        return FORMAL_BACKBONES
    raise ValueError("stage must be serving_smoke or formal")


def entropy(group_sizes: Iterable[int]) -> float:
    sizes = [int(value) for value in group_sizes if int(value) > 0]
    total = sum(sizes)
    if total <= 0:
        return 0.0
    return -sum(
        (size / total) * math.log(size / total)
        for size in sizes
    )


def information_gain(partition: Sequence[str]) -> float:
    counts: dict[str, int] = defaultdict(int)
    for value in partition:
        counts[str(value)] += 1
    return math.log(len(partition)) - sum(
        (count / len(partition)) * math.log(count)
        for count in counts.values()
    )


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    rank_a = average_ranks(values_a)
    rank_b = average_ranks(values_b)
    if np.std(rank_a) == 0.0 or np.std(rank_b) == 0.0:
        return None
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def best_two_step_information(
    observations: dict[str, dict[str, str]],
    root_action: str,
    *,
    unlocked_action: str = "messaging_permissions",
) -> tuple[float, dict[str, str]]:
    worlds = list(observations)
    root_groups: dict[str, list[str]] = defaultdict(list)
    for world in worlds:
        root_groups[observations[world][root_action]].append(world)

    branch_actions: dict[str, str] = {}
    expected_terminal_entropy = 0.0
    for root_outcome, branch_worlds in root_groups.items():
        legal = [action for action in ROOT_ACTIONS if action != root_action]
        if root_action == "installed_apps":
            legal.append(unlocked_action)
        best_action = legal[0]
        best_gain = -1.0
        for action in legal:
            gain = information_gain(
                [observations[world][action] for world in branch_worlds]
            )
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_action = action
        branch_actions[root_outcome] = best_action
        branch_entropy = math.log(len(branch_worlds)) - best_gain
        expected_terminal_entropy += (
            len(branch_worlds) / len(worlds)
        ) * branch_entropy
    return math.log(len(worlds)) - expected_terminal_entropy, branch_actions


def hypothesis_messages(backbone: Sequence[str]) -> list[dict[str, str]]:
    known = (
        " None."
        if not backbone
        else " " + " ".join(BACKGROUND_TEXT[item] for item in backbone)
    )
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
    tools = [
        {"id": action_id, **row}
        for action_id, row in ROOT_ACTIONS.items()
    ] + [UNLOCKED_ACTION]
    return [
        {
            "role": "system",
            "content": (
                "You maintain a target-blind semantic hypothesis support for "
                "interactive troubleshooting. Return strict JSON only. Do not reason "
                "outside the JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "A customer cannot send picture messages from the phone's messaging "
                "app. Generate exactly eight distinct, plausible remaining failure "
                "mechanisms. Common conditions already established in every possible "
                f"world:{known} They are shared context, not the unknown mechanism. "
                "Use the tool descriptions to cover materially different diagnostic "
                "predictions. Multiple status fields may be faulty in one hypothesis. "
                "For every status field, emit exactly normal, faulty, or unknown. "
                "Do not identify a hypothesis as true. Return this exact schema:\n"
                + json.dumps(schema, separators=(",", ":"))
                + "\nAvailable diagnostics:\n"
                + json.dumps(tools, separators=(",", ":"))
            ),
        },
    ]


def parse_hypotheses(text: str) -> list[dict[str, str]]:
    rows = _parse_json_object(text).get("hypotheses")
    if not isinstance(rows, list) or len(rows) != HYPOTHESIS_COUNT:
        raise ValueError(f"hypotheses must contain exactly {HYPOTHESIS_COUNT} rows")
    parsed: list[dict[str, str]] = []
    descriptions: set[str] = set()
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
            if value not in STATUS_VALUES:
                raise ValueError(f"invalid {field} status for {expected_id}")
            parsed_row[field] = value
        parsed.append(parsed_row)
    return parsed


def rollout_messages(
    hypotheses: Sequence[dict[str, str]],
    root_action: str,
    backbone: Sequence[str],
) -> list[dict[str, str]]:
    legal_followups = [
        action
        for action in ROOT_ACTIONS
        if action != root_action
    ]
    if root_action == "installed_apps":
        legal_followups.append("messaging_permissions")
    known = [BACKGROUND_TEXT[item] for item in backbone]
    schema = {
        "root_predictions": [
            {"hypothesis_id": row["id"], "outcome": "short category"}
            for row in hypotheses
        ],
        "branches": [
            {
                "root_outcome": "one category used above",
                "followup_action": "one legal action ID",
                "followup_predictions": [
                    {
                        "hypothesis_id": "each hypothesis in this branch exactly once",
                        "outcome": "short category",
                    }
                ],
            }
        ],
    }
    action_descriptions = {
        action: ROOT_ACTIONS[action]["description"]
        for action in legal_followups
        if action in ROOT_ACTIONS
    }
    if "messaging_permissions" in legal_followups:
        action_descriptions["messaging_permissions"] = UNLOCKED_ACTION["description"]
    return [
        {
            "role": "system",
            "content": (
                "Simulate a two-step diagnostic policy over semantic hypotheses. "
                "Return strict JSON only. Do not reason outside the JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Predict the first diagnostic's observable result under every "
                "hypothesis. Use exactly the same short outcome string when results "
                "would be observationally identical. Create one branch for every "
                "distinct root outcome. In each branch choose one legal follow-up "
                "diagnostic, then predict its outcome for every hypothesis in that "
                "branch. The follow-up may depend on the root observation. Do not "
                "change or add hypotheses. Installed app names are not known until "
                "installed_apps is observed, so messaging_permissions is legal only "
                "after installed_apps. Reading installed apps does not itself reveal "
                "which failure mechanism is true.\n"
                f"Known shared conditions: {json.dumps(known)}\n"
                f"Hypotheses: {json.dumps(list(hypotheses), separators=(',', ':'))}\n"
                f"Root diagnostic: {root_action}: "
                f"{ROOT_ACTIONS[root_action]['description']}\n"
                f"Legal follow-ups: {json.dumps(action_descriptions, separators=(',', ':'))}\n"
                "Return this schema:\n"
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
    predictions = payload.get("root_predictions")
    if not isinstance(predictions, list) or len(predictions) != len(hypothesis_ids):
        raise ValueError("root_predictions lost a hypothesis")
    root_by_id: dict[str, str] = {}
    for row in predictions:
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

    legal_followups = set(ROOT_ACTIONS) - {root_action}
    if root_action == "installed_apps":
        legal_followups.add("messaging_permissions")
    branches = payload.get("branches")
    root_outcomes = set(root_by_id.values())
    if not isinstance(branches, list) or len(branches) != len(root_outcomes):
        raise ValueError("branches do not match distinct root outcomes")
    branch_by_outcome: dict[str, dict[str, Any]] = {}
    followup_by_id: dict[str, str] = {}
    followup_action_by_id: dict[str, str] = {}
    for branch in branches:
        if not isinstance(branch, dict):
            raise ValueError("branch must be an object")
        root_outcome = " ".join(str(branch.get("root_outcome", "")).split())
        action = str(branch.get("followup_action", "")).strip()
        if (
            root_outcome not in root_outcomes
            or root_outcome in branch_by_outcome
            or action not in legal_followups
        ):
            raise ValueError("invalid rollout branch or follow-up action")
        expected_ids = [
            hypothesis_id
            for hypothesis_id in hypothesis_ids
            if root_by_id[hypothesis_id] == root_outcome
        ]
        rows = branch.get("followup_predictions")
        if not isinstance(rows, list) or len(rows) != len(expected_ids):
            raise ValueError("follow-up predictions lost a branch hypothesis")
        parsed_rows: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("follow-up prediction must be an object")
            hypothesis_id = str(row.get("hypothesis_id", ""))
            outcome = " ".join(str(row.get("outcome", "")).split())
            if (
                hypothesis_id not in expected_ids
                or hypothesis_id in followup_by_id
                or not outcome
            ):
                raise ValueError("invalid follow-up prediction")
            followup_by_id[hypothesis_id] = outcome
            followup_action_by_id[hypothesis_id] = action
            parsed_rows.append(
                {"hypothesis_id": hypothesis_id, "outcome": outcome}
            )
        if [row["hypothesis_id"] for row in parsed_rows] != expected_ids:
            raise ValueError("follow-up prediction IDs or order changed")
        branch_by_outcome[root_outcome] = {
            "root_outcome": root_outcome,
            "followup_action": action,
            "followup_predictions": parsed_rows,
        }
    if set(branch_by_outcome) != root_outcomes:
        raise ValueError("rollout omitted a root outcome")
    d1_partition = [root_by_id[hypothesis_id] for hypothesis_id in hypothesis_ids]
    d2_partition = [
        json.dumps(
            [
                root_by_id[hypothesis_id],
                followup_action_by_id[hypothesis_id],
                followup_by_id[hypothesis_id],
            ],
            separators=(",", ":"),
        )
        for hypothesis_id in hypothesis_ids
    ]
    return {
        "root_predictions": predictions,
        "branches": [
            branch_by_outcome[outcome]
            for outcome in dict.fromkeys(d1_partition)
        ],
        "predicted_d1_information": information_gain(d1_partition),
        "predicted_d2_information": information_gain(d2_partition),
    }


def hypothesis_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in STATUS_FIELDS)


OFFICIAL_SIGNATURES = {
    ("faulty", "normal", "normal", "normal", "normal"),
    ("normal", "faulty", "normal", "normal", "normal"),
    ("normal", "normal", "faulty", "normal", "normal"),
    ("normal", "normal", "normal", "faulty", "normal"),
    ("normal", "normal", "normal", "normal", "faulty"),
    ("normal", "normal", "normal", "faulty", "faulty"),
}


def official_signature_coverage(hypotheses: Sequence[dict[str, str]]) -> int:
    generated = {hypothesis_signature(row) for row in hypotheses}
    return sum(
        any(
            all(
                generated_value == "faulty"
                if official_value == "faulty"
                else generated_value != "faulty"
                for generated_value, official_value in zip(
                    generated_signature,
                    official_signature,
                    strict=True,
                )
            )
            for generated_signature in generated
        )
        for official_signature in OFFICIAL_SIGNATURES
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


def _load_tau2_tasks(t3_dir: Path) -> list[Any]:
    _check_t3_commit(t3_dir)
    package_path = str(t3_dir / "verl")
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    from search_r1.tau2_adapter.loader.tasks import get_tasks

    return get_tasks("telecom", task_split_name=None)


def _task_index(tasks: Sequence[Any]) -> dict[frozenset[str], Any]:
    index: dict[frozenset[str], Any] = {}
    for task in tasks:
        task_id = str(task.id)
        if not task_id.startswith("[mms_issue]"):
            continue
        faults_text = task_id.split("]", 1)[1].split("[PERSONA:", 1)[0]
        faults = frozenset(value for value in faults_text.split("|") if value)
        index.setdefault(faults, task)
    return index


def _initialize_environment(t3_dir: Path, task: Any) -> Any:
    package_path = str(t3_dir / "verl")
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    from search_r1.tau2_adapter.loader.registry import get_env_constructor

    environment = get_env_constructor("telecom")(solo_mode=False)
    environment.set_state(
        initialization_data=task.initial_state.initialization_data,
        initialization_actions=task.initial_state.initialization_actions,
        message_history=[],
    )
    return environment


def exact_observations_for_backbone(
    t3_dir: Path,
    task_index: dict[frozenset[str], Any],
    backbone: Sequence[str],
) -> dict[str, dict[str, str]]:
    observations: dict[str, dict[str, str]] = {}
    for variant in FAULT_VARIANTS:
        faults = frozenset((*backbone, variant))
        if faults not in task_index:
            raise ValueError(f"Tau2 task missing for fault set {sorted(faults)}")
        environment = _initialize_environment(t3_dir, task_index[faults])
        row: dict[str, str] = {}
        for action_id, action in ROOT_ACTIONS.items():
            try:
                value = getattr(
                    environment.user_tools,
                    str(action["method"]),
                )()
            except Exception as exc:
                value = f"ERROR:{type(exc).__name__}:{exc}"
            row[action_id] = str(value)
        try:
            permission_value = environment.user_tools.check_app_permissions(
                "messaging"
            )
        except Exception as exc:
            permission_value = f"ERROR:{type(exc).__name__}:{exc}"
        row["messaging_permissions"] = str(permission_value)
        observations[variant] = row
    return observations


def exact_action_values(
    observations: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    worlds = list(observations)
    for action in ROOT_ACTIONS:
        d2_value, branch_actions = best_two_step_information(
            observations,
            action,
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


def _write_raw_checkpoint(
    path: Path | None,
    *,
    stage: str,
    raw: dict[str, Any],
) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": stage,
                "responses": raw,
            },
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
    expected_cases = len(selected_backbones(stage))
    expected_requests = expected_cases * (1 + len(ROOT_ACTIONS))
    base_gates = {
        "all_cases_completed": len(records) == expected_cases,
        "exact_physical_request_count": (
            int(usage["physical_requests"]) == expected_requests
        ),
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_scores_finite": all(
            np.isfinite(action["predicted_d2_information"])
            for record in records
            for action in record["actions"]
        ),
        "all_setup_rollouts_choose_permissions": all(
            all(
                branch["followup_action"] == "messaging_permissions"
                for branch in record["actions_by_id"]["installed_apps"]["branches"]
            )
            for record in records
        ),
    }
    if stage == "serving_smoke":
        gates = dict(base_gates)
        gates["mean_signature_coverage_at_least_4"] = (
            float(np.mean([record["official_signature_coverage"] for record in records]))
            >= 4.0
        )
        gates["all_pass"] = all(gates.values())
        return {
            "num_cases": len(records),
            "mean_official_signature_coverage": float(
                np.mean(
                    [record["official_signature_coverage"] for record in records]
                )
            ),
            "gates": gates,
        }

    predicted_d2: list[float] = []
    exact_d2: list[float] = []
    d2_regrets: list[float] = []
    d1_regrets: list[float] = []
    d2_setup_count = 0
    d1_setup_count = 0
    d2_beats_d1 = 0
    for record in records:
        action_rows = record["actions"]
        predicted_d2.extend(
            float(row["predicted_d2_information"]) for row in action_rows
        )
        exact_d2.extend(float(row["exact_d2_information"]) for row in action_rows)
        oracle = max(float(row["exact_d2_information"]) for row in action_rows)
        d2_selected = max(
            action_rows,
            key=lambda row: (
                float(row["predicted_d2_information"]),
                row["action_id"],
            ),
        )
        d1_selected = max(
            action_rows,
            key=lambda row: (
                float(row["predicted_d1_information"]),
                row["action_id"],
            ),
        )
        d2_regret = oracle - float(d2_selected["exact_d2_information"])
        d1_regret = oracle - float(d1_selected["exact_d2_information"])
        d2_regrets.append(d2_regret)
        d1_regrets.append(d1_regret)
        d2_setup_count += d2_selected["action_id"] == "installed_apps"
        d1_setup_count += d1_selected["action_id"] == "installed_apps"
        d2_beats_d1 += d2_regret + 1e-12 < d1_regret
    correlation = spearman(predicted_d2, exact_d2)
    mean_d2_regret = float(np.mean(d2_regrets))
    mean_d1_regret = float(np.mean(d1_regrets))
    mean_coverage = float(
        np.mean([record["official_signature_coverage"] for record in records])
    )
    summary = {
        "num_cases": len(records),
        "predicted_d2_spearman_vs_exact_d2": correlation,
        "mean_d2_top1_regret_nats": mean_d2_regret,
        "mean_d1_top1_regret_nats": mean_d1_regret,
        "mean_d2_regret_improvement_nats": mean_d1_regret - mean_d2_regret,
        "d2_setup_selected_count": d2_setup_count,
        "d1_setup_selected_count": d1_setup_count,
        "d2_beats_d1_count": d2_beats_d1,
        "mean_official_signature_coverage": mean_coverage,
    }
    gates = {
        **base_gates,
        "mean_signature_coverage_at_least_5": mean_coverage >= 5.0,
        "d2_spearman_at_least_0_25": (
            correlation is not None and correlation >= 0.25
        ),
        "d2_setup_selected_at_least_8": d2_setup_count >= 8,
        "d1_never_selects_setup": d1_setup_count == 0,
        "d2_regret_improves_by_0_15": (
            mean_d1_regret - mean_d2_regret >= 0.15
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
    tasks = _load_tau2_tasks(t3_path)
    tasks_by_faults = _task_index(tasks)
    backbones = selected_backbones(stage)
    if len(config.model_pairs) != 1:
        raise ValueError("Tau2 MMS gate requires exactly one model pair")
    model = build_model_adapter(config.model_pairs[0].questioner, config)
    raw: dict[str, Any] = {}

    hypothesis_raw = model.chat_complete_messages_batched(
        [hypothesis_messages(backbone) for backbone in backbones],
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["hypotheses"] = hypothesis_raw
    _write_raw_checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
    hypotheses_many = [parse_hypotheses(value) for value in hypothesis_raw]

    flat_keys: list[tuple[int, str]] = []
    rollout_prompts: list[list[dict[str, str]]] = []
    for case_index, (backbone, hypotheses) in enumerate(
        zip(backbones, hypotheses_many, strict=True)
    ):
        for action in ROOT_ACTIONS:
            flat_keys.append((case_index, action))
            rollout_prompts.append(
                rollout_messages(hypotheses, action, backbone)
            )
    rollout_raw = model.chat_complete_messages_batched(
        rollout_prompts,
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["rollouts"] = rollout_raw
    _write_raw_checkpoint(raw_checkpoint_path, stage=stage, raw=raw)

    parsed_rollouts: dict[tuple[int, str], dict[str, Any]] = {}
    for response, (case_index, action) in zip(
        rollout_raw, flat_keys, strict=True
    ):
        parsed_rollouts[case_index, action] = parse_rollout(
            response,
            hypotheses=hypotheses_many[case_index],
            root_action=action,
        )

    records: list[dict[str, Any]] = []
    for case_index, (backbone, hypotheses) in enumerate(
        zip(backbones, hypotheses_many, strict=True)
    ):
        observations = exact_observations_for_backbone(
            t3_path,
            tasks_by_faults,
            backbone,
        )
        exact_values = exact_action_values(observations)
        action_rows: list[dict[str, Any]] = []
        actions_by_id: dict[str, dict[str, Any]] = {}
        for action in ROOT_ACTIONS:
            row = {
                "action_id": action,
                **parsed_rollouts[case_index, action],
                **exact_values[action],
            }
            action_rows.append(row)
            actions_by_id[action] = row
        records.append(
            {
                "case_index": case_index,
                "backbone": list(backbone),
                "hypotheses": hypotheses,
                "official_signature_coverage": official_signature_coverage(
                    hypotheses
                ),
                "exact_setup_d1_information": exact_values[
                    "installed_apps"
                ]["exact_d1_information"],
                "exact_setup_d2_information": exact_values[
                    "installed_apps"
                ]["exact_d2_information"],
                "exact_best_direct_d2_information": max(
                    value["exact_d2_information"]
                    for action, value in exact_values.items()
                    if action != "installed_apps"
                ),
                "actions": action_rows,
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
            "backbones": [list(value) for value in backbones],
            "fault_variants": list(FAULT_VARIANTS),
            "hypothesis_count": HYPOTHESIS_COUNT,
            "root_actions": list(ROOT_ACTIONS),
            "root_action_count": len(ROOT_ACTIONS),
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
        config.openrouter_projected_cost_usd = 0.25
        config.openrouter_run_budget_usd = 2.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_run_dir = args.private_raw_dir / args.run_id
    private_run_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_run_dir / "RAW_RESPONSES.json"
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
