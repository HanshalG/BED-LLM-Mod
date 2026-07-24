#!/usr/bin/env python3
"""Evaluate a joint LLM semantic likelihood table on the Tau2 account unlock."""

from __future__ import annotations

import argparse
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
from scripts.tau2_account_only_v4 import (
    ACCOUNT_HYPOTHESES,
    ACCOUNT_WORLD_IDS,
)
from scripts.tau2_account_prerequisite_ranking_gate import (
    FORMAL_PROMPTS as V4_FORMAL_PROMPTS,
    ROOT_ACTIONS,
    UNLOCKED_ACTIONS,
    best_two_step_information,
    exact_action_values,
    exact_observations,
)
from scripts.tau2_mms_prerequisite_ranking_gate import (
    information_gain,
    spearman,
)


SCHEMA_VERSION = 5
SELECTION_SEED = 24321

SMOKE_PROMPTS = (
    "During a fixed airplane-mode session, the subscriber cannot use carrier data.",
    "A mobile-data outage is being diagnosed while the handset stays in airplane mode.",
)

FORMAL_PROMPTS = (
    "Diagnose unavailable cellular data while airplane mode remains fixed on.",
    "The subscriber has no mobile internet during an airplane-mode diagnostic window.",
    "Carrier data cannot be used while the phone is held in airplane mode.",
    "Investigate the line account while cellular radios remain disabled.",
    "The customer's mobile-data service is unavailable in a fixed offline handset state.",
    "Resolve a carrier-data failure without changing the phone's airplane-mode setting.",
    "The line cannot provide cellular internet during this account diagnostic.",
    "Mobile data is inaccessible while device-side radio settings stay fixed.",
    "Check why carrier internet is unavailable under an unchanged airplane-mode state.",
    "The subscriber reports no cellular data during a constrained remote diagnostic.",
    "Assess the carrier line behind a data outage while the handset remains offline.",
    "Investigate absent mobile internet without altering the known phone state.",
)

CONFIRMATION_PROMPTS = (
    "A subscriber cannot access mobile data during a fixed airplane-mode check.",
    "Carrier internet is down while the device state must remain unchanged.",
    "Determine the account condition behind unavailable cellular data.",
    "The handset remains offline while the carrier line is investigated.",
    "Mobile internet is unavailable during a remote account inspection.",
    "The customer needs a line diagnosis without changing phone settings.",
    "Cellular data cannot be reached in the current fixed device state.",
    "Investigate the subscription behind an ongoing mobile-data outage.",
    "The line has no usable carrier internet during this diagnostic.",
    "Account and line records must be checked while the phone remains offline.",
    "The subscriber's cellular data service is currently inaccessible.",
    "Diagnose a carrier-data failure under an unchanged handset state.",
    "Mobile internet remains unavailable throughout the remote inspection.",
    "The customer's line cannot deliver cellular data at present.",
    "Inspect the account cause of an unavailable mobile-data connection.",
    "The phone state is fixed while the subscriber's data service is diagnosed.",
    "Carrier data is unavailable and only diagnostic reads may be performed.",
    "The line's mobile internet service cannot currently be used.",
    "Investigate the subscriber record behind a cellular-data outage.",
    "No carrier internet is available during this constrained diagnostic.",
    "The customer reports an unresolved mobile-data service failure.",
    "Assess the line while cellular connectivity remains unavailable.",
    "The device cannot use mobile internet during the account investigation.",
    "Find the carrier-side state behind absent cellular data.",
)

ALL_ACTIONS = {
    **{
        action_id: str(spec["description"])
        for action_id, spec in ROOT_ACTIONS.items()
    },
    **UNLOCKED_ACTIONS,
}


def stage_prompts(stage: str) -> tuple[str, ...]:
    if stage == "serving_smoke":
        return SMOKE_PROMPTS
    if stage == "formal":
        return FORMAL_PROMPTS
    if stage == "confirmation":
        return CONFIRMATION_PROMPTS
    raise ValueError("stage must be serving_smoke, formal, or confirmation")


def joint_likelihood_messages(ticket: str) -> list[dict[str, str]]:
    schema = {
        "actions": [
            {
                "action_id": action_id,
                "predictions": [
                    {
                        "hypothesis_id": hypothesis["id"],
                        "outcome": "short observable category",
                    }
                    for hypothesis in ACCOUNT_HYPOTHESES
                ],
            }
            for action_id in ALL_ACTIONS
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "Predict observable diagnostic results under semantic hypotheses. "
                "Return strict JSON only, with no reasoning text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Ticket: {ticket}\n"
                "Airplane mode is on and cannot be changed. The four hypotheses "
                "describe the complete uncertainty. For every action, predict the "
                "literal observable result under every hypothesis. Reuse exactly "
                "the same short category whenever the physical action result would "
                "be observationally identical. Do not report a diagnosis, hidden "
                "state, utility, entropy, information gain, or preferred policy. "
                "Do not let wording of the ticket change the specified hypotheses. "
                "The three ID-dependent actions are evaluated after customer_lookup "
                "has supplied the customer and line IDs.\n"
                f"Hypotheses: {json.dumps(ACCOUNT_HYPOTHESES, separators=(',', ':'))}\n"
                f"Actions: {json.dumps(ALL_ACTIONS, separators=(',', ':'))}\n"
                "Return every action and hypothesis exactly once in the supplied "
                "order using this exact schema:\n"
                + json.dumps(schema, separators=(",", ":"))
            ),
        },
    ]


def _normalized_outcome(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def parse_joint_likelihood(text: str) -> dict[str, Any]:
    payload = _parse_json_object(text)
    actions = payload.get("actions")
    action_ids = list(ALL_ACTIONS)
    hypothesis_ids = [row["id"] for row in ACCOUNT_HYPOTHESES]
    if not isinstance(actions, list) or len(actions) != len(action_ids):
        raise ValueError("joint table must contain every action exactly once")

    observations = {hypothesis_id: {} for hypothesis_id in hypothesis_ids}
    parsed_actions: list[dict[str, Any]] = []
    seen_actions: set[str] = set()
    for expected_action, row in zip(action_ids, actions, strict=True):
        if not isinstance(row, dict):
            raise ValueError("joint action row must be an object")
        action_id = str(row.get("action_id", "")).strip()
        if action_id != expected_action or action_id in seen_actions:
            raise ValueError("joint action IDs or order changed")
        seen_actions.add(action_id)
        predictions = row.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(
            hypothesis_ids
        ):
            raise ValueError("joint action lost a hypothesis")
        parsed_predictions: list[dict[str, str]] = []
        seen_hypotheses: set[str] = set()
        for expected_hypothesis, prediction in zip(
            hypothesis_ids, predictions, strict=True
        ):
            if not isinstance(prediction, dict):
                raise ValueError("joint prediction must be an object")
            hypothesis_id = str(prediction.get("hypothesis_id", "")).strip()
            outcome = _normalized_outcome(prediction.get("outcome", ""))
            if (
                hypothesis_id != expected_hypothesis
                or hypothesis_id in seen_hypotheses
                or not outcome
            ):
                raise ValueError("joint hypothesis IDs, order, or outcome changed")
            seen_hypotheses.add(hypothesis_id)
            observations[hypothesis_id][action_id] = outcome
            parsed_predictions.append(
                {"hypothesis_id": hypothesis_id, "outcome": outcome}
            )
        parsed_actions.append(
            {"action_id": action_id, "predictions": parsed_predictions}
        )
    return {"actions": parsed_actions, "observations": observations}


def _legal_followups(root_action: str) -> list[str]:
    legal = [action for action in ROOT_ACTIONS if action != root_action]
    if root_action == "customer_lookup":
        legal.extend(UNLOCKED_ACTIONS)
    return legal


def predicted_action_values(
    observations: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    hypotheses = list(observations)
    values: dict[str, dict[str, Any]] = {}
    for action in ROOT_ACTIONS:
        d2, branches = best_two_step_information(observations, action)
        values[action] = {
            "predicted_d1_information": information_gain(
                [observations[hypothesis][action] for hypothesis in hypotheses]
            ),
            "predicted_d2_information": d2,
            "predicted_d2_branch_actions": branches,
        }
    return values


def _selected_action(
    values: dict[str, dict[str, Any]],
    score_key: str,
) -> str:
    return max(
        values,
        key=lambda action: (float(values[action][score_key]), action),
    )


def _single_branch_followup(
    observations: dict[str, dict[str, str]],
    values: dict[str, dict[str, Any]],
    root_action: str,
) -> str:
    root_outcomes = {
        observations[hypothesis][root_action] for hypothesis in observations
    }
    if len(root_outcomes) != 1:
        raise ValueError(
            "confirmation requires one predicted root category for execution"
        )
    root_outcome = next(iter(root_outcomes))
    return str(values[root_action]["predicted_d2_branch_actions"][root_outcome])


def _greedy_followup(
    observations: dict[str, dict[str, str]],
    root_action: str,
) -> str:
    hypotheses = list(observations)
    legal = _legal_followups(root_action)
    return max(
        legal,
        key=lambda action: (
            information_gain(
                [observations[hypothesis][action] for hypothesis in hypotheses]
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
    confidence: float = 0.90,
    draws: int = 5000,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
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
    root_equivalent = sum(record["all_root_actions_equivalent"] for record in records)
    line_details_four = sum(
        record["line_details_unique_outcomes"] == 4 for record in records
    )
    d1_lookup = sum(record["d1_selected_action"] == "customer_lookup" for record in records)
    d2_lookup = sum(record["d2_selected_action"] == "customer_lookup" for record in records)
    lookup_line_details = sum(
        record["lookup_followup_action"] == "line_details" for record in records
    )
    base_gates = {
        "all_prompt_variants_completed": len(records) == expected,
        "exact_physical_request_count": int(usage["physical_requests"]) == expected,
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_scores_finite": all(
            np.isfinite(action["predicted_d2_information"])
            for record in records
            for action in record["actions"]
        ),
    }
    summary: dict[str, Any] = {
        "num_prompt_variants": len(records),
        "all_root_actions_equivalent_count": root_equivalent,
        "line_details_four_outcomes_count": line_details_four,
        "d1_lookup_selected_count": d1_lookup,
        "d2_lookup_selected_count": d2_lookup,
        "lookup_uses_line_details_count": lookup_line_details,
    }
    if stage == "serving_smoke":
        gates = {
            **base_gates,
            "all_roots_equivalent_both": root_equivalent == 2,
            "line_details_four_outcomes_both": line_details_four == 2,
            "d1_never_selects_lookup": d1_lookup == 0,
            "d2_selects_lookup_both": d2_lookup == 2,
            "lookup_uses_line_details_both": lookup_line_details == 2,
        }
        gates["all_pass"] = all(gates.values())
        summary["gates"] = gates
        return summary

    predicted: list[float] = []
    exact: list[float] = []
    d1_regrets: list[float] = []
    d2_regrets: list[float] = []
    wins = 0
    for record in records:
        actions = record["actions"]
        predicted.extend(
            float(action["predicted_d2_information"]) for action in actions
        )
        exact.extend(float(action["exact_d2_information"]) for action in actions)
        oracle = max(float(action["exact_d2_information"]) for action in actions)
        d1 = record["actions_by_id"][record["d1_selected_action"]]
        d2 = record["actions_by_id"][record["d2_selected_action"]]
        d1_regret = oracle - float(d1["exact_d2_information"])
        d2_regret = oracle - float(d2["exact_d2_information"])
        d1_regrets.append(d1_regret)
        d2_regrets.append(d2_regret)
        wins += d2_regret + 1.0e-12 < d1_regret
    correlation = spearman(predicted, exact)
    mean_d1_regret = float(np.mean(d1_regrets))
    mean_d2_regret = float(np.mean(d2_regrets))
    summary.update(
        {
            "predicted_d2_spearman_vs_exact_d2": correlation,
            "d2_beats_d1_count": wins,
            "mean_d1_top1_regret_nats": mean_d1_regret,
            "mean_d2_top1_regret_nats": mean_d2_regret,
            "mean_regret_improvement_nats": mean_d1_regret - mean_d2_regret,
        }
    )
    gates = {
        **base_gates,
        "all_roots_equivalent_at_least_10": root_equivalent >= 10,
        "line_details_four_outcomes_at_least_10": line_details_four >= 10,
        "d1_never_selects_lookup": d1_lookup == 0,
        "d2_selects_lookup_at_least_10": d2_lookup >= 10,
        "lookup_uses_line_details_at_least_10": lookup_line_details >= 10,
        "d2_spearman_at_least_0_50": (
            correlation is not None and correlation >= 0.50
        ),
        "d2_regret_improves_by_1_00": (
            mean_d1_regret - mean_d2_regret >= 1.00
        ),
        "d2_beats_d1_at_least_10": wins >= 10,
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
    d1_truth_gain = [
        record["d2_truth_log_posterior"]
        - record["d1_truth_log_posterior"]
        for record in records
    ]
    random_truth_gain = [
        record["d2_truth_log_posterior"]
        - record["random_truth_log_posterior"]
        for record in records
    ]
    d1_ci = _bootstrap_mean_ci(d1_minus_d2, seed=SELECTION_SEED + 101)
    random_ci = _bootstrap_mean_ci(random_minus_d2, seed=SELECTION_SEED + 102)
    d1_truth_ci = _bootstrap_mean_ci(d1_truth_gain, seed=SELECTION_SEED + 103)
    random_truth_ci = _bootstrap_mean_ci(
        random_truth_gain, seed=SELECTION_SEED + 104
    )
    d2_wins_d1 = sum(value > 1.0e-12 for value in d1_minus_d2)
    d2_wins_random = sum(value > 1.0e-12 for value in random_minus_d2)
    d2_lookup = sum(record["d2_root_action"] == "customer_lookup" for record in records)
    summary = {
        "num_trajectories": len(records),
        "d2_lookup_selected_count": d2_lookup,
        "d2_wins_d1_count": d2_wins_d1,
        "d2_wins_random_count": d2_wins_random,
        "mean_d1_minus_d2_final_entropy_nats": float(np.mean(d1_minus_d2)),
        "d1_minus_d2_final_entropy_90ci": list(d1_ci),
        "mean_random_minus_d2_final_entropy_nats": float(
            np.mean(random_minus_d2)
        ),
        "random_minus_d2_final_entropy_90ci": list(random_ci),
        "mean_d2_minus_d1_truth_log_posterior": float(np.mean(d1_truth_gain)),
        "d2_minus_d1_truth_log_posterior_90ci": list(d1_truth_ci),
        "mean_d2_minus_random_truth_log_posterior": float(
            np.mean(random_truth_gain)
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
        "d2_selects_lookup_at_least_20": d2_lookup >= 20,
        "d2_wins_d1_at_least_20": d2_wins_d1 >= 20,
        "d2_wins_random_at_least_16": d2_wins_random >= 16,
        "d1_entropy_gain_at_least_1_00": float(np.mean(d1_minus_d2)) >= 1.00,
        "d1_entropy_gain_ci_positive": d1_ci[0] > 0.0,
        "random_entropy_gain_at_least_0_50": float(np.mean(random_minus_d2))
        >= 0.50,
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
    if len(config.model_pairs) != 1:
        raise ValueError("Tau2 account V5 requires exactly one model pair")
    model = build_model_adapter(config.model_pairs[0].questioner, config)
    raw = model.chat_complete_messages_batched(
        [joint_likelihood_messages(prompt) for prompt in prompts],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    _write_raw(raw_checkpoint_path, stage, raw)
    if len(raw) != len(prompts):
        raise ValueError("Tau2 account V5 response count changed")
    parsed = [parse_joint_likelihood(response) for response in raw]

    official_observations = exact_observations(Path(t3_dir))
    official_observations = {
        world_id: official_observations[world_id]
        for world_id in ACCOUNT_WORLD_IDS
    }
    exact_values = exact_action_values(official_observations)
    records: list[dict[str, Any]] = []
    truth_schedule = [
        ACCOUNT_WORLD_IDS[index % len(ACCOUNT_WORLD_IDS)]
        for index in range(len(prompts))
    ]
    random.Random(SELECTION_SEED).shuffle(truth_schedule)

    for prompt_index, (prompt, joint) in enumerate(
        zip(prompts, parsed, strict=True)
    ):
        observations = joint["observations"]
        values = predicted_action_values(observations)
        actions = []
        actions_by_id = {}
        for action_id in ROOT_ACTIONS:
            row = {
                "action_id": action_id,
                **values[action_id],
                **exact_values[action_id],
            }
            actions.append(row)
            actions_by_id[action_id] = row
        d1_root = _selected_action(values, "predicted_d1_information")
        d2_root = _selected_action(values, "predicted_d2_information")
        root_equivalent = all(
            len(
                {
                    observations[hypothesis][action_id]
                    for hypothesis in observations
                }
            )
            == 1
            for action_id in ROOT_ACTIONS
        )
        line_details_unique = len(
            {
                observations[hypothesis]["line_details"]
                for hypothesis in observations
            }
        )
        lookup_followup = _single_branch_followup(
            observations, values, "customer_lookup"
        )
        record: dict[str, Any] = {
            "prompt_variant": prompt_index,
            "ticket": prompt,
            "hypotheses": list(ACCOUNT_HYPOTHESES),
            "joint_predictions": joint["actions"],
            "all_root_actions_equivalent": root_equivalent,
            "line_details_unique_outcomes": line_details_unique,
            "lookup_followup_action": lookup_followup,
            "d1_selected_action": d1_root,
            "d2_selected_action": d2_root,
            "actions": actions,
            "actions_by_id": actions_by_id,
        }
        if stage == "confirmation":
            d1_followup = _greedy_followup(observations, d1_root)
            d2_followup = _single_branch_followup(observations, values, d2_root)
            rng = random.Random(SELECTION_SEED + 1000 + prompt_index)
            random_root = rng.choice(list(ROOT_ACTIONS))
            random_followup = rng.choice(_legal_followups(random_root))
            truth = truth_schedule[prompt_index]
            initial_entropy = math.log(len(official_observations))

            def endpoint(root: str, followup: str) -> tuple[float, float, float]:
                root_information = information_gain(
                    [
                        official_observations[world][root]
                        for world in official_observations
                    ]
                )
                total_information = sequence_information(
                    official_observations, root, followup
                )
                entropy_after_root = initial_entropy - root_information
                final_entropy = initial_entropy - total_information
                entropy_auc = (
                    0.5 * initial_entropy
                    + entropy_after_root
                    + 0.5 * final_entropy
                )
                truth_log_posterior = _truth_log_posterior(
                    official_observations, truth, root, followup
                )
                return final_entropy, entropy_auc, truth_log_posterior

            d1_endpoint = endpoint(d1_root, d1_followup)
            d2_endpoint = endpoint(d2_root, d2_followup)
            random_endpoint = endpoint(random_root, random_followup)
            record.update(
                {
                    "truth_world_id": truth,
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
            "posthoc_development_after_v4": True,
            "v4_formal_prompts_not_reused": not any(
                prompt in V4_FORMAL_PROMPTS for prompt in prompts
            ),
            "fixed_account_support": True,
            "support_size": len(ACCOUNT_HYPOTHESES),
            "account_world_ids": list(ACCOUNT_WORLD_IDS),
            "llm_predicts_joint_semantic_likelihood_table": True,
            "exact_planner_receives_no_official_outputs": True,
            "official_simulator_outputs_hidden_until_evaluation": True,
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
