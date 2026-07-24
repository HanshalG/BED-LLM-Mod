#!/usr/bin/env python3
"""Gate path-dependent suspect beliefs in PAPRIKA murder mysteries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.musique_answer_belief_bridge_gate import (
    BELIEF_SIZE,
    _belief_schema,
    _build_models,
    _checkpoint,
    _normalized,
    _usage_snapshot,
    equivalence_messages,
    parse_belief_response,
    parse_equivalence,
    truth_probability,
)


SCHEMA_VERSION = 1
PAPRIKA_SHA256 = "2ae865e60662135f6e867a95653be48ecc8b6a51cddc5c3e46e2538ae24b2184"
SELECTION_SEED = 24329
DIRECT_ACTION_COUNT = 3
ENABLING_ACTION_COUNT = 3
FIRST_ACTION_COUNT = DIRECT_ACTION_COUNT + ENABLING_ACTION_COUNT
SECOND_ACTION_COUNT = 4
SMOKE_INDICES = (38, 47)
FORMAL_INDICES = (31, 27, 46, 39, 12, 18)
RESERVE_INDICES = (19, 44, 43, 25, 20, 32)
EXPECTED_REQUESTS = {
    "serving_smoke": len(SMOKE_INDICES)
    * (
        1
        + FIRST_ACTION_COUNT
        + FIRST_ACTION_COUNT
        + FIRST_ACTION_COUNT * SECOND_ACTION_COUNT
        + FIRST_ACTION_COUNT * SECOND_ACTION_COUNT
        + 1
        + 1
    ),
    "opportunity": len(FORMAL_INDICES)
    * (
        1
        + FIRST_ACTION_COUNT
        + FIRST_ACTION_COUNT
        + FIRST_ACTION_COUNT * SECOND_ACTION_COUNT
        + FIRST_ACTION_COUNT * SECOND_ACTION_COUNT
        + 1
        + 1
    ),
}


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_indices(stage: str) -> tuple[int, ...]:
    if stage == "serving_smoke":
        return SMOKE_INDICES
    if stage == "opportunity":
        return FORMAL_INDICES
    raise ValueError("stage must be serving_smoke or opportunity")


def culprit_reference(hidden_scenario: str) -> str:
    match = re.search(
        r"(?:hidden culprit|true culprit|murderer|killer) is "
        r"(.+?)(?:(?:, )?who |\.(?: Key| The key| key))",
        hidden_scenario,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("could not parse explicit PAPRIKA culprit reference")
    reference = " ".join(match.group(1).strip(" ,").split())
    if not reference:
        raise ValueError("parsed PAPRIKA culprit reference is empty")
    return reference


def load_selected_cases(
    data_path: str | Path,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    path = Path(data_path)
    if _sha256(path) != PAPRIKA_SHA256:
        raise ValueError("PAPRIKA murder config hash does not match")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("eval")
    if not isinstance(rows, list) or len(rows) != 50:
        raise ValueError("PAPRIKA eval split must contain exactly 50 cases")
    reproduced = tuple(
        int(value)
        for value in np.random.default_rng(SELECTION_SEED).choice(
            len(rows),
            size=len(SMOKE_INDICES + FORMAL_INDICES + RESERVE_INDICES),
            replace=False,
        )
    )
    if reproduced != SMOKE_INDICES + FORMAL_INDICES + RESERVE_INDICES:
        raise ValueError("frozen PAPRIKA target-blind draw does not reproduce")
    selected = [rows[index] for index in selected_indices(stage)]
    for row in selected:
        if not isinstance(row.get("agent"), str) or not isinstance(
            row.get("env"), str
        ):
            raise ValueError("PAPRIKA case is malformed")
        culprit_reference(row["env"])
    return payload, selected


def investigator_messages(
    public_scenario: str,
    *,
    history: Sequence[tuple[str, str]] = (),
    previous_belief: Sequence[dict[str, Any]] | None = None,
    request_actions: bool = False,
) -> list[dict[str, str]]:
    payload: dict[str, Any] = {
        "public_case": public_scenario,
        "investigation_history": [
            {"action": action, "response": response}
            for action, response in history
        ],
    }
    if previous_belief is not None:
        payload["previous_suspect_belief_for_context"] = list(previous_belief)
    schema: dict[str, Any] = {"belief": _belief_schema()}
    if request_actions and not history:
        schema["direct_actions"] = [
            f"single precise suspect-discriminating action {index + 1}"
            for index in range(DIRECT_ACTION_COUNT)
        ]
        schema["enabling_actions"] = [
            f"single precise evidence-unlocking action {index + 1}"
            for index in range(ENABLING_ACTION_COUNT)
        ]
    elif request_actions:
        schema["next_actions"] = [
            f"single observation-conditioned action {index + 1}"
            for index in range(SECOND_ACTION_COUNT)
        ]
    if request_actions and not history:
        action_instruction = (
            f" Also propose exactly {DIRECT_ACTION_COUNT} diverse direct actions "
            "that could immediately discriminate among named suspects, and exactly "
            f"{ENABLING_ACTION_COUNT} diverse enabling investigations intended to "
            "uncover records, objects, people, or contradictions that create a "
            "more decisive follow-up. Each item must be one concrete action or "
            "question the detective can take now."
        )
    elif request_actions:
        action_instruction = (
            f" Also propose exactly {SECOND_ACTION_COUNT} diverse follow-up actions "
            "that use specific information newly revealed in the latest response. "
            "Do not repeat an earlier action."
        )
    else:
        action_instruction = ""
    return [
        {
            "role": "system",
            "content": (
                "Maintain a target-blind probabilistic belief over possible culprits "
                "using only the public case and observed investigation history. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Build the suspect belief afresh. Return exactly {BELIEF_SIZE} "
                "distinct concrete person or role strings and probabilities summing "
                "to 1. Include witnesses or background characters when plausible; "
                "do not assume the culprit is among the primary suspects. Do not "
                "invent evidence or use any hidden scenario."
                f"{action_instruction} Return exactly this schema: "
                + json.dumps(schema, ensure_ascii=True, separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_investigator_response(
    text: str,
    *,
    stage: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    belief, _unused = parse_belief_response(text, available_doc_ids=())
    payload = _parse_json_object(text)
    if stage == "initial":
        direct = payload.get("direct_actions")
        enabling = payload.get("enabling_actions")
        if (
            not isinstance(direct, list)
            or len(direct) != DIRECT_ACTION_COUNT
            or not isinstance(enabling, list)
            or len(enabling) != ENABLING_ACTION_COUNT
        ):
            raise ValueError("initial action groups have the wrong size")
        actions = [*direct, *enabling]
    elif stage == "followup":
        actions = payload.get("next_actions")
        if not isinstance(actions, list) or len(actions) != SECOND_ACTION_COUNT:
            raise ValueError("next_actions has the wrong size")
    else:
        raise ValueError("investigator stage must be initial or followup")
    if (
        any(
            not isinstance(action, str) or not _normalized(action)
            for action in actions
        )
        or len({_normalized(action) for action in actions}) != len(actions)
    ):
        raise ValueError("actions must be distinct nonempty strings")
    return belief, [" ".join(action.split()) for action in actions]


def environment_messages(
    game_config: dict[str, Any],
    hidden_scenario: str,
    *,
    history: Sequence[tuple[str, str]],
    action: str,
) -> list[dict[str, str]]:
    system = str(game_config["env"]).format(env=hidden_scenario).strip()
    optional = str(game_config["env_optional_message"]).format(
        env=hidden_scenario
    ).strip()
    transcript = "\n\n".join(
        f"Detective action {index}: {past_action}\n"
        f"Environment response {index}: {past_response}"
        for index, (past_action, past_response) in enumerate(history, start=1)
    )
    if transcript:
        transcript += "\n\n"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                transcript
                + f"Detective action {len(history) + 1}: {action}\n\n"
                + optional
                + "\nRespond only with the concise in-world result of this action."
            ),
        },
    ]


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    one = [
        float(branch["truth_probability"])
        for branch in record["first_branches"]
    ]
    pair_values = [
        [
            float(second["truth_probability"])
            for second in branch["second_branches"]
        ]
        for branch in record["first_branches"]
    ]
    greedy_first = max(
        range(FIRST_ACTION_COUNT),
        key=lambda index: (one[index], -index),
    )
    oracle_first, oracle_second = max(
        (
            (first_index, second_index)
            for first_index in range(FIRST_ACTION_COUNT)
            for second_index in range(SECOND_ACTION_COUNT)
        ),
        key=lambda pair: (
            pair_values[pair[0]][pair[1]],
            -pair[0],
            -pair[1],
        ),
    )
    greedy_continuation = max(pair_values[greedy_first])
    replay_gap = abs(
        float(record["replay_truth_probability"])
        - pair_values[0][0]
    )
    return {
        "distinct_first_response_count": len(
            {
                branch["response_sha256"]
                for branch in record["first_branches"]
            }
        ),
        "one_step_truth_probability_spread": max(one) - min(one),
        "greedy_first_index": greedy_first,
        "oracle_first_index": oracle_first,
        "oracle_second_index": oracle_second,
        "oracle_first_differs_from_greedy": oracle_first != greedy_first,
        "best_one_step_truth_probability": max(one),
        "oracle_pair_truth_probability": pair_values[oracle_first][oracle_second],
        "greedy_continuation_truth_probability": greedy_continuation,
        "pair_gain_over_best_one_step": (
            pair_values[oracle_first][oracle_second] - max(one)
        ),
        "nonmyopic_probability_gap": (
            pair_values[oracle_first][oracle_second] - greedy_continuation
        ),
        "replay_truth_probability_gap": replay_gap,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    diagnostics = [analyze_record(record) for record in records]
    base_gates = {
        "all_cases_complete": len(records) == len(selected_indices(stage)),
        "exact_physical_request_count": int(usage["physical_requests"])
        == EXPECTED_REQUESTS[stage],
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_replay_gaps_finite": all(
            math.isfinite(row["replay_truth_probability_gap"])
            for row in diagnostics
        ),
    }
    if stage == "serving_smoke":
        gates = {
            **base_gates,
            "all_beliefs_size_8": all(
                record["all_belief_sizes_valid"] for record in records
            ),
            "at_least_3_distinct_first_responses_each": all(
                row["distinct_first_response_count"] >= 3
                for row in diagnostics
            ),
        }
        gates["all_pass"] = all(gates.values())
        return {
            "num_cases": len(records),
            "case_diagnostics": diagnostics,
            "gates": gates,
        }

    initial_mean = float(
        np.mean([record["initial_truth_probability"] for record in records])
    )
    distinct_mean = float(
        np.mean([row["distinct_first_response_count"] for row in diagnostics])
    )
    spread_count = sum(
        row["one_step_truth_probability_spread"] >= 0.10
        for row in diagnostics
    )
    first_differs_count = sum(
        row["oracle_first_differs_from_greedy"] for row in diagnostics
    )
    pair_gain_count = sum(
        row["pair_gain_over_best_one_step"] >= 0.10 for row in diagnostics
    )
    gap_count = sum(
        row["nonmyopic_probability_gap"] >= 0.10 for row in diagnostics
    )
    mean_gain = float(
        np.mean([row["pair_gain_over_best_one_step"] for row in diagnostics])
    )
    mean_gap = float(
        np.mean([row["nonmyopic_probability_gap"] for row in diagnostics])
    )
    replay_gaps = [
        row["replay_truth_probability_gap"] for row in diagnostics
    ]
    summary = {
        "num_cases": len(records),
        "mean_initial_truth_probability": initial_mean,
        "mean_distinct_first_response_count": distinct_mean,
        "one_step_spread_at_least_0_10_count": spread_count,
        "oracle_first_differs_from_greedy_count": first_differs_count,
        "pair_gain_at_least_0_10_count": pair_gain_count,
        "nonmyopic_gap_at_least_0_10_count": gap_count,
        "mean_pair_gain_over_best_one_step": mean_gain,
        "mean_nonmyopic_probability_gap": mean_gap,
        "mean_replay_truth_probability_gap": float(np.mean(replay_gaps)),
        "max_replay_truth_probability_gap": max(replay_gaps),
        "case_diagnostics": [
            {"case_index": record["case_index"], **diagnostic}
            for record, diagnostic in zip(records, diagnostics, strict=True)
        ],
    }
    gates = {
        **base_gates,
        "initial_belief_not_saturated": initial_mean <= 0.35,
        "mean_distinct_first_responses_at_least_4": distinct_mean >= 4.0,
        "one_step_spread_count_at_least_4": spread_count >= 4,
        "oracle_first_differs_count_at_least_2": first_differs_count >= 2,
        "pair_gain_count_at_least_3": pair_gain_count >= 3,
        "nonmyopic_gap_count_at_least_2": gap_count >= 2,
        "mean_pair_gain_at_least_0_10": mean_gain >= 0.10,
        "mean_nonmyopic_gap_at_least_0_07": mean_gap >= 0.07,
        "mean_replay_gap_at_most_0_10": float(np.mean(replay_gaps)) <= 0.10,
        "max_replay_gap_at_most_0_25": max(replay_gaps) <= 0.25,
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def run_gate(
    config: Config,
    *,
    data_path: str | Path,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    game_config, cases = load_selected_cases(data_path, stage)
    if len(config.model_pairs) != 1:
        raise ValueError("PAPRIKA murder gate requires one model pair")
    generator, environment = _build_models(config)
    raw: dict[str, Any] = {}
    try:
        initial_raw = generator.chat_complete_messages_batched(
            [
                investigator_messages(
                    case["agent"],
                    request_actions=True,
                )
                for case in cases
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["initial"] = initial_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        initial = [
            parse_investigator_response(text, stage="initial")
            for text in initial_raw
        ]

        first_keys: list[tuple[int, int]] = []
        first_actions: dict[tuple[int, int], str] = {}
        first_env_messages = []
        for case_offset, (belief, actions) in enumerate(initial):
            shuffled = list(actions)
            np.random.default_rng(
                SELECTION_SEED * 1000 + case_offset
            ).shuffle(shuffled)
            initial[case_offset] = (belief, shuffled)
            for first_index, action in enumerate(shuffled):
                key = (case_offset, first_index)
                first_keys.append(key)
                first_actions[key] = action
                first_env_messages.append(
                    environment_messages(
                        game_config,
                        cases[case_offset]["env"],
                        history=(),
                        action=action,
                    )
                )
        first_env_raw = environment.chat_complete_messages_batched(
            first_env_messages,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["first_environment"] = first_env_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        first_responses = {
            key: response.strip()
            for key, response in zip(
                first_keys, first_env_raw, strict=True
            )
        }
        if any(not response for response in first_responses.values()):
            raise ValueError("environment returned an empty first response")

        first_belief_messages = [
            investigator_messages(
                cases[case_offset]["agent"],
                history=(
                    (
                        first_actions[case_offset, first_index],
                        first_responses[case_offset, first_index],
                    ),
                ),
                previous_belief=initial[case_offset][0],
                request_actions=True,
            )
            for case_offset, first_index in first_keys
        ]
        first_belief_raw = generator.chat_complete_messages_batched(
            first_belief_messages,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["first_beliefs"] = first_belief_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        first_parsed = [
            parse_investigator_response(text, stage="followup")
            for text in first_belief_raw
        ]
        first_lookup: dict[
            tuple[int, int], tuple[list[dict[str, Any]], list[str]]
        ] = {}
        for key, (belief, actions) in zip(
            first_keys, first_parsed, strict=True
        ):
            case_offset, first_index = key
            shuffled = list(actions)
            np.random.default_rng(
                SELECTION_SEED * 1_000_000
                + case_offset * 100
                + first_index
            ).shuffle(shuffled)
            first_lookup[key] = (belief, shuffled)

        second_keys: list[tuple[int, int, int]] = []
        second_actions: dict[tuple[int, int, int], str] = {}
        second_env_messages = []
        for case_offset, first_index in first_keys:
            _belief, actions = first_lookup[case_offset, first_index]
            first_history = (
                (
                    first_actions[case_offset, first_index],
                    first_responses[case_offset, first_index],
                ),
            )
            for second_index, action in enumerate(actions):
                key = (case_offset, first_index, second_index)
                second_keys.append(key)
                second_actions[key] = action
                second_env_messages.append(
                    environment_messages(
                        game_config,
                        cases[case_offset]["env"],
                        history=first_history,
                        action=action,
                    )
                )
        second_env_raw = environment.chat_complete_messages_batched(
            second_env_messages,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["second_environment"] = second_env_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        second_responses = {
            key: response.strip()
            for key, response in zip(
                second_keys, second_env_raw, strict=True
            )
        }
        if any(not response for response in second_responses.values()):
            raise ValueError("environment returned an empty second response")

        final_messages = []
        for case_offset, first_index, second_index in second_keys:
            first_belief, _actions = first_lookup[
                case_offset, first_index
            ]
            final_messages.append(
                investigator_messages(
                    cases[case_offset]["agent"],
                    history=(
                        (
                            first_actions[case_offset, first_index],
                            first_responses[case_offset, first_index],
                        ),
                        (
                            second_actions[
                                case_offset, first_index, second_index
                            ],
                            second_responses[
                                case_offset, first_index, second_index
                            ],
                        ),
                    ),
                    previous_belief=first_belief,
                )
            )
        final_raw = generator.chat_complete_messages_batched(
            final_messages,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["final_beliefs"] = final_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        final_beliefs = [
            parse_belief_response(text, available_doc_ids=())[0]
            for text in final_raw
        ]
        final_lookup = {
            key: belief
            for key, belief in zip(
                second_keys, final_beliefs, strict=True
            )
        }

        replay_messages = []
        for case_offset in range(len(cases)):
            first_belief, _actions = first_lookup[case_offset, 0]
            replay_messages.append(
                investigator_messages(
                    cases[case_offset]["agent"],
                    history=(
                        (
                            first_actions[case_offset, 0],
                            first_responses[case_offset, 0],
                        ),
                        (
                            second_actions[case_offset, 0, 0],
                            second_responses[case_offset, 0, 0],
                        ),
                    ),
                    previous_belief=first_belief,
                )
            )
        replay_raw = generator.chat_complete_messages_batched(
            replay_messages,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["replays"] = replay_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        replay_beliefs = [
            parse_belief_response(text, available_doc_ids=())[0]
            for text in replay_raw
        ]

        states_by_case: list[list[tuple[str, Sequence[dict[str, Any]]]]] = []
        for case_offset, (initial_belief, _actions) in enumerate(initial):
            states: list[tuple[str, Sequence[dict[str, Any]]]] = [
                ("initial", initial_belief)
            ]
            for first_index in range(FIRST_ACTION_COUNT):
                first_belief, _next_actions = first_lookup[
                    case_offset, first_index
                ]
                states.append((f"a{first_index}", first_belief))
                for second_index in range(SECOND_ACTION_COUNT):
                    states.append(
                        (
                            f"a{first_index}>b{second_index}",
                            final_lookup[
                                case_offset, first_index, second_index
                            ],
                        )
                    )
            states.append(("replay:a0>b0", replay_beliefs[case_offset]))
            states_by_case.append(states)
        judge_raw = environment.chat_complete_messages_batched(
            [
                equivalence_messages(
                    culprit_reference(case["env"]),
                    states,
                )
                for case, states in zip(
                    cases, states_by_case, strict=True
                )
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["equivalence"] = judge_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        equivalences = [
            parse_equivalence(text, states)
            for text, states in zip(
                judge_raw, states_by_case, strict=True
            )
        ]

        records = []
        frozen_indices = selected_indices(stage)
        for case_offset, case in enumerate(cases):
            initial_belief, _actions = initial[case_offset]
            equivalence = equivalences[case_offset]
            first_branches = []
            for first_index in range(FIRST_ACTION_COUNT):
                first_belief, _next_actions = first_lookup[
                    case_offset, first_index
                ]
                first_branches.append(
                    {
                        "action": first_actions[
                            case_offset, first_index
                        ],
                        "response_sha256": hashlib.sha256(
                            first_responses[
                                case_offset, first_index
                            ].encode("utf-8")
                        ).hexdigest(),
                        "truth_probability": truth_probability(
                            first_belief,
                            equivalence[f"a{first_index}"],
                        ),
                        "second_branches": [
                            {
                                "action": second_actions[
                                    case_offset,
                                    first_index,
                                    second_index,
                                ],
                                "response_sha256": hashlib.sha256(
                                    second_responses[
                                        case_offset,
                                        first_index,
                                        second_index,
                                    ].encode("utf-8")
                                ).hexdigest(),
                                "truth_probability": truth_probability(
                                    final_lookup[
                                        case_offset,
                                        first_index,
                                        second_index,
                                    ],
                                    equivalence[
                                        f"a{first_index}>b{second_index}"
                                    ],
                                ),
                            }
                            for second_index in range(SECOND_ACTION_COUNT)
                        ],
                    }
                )
            records.append(
                {
                    "case_index": frozen_indices[case_offset],
                    "culprit_reference_sha256": hashlib.sha256(
                        culprit_reference(case["env"]).encode("utf-8")
                    ).hexdigest(),
                    "initial_truth_probability": truth_probability(
                        initial_belief, equivalence["initial"]
                    ),
                    "first_branches": first_branches,
                    "replay_truth_probability": truth_probability(
                        replay_beliefs[case_offset],
                        equivalence["replay:a0>b0"],
                    ),
                    "all_belief_sizes_valid": all(
                        len(belief) == BELIEF_SIZE
                        for _state_id, belief in states_by_case[case_offset]
                    ),
                }
            )
        usage = _usage_snapshot(generator, environment)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(generator, environment),
        ) from exc

    summary = summarize(records, usage, stage=stage)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "dataset": "PAPRIKA murder_mystery eval",
            "dataset_sha256": PAPRIKA_SHA256,
            "selection_seed": SELECTION_SEED,
            "case_indices": list(selected_indices(stage)),
            "reserve_indices": list(RESERVE_INDICES),
            "belief_size": BELIEF_SIZE,
            "first_action_count": FIRST_ACTION_COUNT,
            "second_action_count": SECOND_ACTION_COUNT,
            "expected_physical_requests": EXPECTED_REQUESTS[stage],
            "official_environment_prompt": True,
            "hidden_world_visible_only_to_environment": True,
            "truth_used_only_by_post_generation_equivalence_judge": True,
            "reasoning_disabled": True,
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            "external/paprika/llm_exploration/game/game_configs/"
            "murder_mystery.json"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "opportunity"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.75
        config.openrouter_run_budget_usd = 2.00
    else:
        config.openrouter_projected_cost_usd = 2.25
        config.openrouter_run_budget_usd = 6.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "OPPORTUNITY.json"
    )
    try:
        payload = run_gate(
            config,
            data_path=args.data_path,
            stage=args.stage,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        (args.output_dir / f"{args.stage.upper()}_FAILURE.json").write_text(
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
