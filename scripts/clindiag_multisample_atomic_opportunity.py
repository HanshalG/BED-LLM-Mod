#!/usr/bin/env python3
"""Replicated de-anchored ClinDiag support-transition opportunity gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.clindiag_fixed_slot_audit import fixed_evidence_slots
from scripts.clindiag_staged_generator_gate import (
    CLINDIAG_SOURCE_COMMIT,
    CLINDIAG_ZIP_SHA256,
    COVERAGE_THRESHOLD,
    DIAGNOSIS_COUNT,
    ClinDiagCase,
    load_selected_cases,
)
from scripts.icraft_staged_diagnosis_unlock import (
    parse_diagnoses,
    parse_semantic_diagnosis,
    semantic_diagnosis_messages,
)


SCHEMA_VERSION = 1
SELECTION_SEED = 24325
SAMPLES_PER_STATE = 3
ACTIONS = (
    "present_illness",
    "family_social",
    "exam_1",
    "lab_1",
    "imaging_1",
    "other_1",
)
ACTION_LABELS = {
    "present_illness": "Present illness",
    "family_social": "Family and social context",
    "exam_1": "Physical examination slot",
    "lab_1": "Laboratory slot",
    "imaging_1": "Imaging slot",
    "other_1": "Other diagnostic-test slot",
}
SEQUENCES = tuple(permutations(ACTIONS, 2))
SMOKE_IDS = ("26933852", "rare129")
FORMAL_IDS = ("17700107_1", "20921516", "rare269", "rare295")
RESERVE_IDS = ("23252529", "27783909", "rare257", "rare153")
SMOKE_PATHS = (
    ("present_illness", "lab_1"),
    ("family_social", "imaging_1"),
)
REPLAY_PATHS = SEQUENCES[: len(FORMAL_IDS)]
EXPECTED_SMOKE_REQUESTS = len(SMOKE_IDS) * (
    SAMPLES_PER_STATE * 4 + 1
)
EXPECTED_FORMAL_REQUESTS = len(FORMAL_IDS) * (
    SAMPLES_PER_STATE * (1 + len(ACTIONS) + len(SEQUENCES) + 1)
    + 1
    + len(ACTIONS)
    + len(SEQUENCES)
    + 1
)


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _build_models(config: Config) -> tuple[Any, Any]:
    generator_spec = config.model_pairs[0].questioner
    judge_spec = replace(
        config.model_pairs[0].answerer,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    return (
        build_model_adapter(generator_spec, config),
        build_model_adapter(judge_spec, config),
    )


def _usage_snapshot(generator: Any, judge: Any) -> dict[str, Any]:
    generator_usage = generator.usage_snapshot()
    judge_usage = judge.usage_snapshot()
    return {
        "physical_requests": int(generator_usage["adapter_requests"])
        + int(judge_usage["adapter_requests"]),
        "reasoning_tokens": int(generator_usage["adapter_reasoning_tokens"])
        + int(judge_usage["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(generator_usage["adapter_cost_usd"])
        + float(judge_usage["adapter_cost_usd"]),
        "generator": generator_usage,
        "judge": judge_usage,
    }


def state_id(actions: Sequence[str]) -> str:
    return "initial" if not actions else ">".join(actions)


def refresh_messages(
    case: ClinDiagCase,
    observed_actions: Sequence[str],
) -> list[dict[str, str]]:
    slots = fixed_evidence_slots(case)
    if observed_actions:
        evidence = "\n\n".join(
            f"Step {index} - {ACTION_LABELS[action]}:\n"
            + json.dumps(slots[action], ensure_ascii=True, sort_keys=True)
            for index, action in enumerate(observed_actions, start=1)
        )
    else:
        evidence = "No additional evidence has been acquired."
    return [
        {
            "role": "system",
            "content": (
                "Construct an independent open-world clinical differential from "
                "only the evidence shown. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial presentation:\n{case.initial_information}\n\n"
                f"Evidence acquired in order:\n{evidence}\n\n"
                f"Return exactly {DIAGNOSIS_COUNT} distinct, specific, unifying "
                "diagnoses as "
                '{"diagnoses":["diagnosis name", ...]}. Each item must be one '
                "diagnosis-name string, not an explanation or object. Build the "
                "list afresh from all visible evidence; no earlier differential "
                "is supplied or must be retained. Include rare conditions when "
                "plausible. No answer options, case title, final diagnosis, or "
                "hidden benchmark label is provided."
            ),
        },
    ]


def _generate_raw(
    model: Any,
    messages: Sequence[list[dict[str, str]]],
    config: Config,
    *,
    temperature: float,
) -> tuple[list[Any], list[str]]:
    raw = model.chat_complete_messages_batched(
        list(messages),
        temperature=temperature,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    return [parse_diagnoses(text, DIAGNOSIS_COUNT) for text in raw], raw


def _semantic_raw(
    model: Any,
    requests: Sequence[tuple[str, Sequence[tuple[str, Sequence[str]]]]],
    config: Config,
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    messages = [
        semantic_diagnosis_messages(true_diagnosis, supports)
        for true_diagnosis, supports in requests
    ]
    raw = model.chat_complete_messages_batched(
        messages,
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    parsed = [
        parse_semantic_diagnosis(
            text,
            [support_id for support_id, _support in supports],
        )
        for text, (_diagnosis, supports) in zip(raw, requests, strict=True)
    ]
    return parsed, raw


def _checkpoint(
    path: Path | None,
    *,
    stage: str,
    raw: dict[str, Any],
) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "stage": stage, **raw},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def aggregate_measurements(
    measurements: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    scores = [float(row["best_match_score"]) for row in measurements]
    return {
        "sample_scores": scores,
        "soft_value": float(np.mean(scores)),
        "coverage_probability": float(
            np.mean([score >= COVERAGE_THRESHOLD for score in scores])
        ),
    }


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    one = {
        action: float(value["soft_value"])
        for action, value in record["one_step"].items()
    }
    pairs = {
        key: float(value["soft_value"])
        for key, value in record["sequences"].items()
    }
    greedy = max(ACTIONS, key=lambda action: (one[action], -ACTIONS.index(action)))
    pair_order = [state_id(sequence) for sequence in SEQUENCES]
    oracle_pair = max(
        pair_order,
        key=lambda key: (pairs[key], -pair_order.index(key)),
    )
    greedy_continuation = max(
        value for key, value in pairs.items() if key.startswith(f"{greedy}>")
    )
    best_one = max(one.values())
    replay_key = record["replay_sequence"]
    replay_gap = abs(
        float(record["sequences"][replay_key]["soft_value"])
        - float(record["replay"]["soft_value"])
    )
    one_coverage = [
        float(value["coverage_probability"])
        for value in record["one_step"].values()
    ]
    pair_coverage = [
        float(value["coverage_probability"])
        for value in record["sequences"].values()
    ]
    return {
        "greedy_action": greedy,
        "oracle_sequence": oracle_pair,
        "oracle_first_differs_from_greedy": (
            oracle_pair.split(">", 1)[0] != greedy
        ),
        "one_step_soft_spread": max(one.values()) - min(one.values()),
        "one_step_coverage_spread": max(one_coverage) - min(one_coverage),
        "best_one_step_soft_value": best_one,
        "greedy_continuation_soft_value": greedy_continuation,
        "oracle_pair_soft_value": pairs[oracle_pair],
        "nonmyopic_soft_gap": pairs[oracle_pair] - greedy_continuation,
        "pair_gain_over_best_one_step": pairs[oracle_pair] - best_one,
        "best_one_step_coverage": max(one_coverage),
        "best_pair_coverage": max(pair_coverage),
        "best_pair_coverage_gain": max(pair_coverage) - max(one_coverage),
        "replay_soft_gap": replay_gap,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = [analyze_record(record) for record in records]
    initial_coverage = float(
        np.mean(
            [
                record["initial"]["coverage_probability"]
                for record in records
            ]
        )
    )
    spread = sum(
        row["one_step_coverage_spread"] >= (1.0 / SAMPLES_PER_STATE)
        for row in diagnostics
    )
    pair_gain_cases = sum(
        row["best_pair_coverage_gain"] >= (1.0 / SAMPLES_PER_STATE)
        for row in diagnostics
    )
    gap_cases = sum(row["nonmyopic_soft_gap"] >= 0.10 for row in diagnostics)
    mean_gap = float(np.mean([row["nonmyopic_soft_gap"] for row in diagnostics]))
    mean_pair_gain = float(
        np.mean([row["pair_gain_over_best_one_step"] for row in diagnostics])
    )
    replay_gaps = [row["replay_soft_gap"] for row in diagnostics]
    first_differs = sum(
        row["oracle_first_differs_from_greedy"] for row in diagnostics
    )
    summary = {
        "num_cases": len(records),
        "initial_mean_coverage_probability": initial_coverage,
        "one_step_coverage_spread_cases": spread,
        "pair_coverage_gain_cases": pair_gain_cases,
        "nonmyopic_soft_gap_at_least_0_10_cases": gap_cases,
        "oracle_first_differs_from_greedy_cases": first_differs,
        "mean_nonmyopic_soft_gap": mean_gap,
        "mean_pair_gain_over_best_one_step": mean_pair_gain,
        "mean_replay_soft_gap": float(np.mean(replay_gaps)),
        "max_replay_soft_gap": max(replay_gaps),
        "case_diagnostics": [
            {"source_id": record["source_id"], **diagnostic}
            for record, diagnostic in zip(records, diagnostics, strict=True)
        ],
    }
    gates = {
        "all_four_cases_complete": len(records) == len(FORMAL_IDS),
        "exactly_608_physical_requests": int(usage["physical_requests"])
        == EXPECTED_FORMAL_REQUESTS,
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "initial_support_not_saturated": initial_coverage <= 0.50,
        "one_step_coverage_spread_on_at_least_two": spread >= 2,
        "pair_coverage_gain_on_at_least_two": pair_gain_cases >= 2,
        "nonmyopic_gap_on_at_least_one": gap_cases >= 1,
        "mean_nonmyopic_gap_at_least_0_03": mean_gap >= 0.03,
        "mean_pair_gain_at_least_0_05": mean_pair_gain >= 0.05,
        "oracle_first_differs_on_at_least_one": first_differs >= 1,
        "mean_replay_gap_at_most_0_10": float(np.mean(replay_gaps)) <= 0.10,
        "max_replay_gap_at_most_0_20": max(replay_gaps) <= 0.20,
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def _formal_states() -> tuple[tuple[str, ...], ...]:
    return (
        (),
        *((action,) for action in ACTIONS),
        *SEQUENCES,
    )


def _run_stage(
    config: Config,
    *,
    data_zip: Path,
    stage: str,
    raw_checkpoint_path: Path | None,
) -> dict[str, Any]:
    generator, judge = _build_models(config)
    raw: dict[str, Any] = {}
    try:
        if stage == "serving_smoke":
            cases = load_selected_cases(data_zip, SMOKE_IDS)
            state_schedules = [
                (
                    (),
                    (path[0],),
                    path,
                    path,
                )
                for path in SMOKE_PATHS
            ]
        elif stage == "opportunity":
            cases = load_selected_cases(data_zip, FORMAL_IDS)
            base_states = _formal_states()
            state_schedules = [
                (*base_states, REPLAY_PATHS[index])
                for index in range(len(cases))
            ]
        else:
            raise ValueError("stage must be serving_smoke or opportunity")

        support_keys: list[tuple[int, int, int]] = []
        support_messages = []
        for case_index, (case, states) in enumerate(
            zip(cases, state_schedules, strict=True)
        ):
            for state_index, actions in enumerate(states):
                for sample_index in range(SAMPLES_PER_STATE):
                    support_keys.append(
                        (case_index, state_index, sample_index)
                    )
                    support_messages.append(refresh_messages(case, actions))
        parsed_supports, support_raw = _generate_raw(
            generator,
            support_messages,
            config,
            temperature=float(config.generation_temperature_diverse),
        )
        raw["support_generation"] = support_raw
        _checkpoint(
            raw_checkpoint_path,
            stage=stage,
            raw=raw,
        )
        supports = {
            key: value
            for key, value in zip(
                support_keys, parsed_supports, strict=True
            )
        }

        semantic_requests = []
        semantic_keys: list[tuple[int, int]] = []
        if stage == "serving_smoke":
            for case_index, (case, states) in enumerate(
                zip(cases, state_schedules, strict=True)
            ):
                rows = []
                for state_index, _actions in enumerate(states):
                    for sample_index in range(SAMPLES_PER_STATE):
                        rows.append(
                            (
                                f"state{state_index}:sample{sample_index}",
                                supports[
                                    case_index, state_index, sample_index
                                ],
                            )
                        )
                semantic_keys.append((case_index, -1))
                semantic_requests.append((case.final_diagnosis, rows))
        else:
            for case_index, (case, states) in enumerate(
                zip(cases, state_schedules, strict=True)
            ):
                for state_index, _actions in enumerate(states):
                    rows = [
                        (
                            f"sample{sample_index}",
                            supports[case_index, state_index, sample_index],
                        )
                        for sample_index in range(SAMPLES_PER_STATE)
                    ]
                    semantic_keys.append((case_index, state_index))
                    semantic_requests.append((case.final_diagnosis, rows))
        semantic, semantic_raw = _semantic_raw(
            judge, semantic_requests, config
        )
        raw["semantic_measurement"] = semantic_raw
        _checkpoint(
            raw_checkpoint_path,
            stage=stage,
            raw=raw,
        )

        measurements: dict[tuple[int, int], list[dict[str, Any]]] = {}
        if stage == "serving_smoke":
            for (case_index, _), rows in zip(
                semantic_keys, semantic, strict=True
            ):
                for state_index in range(4):
                    start = state_index * SAMPLES_PER_STATE
                    measurements[case_index, state_index] = rows[
                        start : start + SAMPLES_PER_STATE
                    ]
        else:
            measurements = {
                key: rows
                for key, rows in zip(semantic_keys, semantic, strict=True)
            }

        records = []
        for case_index, (case, states) in enumerate(
            zip(cases, state_schedules, strict=True)
        ):
            aggregates = [
                aggregate_measurements(
                    measurements[case_index, state_index]
                )
                for state_index in range(len(states))
            ]
            if stage == "serving_smoke":
                records.append(
                    {
                        "source_id": case.source_id,
                        "subset": case.subset,
                        "path": state_id(states[2]),
                        "initial": aggregates[0],
                        "one_step": aggregates[1],
                        "two_step": aggregates[2],
                        "replay": aggregates[3],
                        "support_sizes": [
                            len(
                                supports[
                                    case_index, state_index, sample_index
                                ]
                            )
                            for state_index in range(len(states))
                            for sample_index in range(SAMPLES_PER_STATE)
                        ],
                    }
                )
                continue
            state_count = len(_formal_states())
            initial = aggregates[0]
            one_step = {
                action: aggregates[1 + index]
                for index, action in enumerate(ACTIONS)
            }
            sequences = {
                state_id(sequence): aggregates[1 + len(ACTIONS) + index]
                for index, sequence in enumerate(SEQUENCES)
            }
            replay_sequence = state_id(REPLAY_PATHS[case_index])
            records.append(
                {
                    "source_id": case.source_id,
                    "subset": case.subset,
                    "initial": initial,
                    "one_step": one_step,
                    "sequences": sequences,
                    "replay_sequence": replay_sequence,
                    "replay": aggregates[state_count],
                }
            )
        usage = _usage_snapshot(generator, judge)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(generator, judge),
        ) from exc

    if stage == "serving_smoke":
        all_sizes = [
            size for record in records for size in record["support_sizes"]
        ]
        replay_gaps = [
            abs(record["two_step"]["soft_value"] - record["replay"]["soft_value"])
            for record in records
        ]
        gates = {
            "both_cases_complete": len(records) == len(SMOKE_IDS),
            "all_supports_size_12": all(
                size == DIAGNOSIS_COUNT for size in all_sizes
            ),
            "exactly_26_physical_requests": int(usage["physical_requests"])
            == EXPECTED_SMOKE_REQUESTS,
            "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
            "replay_soft_gaps_finite": all(
                np.isfinite(gap) for gap in replay_gaps
            ),
        }
        gates["all_pass"] = all(gates.values())
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if gates["all_pass"] else "gate_failed",
            "protocol": _protocol(stage),
            "gates": gates,
            "records": records,
            "usage": usage,
        }
    summary = summarize(records, usage)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed"
            if summary["gates"]["all_pass"]
            else "opportunity_gate_failed"
        ),
        "protocol": _protocol(stage),
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def _protocol(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "dataset": "ClinDiag-Benchmark",
        "source_commit": CLINDIAG_SOURCE_COMMIT,
        "archive_sha256": CLINDIAG_ZIP_SHA256,
        "selection_seed": SELECTION_SEED,
        "smoke_ids": list(SMOKE_IDS),
        "formal_ids": list(FORMAL_IDS),
        "reserve_ids": list(RESERVE_IDS),
        "actions": list(ACTIONS),
        "ordered_sequences": [state_id(sequence) for sequence in SEQUENCES],
        "samples_per_state": SAMPLES_PER_STATE,
        "diagnosis_count": DIAGNOSIS_COUNT,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "deanchored_full_refresh": True,
        "stored_observations_not_generated": True,
        "truth_used_for_post_generation_measurement_only": True,
        "reasoning_disabled": True,
        "expected_physical_requests": (
            EXPECTED_SMOKE_REQUESTS
            if stage == "serving_smoke"
            else EXPECTED_FORMAL_REQUESTS
        ),
        "raw_responses_private_and_untracked": True,
    }


def run_gate(
    config: Config,
    *,
    data_zip: Path,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    return _run_stage(
        config,
        data_zip=data_zip,
        stage=stage,
        raw_checkpoint_path=raw_checkpoint_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-zip",
        type=Path,
        default=Path("external/ClinDiag/Clindiag_Benchmark.zip"),
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
        config.openrouter_projected_cost_usd = 0.25
        config.openrouter_run_budget_usd = 0.75
    else:
        config.openrouter_projected_cost_usd = 3.00
        config.openrouter_run_budget_usd = 8.00
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
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "OPPORTUNITY_FAILURE.json"
    )
    try:
        payload = run_gate(
            config,
            data_zip=args.data_zip,
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
            "raw_responses_path": str(raw_path),
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
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
            {
                "status": payload["status"],
                **payload.get("gates", {}),
                **payload.get("summary", {}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
