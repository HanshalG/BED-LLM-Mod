"""Audit indexed Gated Sensor follow-up choices on exact branch beliefs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any
from collections import Counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.gated_sensor import GatedSensorModel
from environments.gated_sensor.model import EPSILON
from scripts.nonmyopic_gated_sensor_strategy_prior_v2 import _normalize_response


def _history_key(history: list[dict[str, Any]]) -> tuple[tuple[str, str | None], ...]:
    return tuple((str(item["action"]), item["observation"]) for item in history)


def _strategy_reached_keys(payload: dict[str, Any]) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for trace in payload["traces"]["strategy_eig"]:
        history: list[dict[str, Any]] = []
        for round_index, step in enumerate(trace["steps"]):
            if round_index < len(trace["steps"]) - 1:
                keys.add(
                    (
                        int(trace["trial_index"]),
                        step["state_before"],
                        _history_key(history),
                        2,
                    )
                )
            history.append({"action": step["action"], "observation": step["observation"]})
    return keys


def _request_key(request: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(request["trial_index"]),
        request["active_panel"],
        _history_key(request["history"]),
        int(request["horizon"]),
    )


def _belief_at_request(
    model: GatedSensorModel, request: dict[str, Any]
) -> tuple[Any, Any]:
    belief = model.initial_belief.copy()
    state = model.initial_state
    for item in request["history"]:
        action = str(item["action"])
        observation = item["observation"]
        belief = model.posterior(belief, action, observation)
        state = model.next_state(state, action)
    assert state.active_panel == request["active_panel"]
    return belief, state


def _choice_rows(
    model: GatedSensorModel,
    request: dict[str, Any],
    *,
    request_index: int,
    strategy_reached: bool,
) -> list[dict[str, Any]]:
    belief, state = _belief_at_request(model, request)
    response = json.loads(_normalize_response(str(request["raw_response"])))
    raw_rows = response["choices"]
    roots = tuple(request["roots"])
    menus = request["menus"]
    assert len(raw_rows) == len(roots) == len(menus)
    rows: list[dict[str, Any]] = []
    for slot, (root, raw_row, stored_menus) in enumerate(
        zip(roots, raw_rows, menus, strict=True)
    ):
        child_state = model.next_state(state, root)
        for branch_index, outcome in enumerate(model.outcomes(root)):
            branch = "none" if outcome is None else outcome
            choices = list(stored_menus[branch])
            selected_index = int(raw_row[branch_index])
            selected_action = choices[selected_index]
            assert selected_action in model.legal_actions(child_state)
            posterior = model.posterior(belief, root, outcome)
            eigs = [model.expected_information_gain(posterior, action) for action in choices]
            selected_eig = eigs[selected_index]
            optimal_eig = max(eigs)
            rank = 1 + sum(value > selected_eig + 1e-12 for value in eigs)
            root_kind = model.action_kind(root)
            selected_kind = model.action_kind(selected_action)
            repeated_predicate = (
                root_kind != "activate"
                and selected_kind != "activate"
                and model.action_target(root) == model.action_target(selected_action)
            )
            rows.append(
                {
                    "request_index": request_index,
                    "trial_index": int(request["trial_index"]),
                    "history_length": len(request["history"]),
                    "active_panel": request["active_panel"],
                    "strategy_reached": strategy_reached,
                    "slot": slot,
                    "root_action": root,
                    "root_kind": root_kind,
                    "branch": branch,
                    "selected_index": selected_index,
                    "menu_size": len(choices),
                    "selected_action": selected_action,
                    "selected_kind": selected_kind,
                    "selected_eig": selected_eig,
                    "optimal_eig": optimal_eig,
                    "eig_efficiency": (
                        selected_eig / optimal_eig if optimal_eig > EPSILON else 1.0
                    ),
                    "eig_rank": rank,
                    "normalized_rank": (
                        (rank - 1) / (len(choices) - 1) if len(choices) > 1 else 0.0
                    ),
                    "is_optimal": math.isclose(
                        selected_eig, optimal_eig, abs_tol=1e-12, rel_tol=0.0
                    ),
                    "zero_eig": selected_eig <= EPSILON,
                    "repeats_root_predicate": repeated_predicate,
                    "selects_activation": selected_kind == "activate",
                    "selects_index_zero": selected_index == 0,
                }
            )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("choice audit requires at least one branch choice")
    measurement_rows = [row for row in rows if row["root_kind"] != "activate"]
    activation_rows = [row for row in rows if row["root_kind"] == "activate"]
    return {
        "num_requests": len(
            {row["request_index"] for row in rows}
        ),
        "num_branch_choices": len(rows),
        "mean_immediate_eig_efficiency": statistics.fmean(
            row["eig_efficiency"] for row in rows
        ),
        "mean_normalized_eig_rank": statistics.fmean(
            row["normalized_rank"] for row in rows
        ),
        "optimal_choice_rate": statistics.fmean(float(row["is_optimal"]) for row in rows),
        "zero_eig_choice_rate": statistics.fmean(float(row["zero_eig"]) for row in rows),
        "index_zero_rate": statistics.fmean(float(row["selects_index_zero"]) for row in rows),
        "measurement_branch_count": len(measurement_rows),
        "activation_root_mean_eig_efficiency": statistics.fmean(
            row["eig_efficiency"] for row in activation_rows
        ),
        "measurement_root_mean_eig_efficiency": statistics.fmean(
            row["eig_efficiency"] for row in measurement_rows
        ),
        "activation_root_optimal_choice_rate": statistics.fmean(
            float(row["is_optimal"]) for row in activation_rows
        ),
        "measurement_root_optimal_choice_rate": statistics.fmean(
            float(row["is_optimal"]) for row in measurement_rows
        ),
        "measurement_repeat_root_rate": statistics.fmean(
            float(row["repeats_root_predicate"]) for row in measurement_rows
        ),
        "measurement_activation_rate": statistics.fmean(
            float(row["selects_activation"]) for row in measurement_rows
        ),
        "measurement_zero_eig_rate": statistics.fmean(
            float(row["zero_eig"]) for row in measurement_rows
        ),
        "measurement_index_zero_rate": statistics.fmean(
            float(row["selects_index_zero"]) for row in measurement_rows
        ),
        "measurement_selected_kind_counts": dict(
            sorted(Counter(row["selected_kind"] for row in measurement_rows).items())
        ),
        "measurement_selected_index_counts": {
            str(index): count
            for index, count in sorted(
                Counter(row["selected_index"] for row in measurement_rows).items()
            )
        },
    }


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    assert payload["schema_version"] == 2
    assert payload["mechanics"]["rollout_scoring_made_no_llm_calls"] is True
    config = payload["config"]
    model = GatedSensorModel(
        screen_accuracy=float(config["screen_accuracy"]),
        precise_accuracy=float(config["precise_accuracy"]),
    )
    reached_keys = _strategy_reached_keys(payload)
    all_rows: list[dict[str, Any]] = []
    matched_keys: set[tuple[Any, ...]] = set()
    for request_index, request in enumerate(payload["candidate_requests"]):
        key = _request_key(request)
        strategy_reached = key in reached_keys
        if strategy_reached:
            matched_keys.add(key)
        all_rows.extend(
            _choice_rows(
                model,
                request,
                request_index=request_index,
                strategy_reached=strategy_reached,
            )
        )
    if matched_keys != reached_keys:
        missing = reached_keys - matched_keys
        raise AssertionError(f"missing {len(missing)} StrategyEIG-reached request cells")
    strategy_rows = [row for row in all_rows if row["strategy_reached"]]
    return {
        "schema_version": 1,
        "stage": "gated_sensor_indexed_choice_mechanics",
        "source_run_id": payload["run_id"],
        "interface_version": config["interface_version"],
        "no_llm_calls": True,
        "strategy_reached": _summarize(strategy_rows),
        "all_accepted_requests": _summarize(all_rows),
        "rows": all_rows,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Gated Sensor Indexed-Choice Mechanics",
        "",
        "This zero-call audit reconstructs exact branch beliefs and scores the integer follow-up selected in every indexed menu.",
        "",
        "| Scope | Requests | Branches | EIG efficiency | Optimal | Zero EIG | Index 0 | Mean rank | Repeat root | Activation after measurement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("strategy_reached", "StrategyEIG-reached states"),
        ("all_accepted_requests", "All accepted cells"),
    ):
        row = summary[key]
        lines.append(
            f"| {label} | {row['num_requests']} | {row['num_branch_choices']} | "
            f"{row['mean_immediate_eig_efficiency']:.4f} | {row['optimal_choice_rate']:.4f} | "
            f"{row['zero_eig_choice_rate']:.4f} | {row['index_zero_rate']:.4f} | "
            f"{row['mean_normalized_eig_rank']:.4f} | "
            f"{row['measurement_repeat_root_rate']:.4f} | "
            f"{row['measurement_activation_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Scope | Activation-root efficiency | Measurement-root efficiency | Activation-root optimal | Measurement-root optimal | Measurement zero EIG |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in (
        ("strategy_reached", "StrategyEIG-reached states"),
        ("all_accepted_requests", "All accepted cells"),
    ):
        row = summary[key]
        lines.append(
            f"| {label} | {row['activation_root_mean_eig_efficiency']:.4f} | "
            f"{row['measurement_root_mean_eig_efficiency']:.4f} | "
            f"{row['activation_root_optimal_choice_rate']:.4f} | "
            f"{row['measurement_root_optimal_choice_rate']:.4f} | "
            f"{row['measurement_zero_eig_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Mean normalized rank is zero for a best-EIG menu choice and one for a worst-ranked choice. Repeat-root and activation rates are restricted to branches following a measurement root.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(json.loads(args.result.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary_output.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary["strategy_reached"], indent=2))


if __name__ == "__main__":
    main()
