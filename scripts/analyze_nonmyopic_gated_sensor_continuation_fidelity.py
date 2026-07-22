"""Decompose Gated Sensor proposal quality into root and continuation fidelity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.gated_sensor import GatedSensorModel, SensorState
from environments.gated_sensor.model import EPSILON
from scripts.nonmyopic_gated_sensor_strategy_prior import PolicyState
from scripts.nonmyopic_gated_sensor_oracle import exact_action_values
from scripts.nonmyopic_gated_sensor_strategy_prior_v2 import (
    IndexedStrategyConfig,
    _random_selection_v2,
)


ARMS = ("strategy_eig", "shared_state_random", "random_strategy")


def _fidelity_row(
    *,
    model: GatedSensorModel,
    state: SensorState,
    belief: Any,
    arm: str,
    trial_index: int,
    round_index: int,
    roots: tuple[str, ...],
    proposed_value: float,
) -> dict[str, Any]:
    values, _units = exact_action_values(
        model,
        state=state,
        belief=belief,
        depth=2,
    )
    assert roots and all(root in values for root in roots)
    closed_value = max(values[root] for root in roots)
    exhaustive_value = max(values.values())
    assert proposed_value <= closed_value + 1e-12
    assert closed_value <= exhaustive_value + 1e-12
    return {
        "arm": arm,
        "trial_index": trial_index,
        "round": round_index + 1,
        "active_panel": state.active_panel,
        "candidate_roots": list(roots),
        "proposed_value": proposed_value,
        "same_root_closed_value": closed_value,
        "exhaustive_d2_value": exhaustive_value,
        "continuation_efficiency": (
            proposed_value / closed_value if closed_value > EPSILON else 1.0
        ),
        "root_coverage": (
            closed_value / exhaustive_value if exhaustive_value > EPSILON else 1.0
        ),
        "proposal_exhaustive_fraction": (
            proposed_value / exhaustive_value if exhaustive_value > EPSILON else 1.0
        ),
        "continuation_regret": closed_value - proposed_value,
        "contains_optimal_root": math.isclose(
            closed_value,
            exhaustive_value,
            abs_tol=1e-12,
            rel_tol=0.0,
        ),
        "contains_optimal_continuation": math.isclose(
            proposed_value,
            closed_value,
            abs_tol=1e-12,
            rel_tol=0.0,
        ),
    }


def _rows_for_arm(
    payload: dict[str, Any], arm: str
) -> list[dict[str, Any]]:
    config = payload["config"]
    model = GatedSensorModel(
        screen_accuracy=float(config["screen_accuracy"]),
        precise_accuracy=float(config["precise_accuracy"]),
    )
    rows: list[dict[str, Any]] = []
    for trace in payload["traces"][arm]:
        belief = model.initial_belief.copy()
        state = model.initial_state
        for round_index, step in enumerate(trace["steps"]):
            assert step["state_before"] == state.active_panel
            if round_index < int(config["num_rounds"]) - 1:
                roots = tuple(dict.fromkeys(step["candidate_roots"]))
                rows.append(
                    _fidelity_row(
                        model=model,
                        state=state,
                        belief=belief,
                        arm=arm,
                        trial_index=int(trace["trial_index"]),
                        round_index=round_index,
                        roots=roots,
                        proposed_value=float(step["planning_score"]),
                    )
                )
            belief = model.posterior(belief, step["action"], step["observation"])
            state = model.next_state(state, step["action"])
    return rows


def _shared_state_random_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_config = payload["config"]
    config = IndexedStrategyConfig(**raw_config)
    model = GatedSensorModel(
        screen_accuracy=config.screen_accuracy,
        precise_accuracy=config.precise_accuracy,
    )
    rows: list[dict[str, Any]] = []
    for trace in payload["traces"]["strategy_eig"]:
        belief = model.initial_belief.copy()
        state = model.initial_state
        history: tuple[tuple[str, str | None], ...] = ()
        for round_index, step in enumerate(trace["steps"]):
            if round_index < config.num_rounds - 1:
                selection = _random_selection_v2(
                    model,
                    PolicyState(belief=belief, sensor_state=state, history=history),
                    config,
                    trial_index=int(trace["trial_index"]),
                    round_index=round_index,
                    horizon=2,
                )
                rows.append(
                    _fidelity_row(
                        model=model,
                        state=state,
                        belief=belief,
                        arm="shared_state_random",
                        trial_index=int(trace["trial_index"]),
                        round_index=round_index,
                        roots=tuple(selection.candidate_roots),
                        proposed_value=float(selection.planning_score),
                    )
                )
            observation = step["observation"]
            belief = model.posterior(belief, step["action"], observation)
            state = model.next_state(state, step["action"])
            history = history + ((step["action"], observation),)
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_h2_states": len(rows),
        "mean_continuation_efficiency": statistics.fmean(
            row["continuation_efficiency"] for row in rows
        ),
        "mean_root_coverage": statistics.fmean(row["root_coverage"] for row in rows),
        "mean_proposal_exhaustive_fraction": statistics.fmean(
            row["proposal_exhaustive_fraction"] for row in rows
        ),
        "mean_continuation_regret": statistics.fmean(
            row["continuation_regret"] for row in rows
        ),
        "optimal_root_coverage": statistics.fmean(
            float(row["contains_optimal_root"]) for row in rows
        ),
        "optimal_continuation_rate": statistics.fmean(
            float(row["contains_optimal_continuation"]) for row in rows
        ),
    }


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    assert payload["schema_version"] == 2
    assert payload["mechanics"]["rollout_scoring_made_no_llm_calls"] is True
    rows = {
        "strategy_eig": _rows_for_arm(payload, "strategy_eig"),
        "shared_state_random": _shared_state_random_rows(payload),
        "random_strategy": _rows_for_arm(payload, "random_strategy"),
    }
    summaries = {arm: _summarize(rows[arm]) for arm in ARMS}
    expected_states = int(payload["config"]["num_trials"]) * (
        int(payload["config"]["num_rounds"]) - 1
    )
    assert all(summary["num_h2_states"] == expected_states for summary in summaries.values())
    return {
        "schema_version": 1,
        "stage": "gated_sensor_continuation_fidelity",
        "source_run_id": payload["run_id"],
        "no_llm_calls": True,
        "arms": summaries,
        "llm_minus_random_continuation_efficiency": (
            summaries["strategy_eig"]["mean_continuation_efficiency"]
            - summaries["shared_state_random"]["mean_continuation_efficiency"]
        ),
        "llm_minus_random_proposal_exhaustive_fraction": (
            summaries["strategy_eig"]["mean_proposal_exhaustive_fraction"]
            - summaries["shared_state_random"]["mean_proposal_exhaustive_fraction"]
        ),
        "llm_minus_reached_random_continuation_efficiency": (
            summaries["strategy_eig"]["mean_continuation_efficiency"]
            - summaries["random_strategy"]["mean_continuation_efficiency"]
        ),
        "rows": rows,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Gated Sensor Continuation-Fidelity Diagnostic",
        "",
        "This zero-LLM-call audit separates fixed-root coverage from branch-continuation quality.",
        "",
        "| Arm | h2 states | Continuation efficiency | Root coverage | Proposal / exhaustive d2 | Optimal continuation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, label in (
        ("strategy_eig", "LLM StrategyEIG"),
        ("shared_state_random", "Matched random on LLM states"),
        ("random_strategy", "Reached random arm"),
    ):
        row = summary["arms"][arm]
        lines.append(
            f"| {label} | {row['num_h2_states']} | "
            f"{row['mean_continuation_efficiency']:.4f} | "
            f"{row['mean_root_coverage']:.4f} | "
            f"{row['mean_proposal_exhaustive_fraction']:.4f} | "
            f"{row['optimal_continuation_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "LLM-minus-random continuation efficiency: "
            f"`{summary['llm_minus_random_continuation_efficiency']:+.4f}`.",
            "LLM-minus-random proposal/exhaustive fraction: "
            f"`{summary['llm_minus_random_proposal_exhaustive_fraction']:+.4f}`.",
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
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary_output.write_text(render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "llm_minus_random_continuation_efficiency": summary[
                    "llm_minus_random_continuation_efficiency"
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
