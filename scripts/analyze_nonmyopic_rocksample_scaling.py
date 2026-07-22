"""Audit exact action-tree scorer scaling across confirmed Rock Diagnosis maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


EXPECTED_RUNS = (
    {
        "run_id": "nonmyopic-rock-branch-strategy-v2-confirmation-20260720",
        "maps": ("3-6", "5-7"),
        "num_trials_per_map": 30,
        "num_rounds": 8,
        "num_strategies": 6,
    },
    {
        "run_id": "nonmyopic-rocksample-7-8-scale-20260721",
        "maps": ("7-8",),
        "num_trials_per_map": 30,
        "num_rounds": 10,
        "num_strategies": 6,
    },
    {
        "run_id": "nonmyopic-rocksample-11-11-gemma-slot-confirmation-20260721",
        "maps": ("11-11",),
        "num_trials_per_map": 30,
        "num_rounds": 12,
        "num_strategies": 6,
    },
    {
        "run_id": "nonmyopic-rocksample-15-15-vllm-replication-20260722",
        "maps": ("15-15",),
        "num_trials_per_map": 30,
        "num_rounds": 15,
        "num_strategies": 4,
    },
)


def analyze(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(payloads) == len(EXPECTED_RUNS)
    rows: list[dict[str, Any]] = []
    run_usage: list[dict[str, Any]] = []
    for payload, expected in zip(payloads, EXPECTED_RUNS):
        assert payload["run_id"] == expected["run_id"]
        assert tuple(payload["config"]["map_names"]) == expected["maps"]
        assert payload["config"]["num_trials_per_map"] == expected["num_trials_per_map"]
        assert payload["config"]["num_rounds"] == expected["num_rounds"]
        assert payload["config"]["num_strategies"] == expected["num_strategies"]
        assert payload["gate_passed"] is True
        assert payload["mechanics"]["rollout_scoring_llm_calls"] == 0
        for map_name in expected["maps"]:
            summaries = payload["maps"][map_name]["summary"]
            strategy_units = float(
                summaries["strategy_eig"]["mean_scorer_units_per_decision"]
            )
            exhaustive_units = float(
                summaries["exhaustive_d2"]["mean_scorer_units_per_decision"]
            )
            num_rocks = int(map_name.split("-")[0])
            rows.append(
                {
                    "map_name": map_name,
                    "num_rocks": num_rocks,
                    "hidden_states": 2**num_rocks,
                    "num_strategies": expected["num_strategies"],
                    "strategy_exact_units_per_decision": strategy_units,
                    "exhaustive_d2_units_per_decision": exhaustive_units,
                    "exhaustive_to_strategy_unit_ratio": exhaustive_units
                    / strategy_units,
                    "strategy_logical_llm_calls_per_decision": float(
                        summaries["strategy_eig"][
                            "mean_logical_llm_calls_per_decision"
                        ]
                    ),
                }
            )
        usage = payload["usage"]
        run_usage.append(
            {
                "run_id": payload["run_id"],
                "maps": list(expected["maps"]),
                "physical_requests": int(usage["requests"]),
                "run_cost_usd": float(usage["run_cost_usd"]),
            }
        )

    assert all(row["strategy_logical_llm_calls_per_decision"] == 1.0 for row in rows)
    assert [row["hidden_states"] for row in rows] == sorted(
        row["hidden_states"] for row in rows
    )
    exhaustive_units = [row["exhaustive_d2_units_per_decision"] for row in rows]
    ratios = [row["exhaustive_to_strategy_unit_ratio"] for row in rows]
    assert exhaustive_units == sorted(exhaustive_units)
    assert ratios == sorted(ratios)
    return {
        "schema_version": 1,
        "claim": "registered_bounded_k_action_tree_work_grows_slower_than_exhaustive_d2_width",
        "unit_definition": (
            "One exact scorer unit is one evaluated action node in the depth-two "
            "policy tree; it is not a wall-clock or hidden-state likelihood operation."
        ),
        "rollout_scoring_llm_calls": 0,
        "rows": rows,
        "run_usage": run_usage,
    }


def render_report(audit: dict[str, Any]) -> str:
    lines = [
        "# Rock Diagnosis Exact-Scorer Scaling",
        "",
        audit["unit_definition"],
        "",
        "| Map | Hidden states | K | StrategyEIG units / decision | Exhaustive d2 units / decision | Exhaustive / StrategyEIG | LLM calls / decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in audit["rows"]:
        lines.append(
            f"| {row['map_name']} | {row['hidden_states']:,} | "
            f"{row['num_strategies']} | "
            f"{row['strategy_exact_units_per_decision']:.2f} | "
            f"{row['exhaustive_d2_units_per_decision']:.2f} | "
            f"{row['exhaustive_to_strategy_unit_ratio']:.2f}x | "
            f"{row['strategy_logical_llm_calls_per_decision']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Registered bounded-K StrategyEIG keeps exact action-tree width small while "
            "exhaustive d2 expands every legal root and continuation. The relative node "
            "reduction grows monotonically from 4.70x to 64.26x across the confirmed maps. "
            "Every StrategyEIG decision uses one proposal call and exact rollout scoring "
            "uses no LLM calls.",
            "",
            "The first four runs use K6; the preregistered 15-rock run uses K4 after K4 "
            "matched K6 endpoint quality in the 11-rock width study. This is observed "
            "registered-budget scaling, not a fixed-K causal comparison.",
            "",
            "These ratios isolate action-tree work. Both methods still evaluate likelihoods "
            "over the full hidden-state vector, so they are not wall-clock speedups.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_scaling(audit: dict[str, Any], output_path: Path) -> None:
    states = [row["hidden_states"] for row in audit["rows"]]
    strategy = [row["strategy_exact_units_per_decision"] for row in audit["rows"]]
    exhaustive = [row["exhaustive_d2_units_per_decision"] for row in audit["rows"]]
    labels = [row["map_name"] for row in audit["rows"]]
    fig, axis = plt.subplots(figsize=(7.0, 4.1))
    axis.plot(
        states,
        strategy,
        color="#c83b36",
        marker="o",
        linewidth=2.2,
        label="StrategyEIG (registered K)",
    )
    axis.plot(states, exhaustive, color="#222222", marker="s", linewidth=2.2, label="Exhaustive d2")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(states, [f"{label}\n{state:,}" for label, state in zip(labels, states)])
    axis.set_xlabel("Map and hidden-state count")
    axis.set_ylabel("Exact action-tree scorer units / decision")
    axis.set_title("Rock Diagnosis verifier scaling")
    axis.grid(True, which="both", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs=4, type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary_output.write_text(render_report(audit), encoding="utf-8")
    plot_scaling(audit, args.plot_output)
    print(json.dumps({"rows": len(audit["rows"])}, indent=2))


if __name__ == "__main__":
    main()
