"""Audit and summarize the preregistered RockSample[7,8] scale extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_nonmyopic_rock_branch_result import (
    ARMS,
    ARM_LABELS,
    ARM_STYLES,
    BASELINES,
)


EXPECTED_MAP = "7-8"


def _audit_qualification(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    assert config == {
        "map_name": EXPECTED_MAP,
        "num_trials": 1000,
        "num_rounds": 10,
        "candidate_widths": [12],
        "seed": 24071,
        "trial_offset": 0,
        "bootstrap_replicates": 10_000,
        "half_efficiency_distance": 0.6931471805599453,
    }
    assert payload["no_llm_calls"] is True
    source = payload["source"]
    assert source["map"] == EXPECTED_MAP
    assert source["page"] == 5
    assert source["map_spec"]["rock_positions"] == [
        [1, 0],
        [5, 1],
        [2, 2],
        [3, 2],
        [6, 3],
        [0, 5],
        [3, 5],
        [2, 6],
    ]
    width = payload["widths"]["12"]
    assert width["passed_confirmation_gate"] is True
    assert payload["decision"]["confirmation_passes"] is True
    assert all(width["mechanics"].values())
    comparisons = width["comparisons"]
    for label in ("d2_minus_shared_d1", "d2_minus_call_matched_width"):
        comparison = comparisons[label]
        assert comparison["final_entropy_reduction_ci95"][0] > 0.0
    return {
        "passed": True,
        "source": source,
        "comparisons": comparisons,
        "mechanics": width["mechanics"],
    }


def _audit_confirmation(payload: dict[str, Any]) -> dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["stage"] == "L1"
    assert payload["dry_run"] is False
    assert payload["run_id"] == "nonmyopic-rocksample-7-8-scale-20260721"
    config = payload["config"]
    assert config == {
        "map_names": [EXPECTED_MAP],
        "num_trials_per_map": 30,
        "num_rounds": 10,
        "num_strategies": 6,
        "planning_horizon": 2,
        "seed": 24072,
        "bootstrap_replicates": 10_000,
        "temperature": 0.0,
        "validation_retries": 1,
        "trial_concurrency": 32,
        "strategy_schema": "branch_policy_v2",
        "primary_endpoint": "entropy_auc",
    }

    mechanics = payload["mechanics"]
    required_mechanics = (
        "all_selected_actions_legal",
        "initial_strategy_cells_shared_with_d1",
        "width_logical_llm_calls_match_strategy_eig",
        "width_exact_scorer_units_match_strategy_eig",
        "random_strategy_cells_have_k_candidates",
    )
    assert mechanics["terminal_cell_failures"] == 0
    assert mechanics["rollout_scoring_llm_calls"] == 0
    assert all(mechanics[key] for key in required_mechanics)
    assert mechanics["accepted_llm_cells"] == len(payload["candidate_requests"])
    assert mechanics["raw_rejected_responses"] == len(payload["invalid_responses"])
    assert mechanics["physical_llm_requests"] == (
        len(payload["candidate_requests"]) + len(payload["invalid_responses"])
    )
    assert payload["usage"]["requests"] == mechanics["physical_llm_requests"]

    resume = payload["resume"]
    assert resume is not None
    assert resume["accepted_cells_reused"] > 0
    assert resume["rejected_responses_preserved"] > 0
    assert resume["prior_error"]
    assert resume["failure_artifact"].endswith("L1_FAILURE.json")

    traces = payload["traces"][EXPECTED_MAP]
    reference_pairs = [
        (trace["trial_index"], trace["truth_index"]) for trace in traces["strategy_eig"]
    ]
    assert [trial for trial, _truth in reference_pairs] == list(range(30))
    for arm in ARMS:
        assert len(traces[arm]) == 30
        assert [
            (trace["trial_index"], trace["truth_index"]) for trace in traces[arm]
        ] == reference_pairs
        assert all(len(trace["steps"]) == 10 for trace in traces[arm])
        assert all(trace["map_name"] == EXPECTED_MAP for trace in traces[arm])
        assert all(trace["arm"] == arm for trace in traces[arm])

    comparisons: dict[str, Any] = {}
    paired = payload["maps"][EXPECTED_MAP]["paired"]
    for baseline in BASELINES:
        comparison = paired[f"strategy_eig_minus_{baseline}"]
        comparisons[baseline] = {
            "entropy_auc_gain": comparison["entropy_auc_gain_mean"],
            "entropy_auc_ci95": comparison["entropy_auc_gain_ci95"],
            "truth_log_auc_gain": comparison["truth_log_probability_auc_gain_mean"],
            "truth_log_auc_ci95": comparison["truth_log_probability_auc_gain_ci95"],
            "final_entropy_gain": comparison["final_entropy_gain_mean"],
            "final_entropy_ci95": comparison["final_entropy_gain_ci95"],
            "wins_ties_losses": comparison["entropy_auc_wins_ties_losses"],
        }

    primary_passed = all(
        comparison["entropy_auc_ci95"][0] > 0.0 for comparison in comparisons.values()
    )
    corroboration_passed = all(
        comparison["truth_log_auc_gain"] > 0.0
        and comparison["truth_log_auc_ci95"][1] >= 0.0
        for comparison in comparisons.values()
    )
    assert primary_passed == payload["gate_passed"]
    assert primary_passed == payload["maps"][EXPECTED_MAP]["gate_passed"]

    arms: dict[str, Any] = {}
    for arm in ARMS:
        steps = [step for trace in traces[arm] for step in trace["steps"]]
        h2_steps = [step for trace in traces[arm] for step in trace["steps"][:-1]]
        move_count = sum(step["action"].startswith("move-") for step in steps)
        arms[arm] = {
            "move_count": move_count,
            "decision_count": len(steps),
            "move_rate": move_count / len(steps),
            "mean_h2_exhaustive_fraction": sum(
                float(step["exhaustive_fraction"]) for step in h2_steps
            )
            / len(h2_steps),
            "round_entropy_mean": payload["maps"][EXPECTED_MAP]["summary"][arm][
                "round_entropy_mean"
            ],
        }
    return {
        "primary_gate_passed": primary_passed,
        "truth_log_corroboration_passed": corroboration_passed,
        "mechanics_passed": True,
        "comparisons": comparisons,
        "arms": arms,
        "mechanics": mechanics,
        "resume": resume,
        "usage": payload["usage"],
    }


def analyze(
    qualification_payload: dict[str, Any], confirmation_payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        "qualification": _audit_qualification(qualification_payload),
        "confirmation": _audit_confirmation(confirmation_payload),
    }


def render_summary(audit: dict[str, Any]) -> str:
    qualification = audit["qualification"]
    confirmation = audit["confirmation"]
    exact = qualification["comparisons"]["d2_minus_shared_d1"]
    lines = [
        "# RockSample[7,8] StrategyEIG Scale Extension",
        "",
        "Both preregistered stages passed on the canonical eight-rock geometry.",
        "",
        "## Exact Qualification",
        "",
        f"Across 1,000 paired trajectories, exhaustive d2 reduced final entropy by "
        f"`{exact['final_entropy_reduction_mean']:+.4f}` nats relative to exhaustive d1 "
        f"(95% CI `[{exact['final_entropy_reduction_ci95'][0]:+.4f}, "
        f"{exact['final_entropy_reduction_ci95'][1]:+.4f}]`; W/T/L "
        f"`{exact['wins_ties_losses'][0]}/{exact['wins_ties_losses'][1]}/"
        f"{exact['wins_ties_losses'][2]}`).",
        "",
        "## LLM Strategy Confirmation",
        "",
        "Positive paired gains favor StrategyEIG.",
        "",
        "| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |",
        "| --- | --- | --- | --- |",
    ]
    for baseline in BASELINES:
        comparison = confirmation["comparisons"][baseline]
        entropy_ci = comparison["entropy_auc_ci95"]
        truth_ci = comparison["truth_log_auc_ci95"]
        wtl = comparison["wins_ties_losses"]
        lines.append(
            f"| {ARM_LABELS[baseline]} | {comparison['entropy_auc_gain']:+.4f} "
            f"[{entropy_ci[0]:+.4f}, {entropy_ci[1]:+.4f}] | "
            f"{comparison['truth_log_auc_gain']:+.4f} "
            f"[{truth_ci[0]:+.4f}, {truth_ci[1]:+.4f}] | {wtl[0]}/{wtl[1]}/{wtl[2]} |"
        )
    lines.extend(
        [
            "",
            f"- Primary entropy-AUC gate: **{confirmation['primary_gate_passed']}**.",
            f"- Truth-log corroboration: **{confirmation['truth_log_corroboration_passed']}**.",
            f"- Cumulative serving cost: `${confirmation['usage']['run_cost_usd']:.8f}` "
            f"across `{confirmation['usage']['requests']}` requests.",
            f"- Reused accepted cells: `{confirmation['resume']['accepted_cells_reused']}`; "
            f"raw rejected responses retained: "
            f"`{confirmation['mechanics']['raw_rejected_responses']}`.",
            "",
            "## Mechanism",
            "",
            "| Arm | Movement decisions | Movement rate | Mean h2 exhaustive fraction |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm in ARMS:
        diagnostics = confirmation["arms"][arm]
        lines.append(
            f"| {ARM_LABELS[arm]} | {diagnostics['move_count']}/"
            f"{diagnostics['decision_count']} | {diagnostics['move_rate']:.3f} | "
            f"{diagnostics['mean_h2_exhaustive_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "StrategyEIG again spends zero-immediate-information actions on movement before "
            "checking. Shared-roots d1 and exhaustive d1 width cannot value that enabling action, "
            "while random branch policies retain the verifier but remove the LLM search prior.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_entropy(audit: dict[str, Any], output_path: Path) -> None:
    confirmation = audit["confirmation"]
    fig, axis = plt.subplots(figsize=(7.6, 3.8))
    rounds = range(1, 11)
    for arm in ARMS:
        axis.plot(
            rounds,
            confirmation["arms"][arm]["round_entropy_mean"],
            label=ARM_LABELS[arm],
            markersize=4,
            **ARM_STYLES[arm],
        )
    axis.set_title("Canonical RockSample[7,8] diagnosis geometry")
    axis.set_xlabel("Round")
    axis.set_ylabel("Mean posterior entropy (nats)")
    axis.set_xticks(list(rounds))
    axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower left", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualification", type=Path)
    parser.add_argument("confirmation", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(
        json.loads(args.qualification.read_text(encoding="utf-8")),
        json.loads(args.confirmation.read_text(encoding="utf-8")),
    )
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary_output.write_text(render_summary(audit), encoding="utf-8")
    plot_entropy(audit, args.plot_output)
    print(
        json.dumps(
            {
                "qualification_passed": audit["qualification"]["passed"],
                "primary_gate_passed": audit["confirmation"]["primary_gate_passed"],
                "truth_log_corroboration_passed": audit["confirmation"][
                    "truth_log_corroboration_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
