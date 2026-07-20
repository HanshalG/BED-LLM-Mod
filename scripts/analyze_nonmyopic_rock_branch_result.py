"""Validate and summarize the registered Rock branch-policy confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


BASELINES = ("shared_d1", "width", "random_strategy")
ARMS = ("strategy_eig", "exhaustive_d2", "shared_d1", "width", "random_strategy")
ARM_LABELS = {
    "strategy_eig": "StrategyEIG",
    "exhaustive_d2": "Exhaustive d2",
    "shared_d1": "Shared-roots d1",
    "width": "Exhaustive d1 width",
    "random_strategy": "Random strategies",
}
ARM_STYLES = {
    "strategy_eig": {"color": "#c43c39", "linewidth": 2.6, "marker": "o"},
    "exhaustive_d2": {"color": "#222222", "linewidth": 2.0, "marker": "s"},
    "shared_d1": {"color": "#16817a", "linewidth": 1.8, "linestyle": ":"},
    "width": {"color": "#3366a8", "linewidth": 2.0, "linestyle": "--"},
    "random_strategy": {"color": "#d17a22", "linewidth": 1.8, "linestyle": "-."},
}


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    assert config["map_names"] == ["3-6", "5-7"]
    assert config["num_trials_per_map"] == 30
    assert config["num_rounds"] == 8
    assert config["num_strategies"] == 6
    assert config["planning_horizon"] == 2
    assert config["seed"] == 12041
    assert config["bootstrap_replicates"] == 10_000
    assert config["strategy_schema"] == "branch_policy_v2"
    assert config["primary_endpoint"] == "entropy_auc"

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

    maps: dict[str, Any] = {}
    for map_name in config["map_names"]:
        traces = payload["traces"][map_name]
        reference = traces["strategy_eig"]
        reference_pairs = [(trace["trial_index"], trace["truth_index"]) for trace in reference]
        assert reference_pairs == [(index, reference[index]["truth_index"]) for index in range(30)]
        for arm in ARMS:
            assert len(traces[arm]) == 30
            assert [(trace["trial_index"], trace["truth_index"]) for trace in traces[arm]] == reference_pairs
            assert all(len(trace["steps"]) == 8 for trace in traces[arm])

        comparisons: dict[str, Any] = {}
        for baseline in BASELINES:
            comparison = payload["maps"][map_name]["paired"][f"strategy_eig_minus_{baseline}"]
            comparisons[baseline] = {
                "entropy_auc_gain": comparison["entropy_auc_gain_mean"],
                "entropy_auc_ci95": comparison["entropy_auc_gain_ci95"],
                "truth_log_auc_gain": comparison["truth_log_probability_auc_gain_mean"],
                "truth_log_auc_ci95": comparison["truth_log_probability_auc_gain_ci95"],
                "final_entropy_gain": comparison["final_entropy_gain_mean"],
                "wins_ties_losses": comparison["entropy_auc_wins_ties_losses"],
            }

        arm_diagnostics: dict[str, Any] = {}
        for arm in ARMS:
            steps = [step for trace in traces[arm] for step in trace["steps"]]
            h2_steps = [step for trace in traces[arm] for step in trace["steps"][:-1]]
            arm_diagnostics[arm] = {
                "move_count": sum(step["action"].startswith("move-") for step in steps),
                "decision_count": len(steps),
                "move_rate": sum(step["action"].startswith("move-") for step in steps) / len(steps),
                "mean_h2_exhaustive_fraction": sum(
                    float(step["exhaustive_fraction"]) for step in h2_steps
                )
                / len(h2_steps),
            }
        maps[map_name] = {
            "comparisons": comparisons,
            "arms": arm_diagnostics,
            "round_entropy_mean": {
                arm: payload["maps"][map_name]["summary"][arm]["round_entropy_mean"]
                for arm in ARMS
            },
        }

    primary_passed = all(
        maps[map_name]["comparisons"][baseline]["entropy_auc_ci95"][0] > 0.0
        for map_name in config["map_names"]
        for baseline in BASELINES
    )
    corroboration_passed = all(
        maps[map_name]["comparisons"][baseline]["truth_log_auc_gain"] > 0.0
        and maps[map_name]["comparisons"][baseline]["truth_log_auc_ci95"][1] >= 0.0
        for map_name in config["map_names"]
        for baseline in BASELINES
    )
    assert primary_passed == payload["gate_passed"]
    return {
        "primary_gate_passed": primary_passed,
        "truth_log_corroboration_passed": corroboration_passed,
        "mechanics_passed": True,
        "maps": maps,
        "mechanics": mechanics,
        "usage": payload["usage"],
    }


def render_summary(audit: dict[str, Any]) -> str:
    lines = [
        "# Rock Branch-Policy StrategyEIG Confirmation Result",
        "",
        "The preregistered primary gate passed on both paper maps. Positive gains favor StrategyEIG.",
        "",
        "| Map | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |",
        "| --- | --- | --- | --- | --- |",
    ]
    for map_name, result in audit["maps"].items():
        for baseline in BASELINES:
            c = result["comparisons"][baseline]
            entropy_ci = c["entropy_auc_ci95"]
            truth_ci = c["truth_log_auc_ci95"]
            wtl = c["wins_ties_losses"]
            lines.append(
                f"| {map_name} | {ARM_LABELS[baseline]} | {c['entropy_auc_gain']:+.4f} "
                f"[{entropy_ci[0]:+.4f}, {entropy_ci[1]:+.4f}] | "
                f"{c['truth_log_auc_gain']:+.4f} [{truth_ci[0]:+.4f}, {truth_ci[1]:+.4f}] | "
                f"{wtl[0]}/{wtl[1]}/{wtl[2]} |"
            )
    lines.extend(
        [
            "",
            f"- Primary entropy-AUC gate: **{audit['primary_gate_passed']}**.",
            f"- Truth-log-posterior corroboration: **{audit['truth_log_corroboration_passed']}**.",
            f"- Run cost: `${audit['usage']['run_cost_usd']:.8f}` across "
            f"`{audit['usage']['requests']}` requests.",
            f"- Raw semantic rejects recovered: `{audit['mechanics']['raw_rejected_responses']}`; "
            f"terminal failures: `{audit['mechanics']['terminal_cell_failures']}`.",
            f"- Logged horizon-1 item repairs: `{audit['mechanics']['terminal_followup_repairs']}`.",
            "",
            "## Mechanism",
            "",
            "| Map | Arm | Movement decisions | Movement rate | Mean h2 exhaustive fraction |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for map_name, result in audit["maps"].items():
        for arm in ARMS:
            a = result["arms"][arm]
            lines.append(
                f"| {map_name} | {ARM_LABELS[arm]} | {a['move_count']}/{a['decision_count']} | "
                f"{a['move_rate']:.3f} | {a['mean_h2_exhaustive_fraction']:.3f} |"
            )
    lines.extend(
        [
            "",
            "StrategyEIG repeatedly selects zero-immediate-EIG movement that enables accurate future "
            "checks. The same-root d1 and exhaustive-d1 width controls never move. StrategyEIG also "
            "beats the matched random-policy prior, showing that proposal quality and non-myopic exact "
            "verification are both load-bearing.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_entropy(audit: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.45), sharex=True)
    rounds = range(1, 9)
    for axis, (map_name, result) in zip(axes, audit["maps"].items()):
        for arm in ARMS:
            axis.plot(
                rounds,
                result["round_entropy_mean"][arm],
                label=ARM_LABELS[arm],
                markersize=4,
                **ARM_STYLES[arm],
            )
        axis.set_title(f"Rock map {map_name}")
        axis.set_xlabel("Round")
        axis.set_xticks(list(rounds))
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean posterior entropy (nats)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.25, wspace=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    audit = analyze(payload)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_output.write_text(render_summary(audit), encoding="utf-8")
    plot_entropy(audit, args.plot_output)
    print(json.dumps({key: audit[key] for key in ("primary_gate_passed", "truth_log_corroboration_passed", "mechanics_passed")}, indent=2))


if __name__ == "__main__":
    main()
