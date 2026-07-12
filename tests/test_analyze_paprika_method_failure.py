from scripts.analyze_paprika_method_failure import (
    action_summary,
    arbitration_diagnostics,
)


def _turn(
    query: str,
    *,
    goal: bool = False,
    clean: bool = True,
    selected_index: int = 0,
    overridden: bool = False,
) -> dict:
    scores = [0.1, 0.3, 0.2]
    return {
        "query": query,
        "kind": "solution" if goal else "diagnostic",
        "outcomes": ["yes", "no", "unknown"],
        "reply": f"reply to {query}",
        "mapped_cleanly": clean,
        "goal_reached": goal,
        "selection_extras": {
            "candidate_scores": scores,
            "selected_index": selected_index,
            "native_overridden": overridden,
            "score_gap_vs_native": scores[selected_index] - scores[0],
            "one_se_threshold": 0.05,
        },
    }


def test_action_summary_counts_solution_actions_and_mapping_coverage() -> None:
    records = [
        {
            "task_id": "task-0",
            "turns": [
                _turn("inspect", clean=False),
                _turn("repair", goal=True, selected_index=1, overridden=True),
            ],
            "final_metrics": {"true_solution_exact_mass": 0.0},
        }
    ]

    summary = action_summary(records, round_budget=3)

    assert summary["resolved"] == 1
    assert summary["resolution_curve"] == [0.0, 1.0, 1.0]
    assert summary["action_kinds"] == {"diagnostic": 1, "solution": 1}
    assert summary["mapping_coverage"] == 0.5
    assert summary["true_solution_exact_mass"] == {
        "logged_tasks": 1,
        "nonzero_tasks": 0,
    }


def test_arbitration_diagnostics_tracks_first_divergence_and_stale_override() -> None:
    shared_unclean = _turn("shared", clean=False)
    selected = _turn("selected", selected_index=1, overridden=True)
    default = _turn("default", goal=True)
    arbitration = [
        {"task_id": "task-0", "turns": [shared_unclean, selected]},
    ]
    candidate0 = [
        {"task_id": "task-0", "turns": [shared_unclean, default]},
    ]

    result = arbitration_diagnostics(arbitration, candidate0, round_budget=2)

    assert result["override_count"] == 1
    assert result["overrides_immediately_after_unclean_mapping"] == 1
    assert result["native_default_eig_rank"] == {"3": 2}
    assert result["first_divergence"]["count"] == 1
    assert result["first_divergence"]["immediate_resolution_pair"] == {
        "default_only": 1
    }
    assert result["first_divergence"]["eventual_censored_turn_outcome"] == {
        "loss": 1
    }
