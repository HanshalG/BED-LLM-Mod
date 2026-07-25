from __future__ import annotations

import copy
from pathlib import Path

from helpers import load_config
from scripts import tau_knowledge_belief_alignment_ablation as ablation


ROOT = Path(__file__).resolve().parents[1]
SMOKE = (
    ROOT
    / "results/nonmyopic/tau_knowledge_receding_continuation_v3_smoke"
    / "tau-knowledge-receding-v3-smoke-20260725T023000Z"
    / "SERVING_SMOKE.json"
)
CONFIRMATION = (
    ROOT
    / "results/nonmyopic/"
    "tau_knowledge_receding_continuation_v3_1_confirmation"
    / "tau-knowledge-receding-v3-1-confirmation-20260725T030000Z"
    / "CONFIRMATION.json"
)


def test_derangement_changes_only_refreshed_belief_alignment():
    source = ablation.load_source(CONFIRMATION, stage="confirmation")
    transformed, diagnostics = ablation.derange_refreshed_beliefs(
        source["records"]
    )
    intervention = ablation.validate_intervention(
        source["records"], transformed, diagnostics
    )

    assert intervention["all_permutations_are_derangements"]
    assert intervention["all_belief_multisets_preserved"]
    assert intervention["nonbelief_fields_unchanged"]
    assert all(
        row["source_branch_for_target"]
        == ablation._task_derangement(row["task_id"], 5)
        for row in diagnostics
    )


def test_derangement_is_deterministic_and_does_not_mutate_source():
    source = ablation.load_source(SMOKE, stage="serving_smoke")
    original = copy.deepcopy(source["records"])
    first, first_diagnostics = ablation.derange_refreshed_beliefs(
        source["records"]
    )
    second, second_diagnostics = ablation.derange_refreshed_beliefs(
        source["records"]
    )

    assert source["records"] == original
    assert first == second
    assert first_diagnostics == second_diagnostics


def test_full_alignment_compared_with_itself_has_zero_effect():
    source = ablation.load_source(CONFIRMATION, stage="confirmation")
    comparison = ablation.compare_to_full_alignment(
        source,
        source["summary"],
        source["nonmyopic_scores"],
    )

    assert comparison["full_minus_shuffled_root_accuracy"] == 0
    assert comparison["full_minus_shuffled_continuation_accuracy"] == 0
    assert comparison["full_minus_shuffled_endpoint_total"] == 0


def test_smoke_summary_uses_ablation_request_count(monkeypatch):
    source = ablation.load_source(SMOKE, stage="serving_smoke")

    def fake_summarize(*args, **kwargs):
        return {
            "root_diagnostics": [{}] * 10,
            "policy_diagnostics": [],
            "gates": {"original_gate": False},
        }

    monkeypatch.setattr(ablation, "summarize", fake_summarize)
    intervention = {
        "all_permutations_are_derangements": True,
        "all_belief_multisets_preserved": True,
        "nonbelief_fields_unchanged": True,
    }
    summary, comparison = ablation.summarize_ablation(
        source,
        [[]] * 2,
        {"physical_requests": 12, "reasoning_tokens": 0},
        stage="serving_smoke",
        shuffled_root_scores=[
            {"scores": [1, 2, 3, 4, 5]},
            {"scores": [1, 2, 3, 4, 5]},
        ],
        intervention=intervention,
    )

    assert comparison is None
    assert summary["reference_efficacy_gates"] == {"original_gate": False}
    assert summary["gates"]["all_pass"]
    assert summary["gates"]["exact_physical_request_count"]


def test_ablation_config_selects_frozen_model_and_seed():
    config = load_config(
        str(
            ROOT
            / "configs/"
            "config_tau_knowledge_belief_alignment_ablation_openrouter.yaml"
        )
    )

    assert config.model_pairs[0].questioner.model == "openai/gpt-5.4"
    assert config.mediq_seed == 24343
