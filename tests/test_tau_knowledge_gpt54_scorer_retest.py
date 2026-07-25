from __future__ import annotations

from scripts.analyze_tau_knowledge_gpt54_scorer_retest import (
    analyze_replicates,
    mean_pairwise_agreement,
)
from scripts.tau_knowledge_cross_model_scorer import (
    CONFIRMATION_ARTIFACT_SHA256,
    NONSEMANTIC_ANALYSIS_SHA256,
)
from scripts.tau_knowledge_gpt54_scorer_retest import (
    INTERFACE_VERSION,
    MODEL_ID,
)
from scripts.tau_knowledge_receding_continuation import (
    FRESH_CONFIRMATION_IDS,
)


def _payload(replicate_index: int) -> dict:
    policies = [
        {
            "task_id": task_id,
            "nonmyopic_root_index": 1,
            "nonmyopic_receding_value": 2,
            "myopic_receding_value": 1,
            "nonmyopic_advantage_over_myopic": 1,
        }
        for task_id in FRESH_CONFIRMATION_IDS
    ]
    roots = [{"selected_followup_index": 2} for _ in range(100)]
    return {
        "status": "passed",
        "protocol": {
            "stage": "confirmation",
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "replicate_index": replicate_index,
            "source_artifact_sha256": CONFIRMATION_ARTIFACT_SHA256,
            "nonsemantic_analysis_sha256": NONSEMANTIC_ANALYSIS_SHA256,
            "task_ids": list(FRESH_CONFIRMATION_IDS),
        },
        "summary": {
            "gates": {"all_pass": True},
            "nonmyopic_root_pairwise_accuracy": 0.70,
            "myopic_root_pairwise_accuracy": 0.55,
            "root_pairwise_accuracy_gain": 0.15,
            "focused_pairwise_accuracy": 0.75,
            "focused_optimal_followup_rate": 0.80,
            "focused_mean_regret": 0.20,
            "policy_diagnostics": policies,
            "root_diagnostics": roots,
        },
        "usage": {
            "physical_requests": 140,
            "reasoning_tokens": 0,
            "adapter_cost_usd": 1.40,
        },
    }


def test_mean_pairwise_agreement() -> None:
    assert mean_pairwise_agreement(
        [[0, 1, 2, 3], [0, 1, 3, 3], [0, 2, 2, 3]]
    ) == 2 / 3


def test_retest_summary_passes_frozen_gates() -> None:
    result = analyze_replicates([_payload(1), _payload(2), _payload(3)])
    assert result["status"] == "passed"
    assert result["summary"]["total_physical_requests"] == 420
    assert result["summary"]["endpoint_pass_count"] == 3


def test_retest_summary_requires_two_original_gate_passes() -> None:
    payloads = [_payload(1), _payload(2), _payload(3)]
    payloads[0]["summary"]["gates"]["all_pass"] = False
    payloads[1]["summary"]["gates"]["all_pass"] = False
    result = analyze_replicates(payloads)
    assert result["status"] == "gate_failed"
    assert not result["summary"]["gates"][
        "at_least_2_original_all_gate_passes"
    ]
