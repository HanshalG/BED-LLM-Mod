from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_paprika_headline import analyze


def _write_run(
    root: Path,
    methods: dict[str, list[int | None]],
    *,
    start: int = 10,
) -> None:
    items = []
    for item_index, (method, resolution_turns) in enumerate(methods.items()):
        item_dir = root / "items" / f"{item_index:03d}_{method}"
        item_dir.mkdir(parents=True)
        records = []
        for offset, resolution in enumerate(resolution_turns):
            turns = []
            for turn in range(1, (resolution or 5) + 1):
                extras = None
                if method in {"NaivePrimaryArbitration", "NaivePrimaryCandidate0"}:
                    extras = {
                        "candidate_queries": ["native", "alternative-1", "alternative-2"],
                        "selected_index": 0,
                        "native_overridden": False,
                    }
                turns.append(
                    {
                        "query": "native",
                        "reply": f"reply-{turn}",
                        "mapped_cleanly": True,
                        "goal_reached": resolution == turn,
                        "selection_extras": extras,
                    }
                )
            records.append(
                {
                    "task_id": f"customer_service:eval:{start + offset:04d}",
                    "turns": turns,
                }
            )
        artifact = item_dir / "paprika_smoke.json"
        artifact.write_text(json.dumps(records))
        metrics = {
            "structured_parse_failures": [0],
            "simulator_faithfulness_observations": [50],
            "simulator_faithfulness_checks": [50],
            "simulator_faithfulness_raw_contradictions": [0],
            "simulator_faithfulness_repairs": [0],
            "simulator_faithfulness_failures": [0],
            "simulator_faithfulness_final_inconsistency_rate": [0],
            "simulator_terminal_claims": [1],
            "simulator_terminal_checks": [1],
            "simulator_terminal_rejections": [0],
            "backend_cost_usd": [1.0],
            "backend_requests": [100],
        }
        items.append(
            {
                "method": method,
                "artifacts": {"paprika_smoke": str(artifact.relative_to(root))},
                "metrics": metrics,
            }
        )
    (root / "metrics.json").write_text(json.dumps({"items": items}))


def test_headline_analyzer_enforces_tasks_pairing_and_two_comparisons(
    tmp_path: Path,
) -> None:
    headline = tmp_path / "headline"
    nonthinking = tmp_path / "nonthinking"
    arbitration = [1] * 20 + [2] * 10 + [None] * 20
    candidate0 = [2] * 20 + [3] * 10 + [None] * 20
    thinking = [3] * 20 + [4] * 10 + [None] * 20
    nonthinking_turns = [4] * 20 + [5] * 10 + [None] * 20
    _write_run(
        headline,
        {
            "NaivePrimaryArbitration": arbitration,
            "NaivePrimaryCandidate0": candidate0,
            "naive": thinking,
        },
    )
    _write_run(nonthinking, {"naive": nonthinking_turns})
    result = analyze(headline, nonthinking)
    assert result["claim_read_before_manual_review"] == "claim_b_confirmed"
    assert result["candidate_pairing"]["valid"] is True
    assert result["candidate_pairing"]["eligible_identical_history_turns"] >= 50
    assert result["primary_arbitration_vs_naive_thinking"]["wins"] == 30
    assert result["coprimary_arbitration_vs_candidate0"]["wins"] == 30
    assert result["manual_review_required"] is True
    assert len(result["manual_review_plan"]["random_spot_check_task_ids"]) == 10


def test_headline_analyzer_adds_best_n_context_without_changing_claim_rule(
    tmp_path: Path,
) -> None:
    headline = tmp_path / "headline"
    nonthinking = tmp_path / "nonthinking"
    best_n = tmp_path / "best_n"
    arbitration = [1] * 20 + [2] * 10 + [None] * 20
    candidate0 = [2] * 20 + [3] * 10 + [None] * 20
    thinking = [3] * 20 + [4] * 10 + [None] * 20
    best_n_turns = [2] * 20 + [3] * 10 + [None] * 19 + [1]
    _write_run(
        headline,
        {
            "NaivePrimaryArbitration": arbitration,
            "NaivePrimaryCandidate0": candidate0,
            "naive": thinking,
        },
    )
    _write_run(nonthinking, {"naive": [4] * 30 + [None] * 20})
    _write_run(best_n, {"EIG": best_n_turns})
    result = analyze(headline, nonthinking, best_n_run=best_n)
    assert result["claim_read_before_manual_review"] == "claim_b_confirmed"
    assert result["best_n_endpoint_valid_automated"] is True
    assert result["context_best_n_vs_naive_thinking"]["wins"] == 31
    assert result["context_best_n_vs_arbitration"]["losses"] == 30
    assert (
        "customer_service:eval:0059"
        in result["manual_review_plan"]["success_disagreement_task_ids"]
    )


def test_headline_analyzer_rejects_mismatched_root_proposals(tmp_path: Path) -> None:
    headline = tmp_path / "headline"
    nonthinking = tmp_path / "nonthinking"
    turns = [None] * 50
    _write_run(
        headline,
        {
            "NaivePrimaryArbitration": turns,
            "NaivePrimaryCandidate0": turns,
            "naive": turns,
        },
    )
    _write_run(nonthinking, {"naive": turns})
    payload = json.loads(
        (headline / "items/001_NaivePrimaryCandidate0/paprika_smoke.json").read_text()
    )
    payload[0]["turns"][0]["selection_extras"]["candidate_queries"][0] = "different"
    (headline / "items/001_NaivePrimaryCandidate0/paprika_smoke.json").write_text(
        json.dumps(payload)
    )
    result = analyze(headline, nonthinking)
    assert result["candidate_pairing"]["valid"] is False
    assert result["claim_read_before_manual_review"] == "invalid_candidate_pairing"
