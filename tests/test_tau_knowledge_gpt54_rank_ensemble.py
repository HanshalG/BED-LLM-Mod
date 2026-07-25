from __future__ import annotations

import glob
import json
from pathlib import Path

from scripts.analyze_tau_knowledge_gpt54_rank_ensemble import (
    aggregate_score_rows,
    analyze_blocks,
    midranks,
)
from scripts.tau_knowledge_cross_model_scorer import load_records


def test_midranks_handle_ties() -> None:
    assert midranks([10, 20, 20, 5]) == [1.0, 2.5, 2.5, 0.0]


def test_rank_aggregation_removes_score_scale() -> None:
    rows = [
        [{"scores": [0, 10, 20], "best_followup_indices": []}],
        [{"scores": [0, 1, 2], "best_followup_indices": []}],
        [{"scores": [10, 20, 30], "best_followup_indices": []}],
    ]
    result = aggregate_score_rows(rows)
    assert result[0]["scores"] == [0.0, 1.0, 2.0]


def test_frozen_development_rank_ensemble_passes_as_disclosed() -> None:
    paths = sorted(
        glob.glob(
            "results/nonmyopic/tau_knowledge_gpt54_scorer_retest/"
            "*/REPLICATE_*.json"
        )
    )
    payloads = [json.loads(Path(path).read_text()) for path in paths]
    source = Path(
        "results/nonmyopic/"
        "tau_knowledge_receding_continuation_v3_1_confirmation/"
        "tau-knowledge-receding-v3-1-confirmation-20260725T030000Z/"
        "CONFIRMATION.json"
    )
    records = load_records(source, stage="confirmation")
    result = analyze_blocks(payloads, payloads, records=records)
    assert result["status"] == "passed"
    summary = result["summary"]["development"]
    assert summary["endpoint_total"] == 30
    assert summary["myopic_endpoint_total"] == 26
    assert summary["myopic_wins"] == 5
    assert summary["myopic_losses"] == 2
