import copy
import json
from pathlib import Path

import pytest

from scripts.analyze_nonmyopic_rocksample_scaling import analyze


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "results/nonmyopic/rock_branch_strategy_v2_confirmation_20260720/L1.json",
    ROOT / "results/nonmyopic/rocksample_7_8_scale_20260721/L1.json",
    ROOT
    / "results/nonmyopic/rocksample_11_11_gemma_slot_confirmation_20260721/L1.json",
    ROOT / "results/nonmyopic/rocksample_15_15_vllm_replication_20260722/L1.json",
)


def _payloads() -> list[dict]:
    return [json.loads(path.read_text()) for path in SOURCES]


def test_scaling_audit_uses_all_confirmed_maps() -> None:
    audit = analyze(_payloads())

    assert [row["map_name"] for row in audit["rows"]] == [
        "3-6",
        "5-7",
        "7-8",
        "11-11",
        "15-15",
    ]
    assert audit["rows"][-1]["hidden_states"] == 32768
    assert audit["rows"][-1]["num_strategies"] == 4
    assert audit["rows"][-1]["exhaustive_to_strategy_unit_ratio"] == pytest.approx(
        64.2645833333
    )
    assert audit["rollout_scoring_llm_calls"] == 0


def test_scaling_audit_rejects_wrong_run() -> None:
    payloads = copy.deepcopy(_payloads())
    payloads[1]["run_id"] = "wrong"

    with pytest.raises(AssertionError):
        analyze(payloads)
