import json

import pytest

from scripts.analyze_tau_knowledge_receding_v3_failure import (
    normalize_diagnostic_scores,
)


def _payload():
    return {
        "followup_1_score": "02",
        "followup_2_score": "35",
        "followup_3_score": "68",
        "followup_4_score": "94",
    }


def test_diagnostic_normalizes_only_zero_padding():
    parsed, noncanonical = normalize_diagnostic_scores(
        json.dumps(_payload())
    )
    assert parsed["scores"] == [2, 35, 68, 94]
    assert noncanonical


def test_diagnostic_rejects_non_digit_or_out_of_band_values():
    payload = _payload()
    payload["followup_1_score"] = 2
    with pytest.raises(ValueError, match="digit string"):
        normalize_diagnostic_scores(json.dumps(payload))
    payload = _payload()
    payload["followup_1_score"] = "22"
    with pytest.raises(ValueError, match="valid band"):
        normalize_diagnostic_scores(json.dumps(payload))
