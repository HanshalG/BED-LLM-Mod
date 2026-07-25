import json

import pytest

from scripts.tau_knowledge_receding_continuation_v3_1 import (
    parse_zero_padding_scores,
)


def _payload():
    return {
        "followup_1_score": "02",
        "followup_2_score": "35",
        "followup_3_score": "68",
        "followup_4_score": "94",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0), ("00", 0), ("2", 2), ("02", 2), ("35", 35), ("99", 99)],
)
def test_v3_1_accepts_one_or_two_digit_band_values(value, expected):
    payload = _payload()
    payload["followup_1_score"] = value
    parsed = parse_zero_padding_scores(json.dumps(payload))
    assert parsed["scores"][0] == expected


@pytest.mark.parametrize("value", [2, "002", "10", "29", "40", "100"])
def test_v3_1_rejects_other_types_lengths_and_bands(value):
    payload = _payload()
    payload["followup_1_score"] = value
    with pytest.raises(ValueError):
        parse_zero_padding_scores(json.dumps(payload))


def test_v3_1_rejects_extra_keys():
    payload = _payload()
    payload["explanation"] = "extra"
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_zero_padding_scores(json.dumps(payload))
