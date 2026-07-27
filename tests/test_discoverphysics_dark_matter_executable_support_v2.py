from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.discoverphysics_dark_matter_executable_support_v2 import (
    parse_weighted_support,
)
from tests.test_discoverphysics_dark_matter_executable_support import (
    _valid_payload,
)


def _weighted_payload() -> dict:
    payload = _valid_payload()
    for hypothesis in payload["hypotheses"]:
        probability = hypothesis.pop("probability")
        hypothesis["weight"] = int(round(probability * 100))
    return payload


def test_weighted_parser_normalizes_integer_weights():
    support = parse_weighted_support(json.dumps(_weighted_payload()))

    assert np.isclose(sum(item["probability"] for item in support), 1.0)
    assert np.isclose(
        sum(item["probability"] for item in support if item["region"] == "NE"),
        0.4,
    )


def test_weighted_parser_rejects_probability_field():
    payload = _weighted_payload()
    payload["hypotheses"][0]["probability"] = 0.2

    with pytest.raises(ValueError, match="must use weight only"):
        parse_weighted_support(json.dumps(payload))


def test_weighted_parser_rejects_noninteger_weight():
    payload = _weighted_payload()
    payload["hypotheses"][0]["weight"] = 20.5

    with pytest.raises(ValueError, match="integer"):
        parse_weighted_support(json.dumps(payload))
