import json

import pytest

from scripts.ambig_iac_schema_v2_smoke import parse_generated_spec_v2


def _valid_payload():
    return {
        "resources": [
            {"label": "network", "address": "aws_vpc.main"},
            {"label": "subnet", "address": "aws_subnet.public"},
        ],
        "dependencies": [{"source": "subnet", "depends_on": "network"}],
        "attribute_keys": [{"label": "network", "keys": ["cidr_block"]}],
    }


def test_v2_parser_converts_flat_arrays_to_canonical_spec():
    parsed = parse_generated_spec_v2(json.dumps(_valid_payload()))
    assert parsed["resources"] == {
        "network": "aws_vpc.main",
        "subnet": "aws_subnet.public",
    }
    assert parsed["topology"] == {"subnet": ["network"]}
    assert parsed["attributes"] == {"network": {"cidr_block": True}}


def test_v2_parser_rejects_unknown_dependency_label():
    payload = _valid_payload()
    payload["dependencies"][0]["depends_on"] = "missing"
    with pytest.raises(ValueError, match="unknown resource"):
        parse_generated_spec_v2(json.dumps(payload))


def test_v2_parser_rejects_nested_resource_address():
    payload = _valid_payload()
    payload["resources"][0]["address"] = {"type": "aws_vpc"}
    with pytest.raises(ValueError, match="addresses are invalid"):
        parse_generated_spec_v2(json.dumps(payload))


def test_v2_parser_rejects_extra_schema_keys():
    payload = _valid_payload()
    payload["explanation"] = "not permitted"
    with pytest.raises(ValueError, match="exactly the three V2 keys"):
        parse_generated_spec_v2(json.dumps(payload))
