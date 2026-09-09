"""Preserve known schema/parser discrepancies without changing the closed run."""
import json
import pytest
from scripts.rearc_mechanism_interface import parse_plan,schema


def value(description='rule'):
    return {'plans':[{'id':f'p{i}','description':description} for i in range(4)]}


def test_character_limit_differs_from_frozen_byte_parser():
    item = schema('plan')['json_schema']['schema']['properties']['plans']['items']
    limit = item['properties']['description']['maxLength']
    ascii_plan = value('a'*limit)
    assert parse_plan(json.dumps(ascii_plan))==ascii_plan
    unicode_plan = value('a'*(limit-1)+'\u2014')
    assert len(unicode_plan['plans'][0]['description'])==limit
    assert len(unicode_plan['plans'][0]['description'].encode())>limit
    with pytest.raises(ValueError,match='ordered bounded plan'):
        parse_plan(json.dumps(unicode_plan))


def test_array_enum_does_not_express_required_order_or_unique_ids():
    item = schema('plan')['json_schema']['schema']['properties']['plans']['items']
    for entries in (value()['plans'][::-1],[value()['plans'][0]]*4):
        assert all(row['id'] in item['properties']['id']['enum'] for row in entries)
        with pytest.raises(ValueError,match='ordered bounded plan'):
            parse_plan(json.dumps({'plans':entries}))


def test_schema_minlength_does_not_express_nonblank_text():
    item = schema('plan')['json_schema']['schema']['properties']['plans']['items']
    assert len(' ')>=item['properties']['description']['minLength']
    with pytest.raises(ValueError,match='ordered bounded plan'):
        parse_plan(json.dumps(value(' ')))
