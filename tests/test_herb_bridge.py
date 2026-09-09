"""Replay checks on the banked real Julia iterator output, without API calls."""
import json
from pathlib import Path
import tomllib

import pytest

from scripts.rearc_program_graph import validate_graph

ROOT = Path(__file__).resolve().parents[1] / 'results/nonmyopic/herb_bridge_runtime_20260909'


def rows():
    return tomllib.loads((ROOT / 'candidates.toml').read_text())['rows']


def test_nested_callable_lowered_to_prior_reference():
    graph = next(r['graph'] for r in rows() if r['expression'] == 'apply(compose(identity, identity), I)')
    assert graph['steps'] == [
        {'id': 'x0', 'op': 'compose', 'args': ['identity', 'identity']},
        {'id': 'x1', 'op': 'apply', 'args': ['x0', 'I']},
    ]
    validate_graph(graph, {'compose', 'identity', 'apply'}, set())


def test_repeated_subexpression_shares_graph_node():
    graph = next(r['graph'] for r in rows() if r['expression'] == 'paint(vmirror(I), asobject(vmirror(I)))')
    assert len(graph['steps']) == 3
    assert graph['steps'][1]['args'] == ['x0']
    assert graph['steps'][2]['args'] == ['x0', 'x1']


def test_bad_field_is_rejected():
    old = tomllib.loads((ROOT / 'v1_wrong_field.toml').read_text())['rows'][0]['graph']
    with pytest.raises(ValueError, match='step fields'):
        validate_graph(old, {'identity'}, set())


def test_runtime_receipts_and_scope():
    result = json.loads((ROOT / 'result.json').read_text())
    assert result['status'] == 'bridge_fixture_pass'
    assert len(result['records']) == 7
    assert not result['full_dsl_search_qualified'] and not result['depth_result']
    for row in result['records']:
        receipt = row['result']
        assert receipt['uid'] == 65534
        assert all(receipt[k] for k in ('read_only', 'network_denied', 'no_api_key'))
