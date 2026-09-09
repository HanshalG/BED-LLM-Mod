import json

import pytest

from environments.chembench_mopen.contextual_structure_proposer import (
    PublicScientificContext, build_contextual_messages,
)
from environments.chembench_mopen.structure_proposer import build_messages


CONTEXT = PublicScientificContext('Enzyme assay', 'Initial reaction rate',
                                  ('Temperature is held constant.',))


def args(mode='history_aware', value=.3):
    return dict(history_inputs=[[1, 0, 1, 0, 1, 310, 7]], observations=[value],
                public_bounds=[[0, 10]]*4+[[.1, 2], [280, 340], [4, 10]],
                parameter_bounds=(.01, 10), sigma=.1, mode=mode)


def test_context_fixed_and_blindness_preserved():
    a = build_contextual_messages(public_context=CONTEXT, **args('history_blind', .2))
    b = build_contextual_messages(public_context=CONTEXT, **args('history_blind', 1.8))
    assert a == b
    aware = build_contextual_messages(public_context=CONTEXT, **args())
    assert json.loads(a[1]['content'])['public_scientific_context'] == json.loads(
        aware[1]['content'])['public_scientific_context']
    assert json.loads(aware[1]['content'])['history']


def test_original_payload_grammar_and_prior_unchanged():
    old = build_messages(**args())
    new = build_contextual_messages(public_context=CONTEXT, **args())
    payload = json.loads(new[1]['content'])
    payload.pop('public_scientific_context')
    assert payload == json.loads(old[1]['content'])
    assert new[0]['content'].startswith(old[0]['content'])
    assert build_messages(**args()) == old


def test_context_hash_and_copies_cannot_mutate_source():
    before = CONTEXT.sha256
    payload = CONTEXT.as_payload()
    payload['known_conditions'].append('different')
    assert CONTEXT.sha256 == before
    assert PublicScientificContext(CONTEXT.domain, CONTEXT.measurement, ()).sha256 != before


@pytest.mark.parametrize('values', [('', 'rate', ()), ('x'*513, 'rate', ()),
                                   ('domain', 'rate', []), ('domain', 'rate', ('',)),
                                   ('domain', 'rate', ('x',)*9)])
def test_invalid_context_rejected(values):
    with pytest.raises(ValueError):
        PublicScientificContext(*values)


def test_hidden_metadata_not_an_interface_argument():
    with pytest.raises(TypeError):
        build_contextual_messages(public_context=CONTEXT, true_equation='hidden', **args())
