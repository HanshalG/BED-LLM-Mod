import copy
import json

import pytest
from jsonschema import Draft202012Validator

from environments.program_induction.constrained import schema, encode, decode, _options
from environments.program_induction.prior import statement_options, program_probability
from scripts.deepcoder_opportunity import load_dsl, sample_program


def test_source_roundtrip_and_schema_preserve_prior():
    dsl = load_dsl()
    validator = Draft202012Validator(schema(dsl))
    for seed in range(100):
        original = sample_program(dsl, seed)
        encoded = encode(dsl, [original])
        validator.validate(encoded)
        decoded, = decode(dsl, json.dumps(encoded))
        assert str(decoded) == str(original)
        assert program_probability(dsl, decoded) == program_probability(dsl, original)


def test_all_reachable_type_states_have_exact_source_choice_sets():
    dsl = load_dsl()
    s = schema(dsl)
    visited = set()

    def visit(types):
        if types in visited:
            return
        visited.add(types)
        key = ''.join('l' if t is list else 'i' for t in types)
        entry = s['$defs'][key]
        branches = entry.get('anyOf', [entry])
        if len(types) == 6:
            assert branches == [{'type': 'null'}]
            return
        expected = statement_options(dsl, [(f'x{i}', t) for i, t in enumerate(types)], len(types) > 2)
        labels = []
        for b in branches:
            if b['type'] == 'object':
                labels.extend(b['properties']['statement']['enum'])
        expected_labels = [str(dsl.Statement(f'x{len(types)}', op, args)) for op, args in expected]
        assert set(labels) == set(expected_labels)
        assert len(labels) == len(expected_labels)
        assert ({'type': 'null'} in branches) == (len(types) >= 4)
        assert set(_options(dsl, types)) == set(labels)
        for op, _ in expected:
            visit(types+(op.output_type,))

    visit((list, list))
    assert len(visited) == len(s['$defs'])


def valid():
    return {'programs': [{'statement': 'x2 = Sort x0', 'next': {
        'statement': 'x3 = Head x2', 'next': None}}]}


@pytest.mark.parametrize('bad', ['previous', 'type', 'scope', 'early', 'extra', 'late'])
def test_invalid_programs_rejected_by_schema_and_decoder(bad):
    dsl = load_dsl()
    data = valid()
    node = data['programs'][0]
    if bad == 'previous':
        node['next']['statement'] = 'x3 = Head x1'
    if bad == 'type':
        node['next']['statement'] = 'x3 = Access x2 x0'
    if bad == 'scope':
        node['next']['statement'] = 'x3 = Head x4'
    if bad == 'early':
        node['next'] = None
    if bad == 'extra':
        node['secret'] = 1
    if bad == 'late':
        node['next']['next'] = {'statement': 'x4 = Take x3 x0', 'next': {
            'statement': 'x5 = Head x4', 'next': {'statement': 'x6 = Take x5 x0', 'next': None}}}
    assert not Draft202012Validator(schema(dsl)).is_valid(data)
    with pytest.raises(ValueError):
        decode(dsl, json.dumps(data))


def test_whole_batch_rejection_duplicates_and_no_legacy_repair():
    dsl = load_dsl()
    data = valid()
    data['programs'].append(copy.deepcopy(data['programs'][0]))
    assert len(decode(dsl, json.dumps(data))) == 1
    data['programs'][1]['next']['statement'] = 'x3 = Head x1'
    with pytest.raises(ValueError):
        decode(dsl, json.dumps(data))
    with pytest.raises(ValueError):
        decode(dsl, '{"programs":[{"steps":[{"op":"Sort","args":["x0"]}]}]}')
    with pytest.raises(ValueError):
        decode(dsl, '{"programs":[],"programs":[]}')
