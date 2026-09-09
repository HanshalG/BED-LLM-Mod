import hashlib
import json
import jsonschema
import pytest
from scripts import tree_copy_fidelity as probe
from tests.test_physgym_semantic_probe import response


def test_expected_target_and_paired_inputs():
    for arm in ('recursive', 'string'):
        payload = probe.TARGET if arm == 'recursive' else {'payload': json.dumps(probe.TARGET)}
        body = probe.body(arm)
        jsonschema.validate(payload, body['response_format']['json_schema']['schema'])
        assert probe.measure(json.dumps(payload), arm)['exact_copy']
    a, b = [probe.body(arm) for arm in ('recursive', 'string')]
    assert a['seed'] == b['seed'] and a['reasoning'] == b['reasoning']
    assert a['messages'][1] == b['messages'][1]


def test_wrong_or_invalid_response_is_not_a_pass():
    assert not probe.measure('{"trees":[{"op":"pi"}]}', 'recursive')['exact_copy']
    assert not probe.measure('{"payload":"oops"}', 'string')['format_valid']
    assert not probe.measure('{"trees":[]}', 'recursive')['format_valid']


def test_full_paired_null_still_calls_control_and_replays(tmp_path):
    class Block:
        root = tmp_path
        report = dict(status='incomplete', calls=0, accepted_cost_usd=.002,
                      protocol_sha256=probe.PROTOCOL_SHA,
                      implementation_sha256={p: hashlib.sha256(probe.Path(p).read_bytes()).hexdigest() for p in probe.BINDINGS})
        def request(self, tag, body):
            self.report['calls'] += 1
            raw = response()
            raw['choices'][0]['message']['content'] = ('{"trees":[{"op":"pi"}]}' if tag == 'recursive'
                                                       else json.dumps({'payload': json.dumps(probe.TARGET)}))
            (tmp_path/(tag+'.request.json')).write_text(json.dumps(body))
            (tmp_path/(tag+'.response.json')).write_text(json.dumps(raw))
            return raw
    block = Block()
    probe.collect(block)
    (tmp_path/'result.json').write_text(json.dumps(block.report))
    assert block.report['calls'] == 2 and not block.report['both_copy']
    assert probe.replay(tmp_path)['results']['string']['exact_copy']
    path = tmp_path/'recursive.request.json'
    data = json.loads(path.read_text())
    data['seed'] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='request changed'):
        probe.replay(tmp_path)


def test_banked_nontrivial_copy_replay():
    result = probe.replay(probe.ROOT)
    assert all(row['exact_copy'] for row in result['results'].values())
    assert result['cost'] == pytest.approx(.0003266)
