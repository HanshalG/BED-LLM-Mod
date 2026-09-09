import hashlib
import json
import pytest
from scripts import compositional_edit_qualification as probe
from tests.test_physgym_semantic_probe import response


def fake():
    raw = response()
    raw['choices'][0]['message']['content'] = '{"edits":[{"operation":"multiply","correction":{"op":"constant","value":1}}]}'
    return raw


def test_history_privacy_and_source_separation():
    case = probe.cases()['0']
    a, b = [probe.body(case, arm, 0) for arm in ('control', 'refresh')]
    assert a['seed'] == b['seed'] and a['messages'][0] == b['messages'][0]
    pa, pb = [json.loads(v['messages'][1]['content']) for v in (a, b)]
    assert len(pa['history']) == 3 and len(pb['history']) == 6
    assert pa['history'] == pb['history'][:3]
    assert not {'targets', 'truth', 'source', 'task_id'} & set(pa)
    assert probe.SOURCES[0] not in json.dumps(a)
    assert case['history'][3][1] not in [r['observed_log_response'] for r in pa['history']]


def test_full_eight_call_replay_and_seal(tmp_path):
    class Block:
        root = tmp_path
        report = dict(status='incomplete', calls=0, accepted_cost_usd=.008,
            protocol_sha256=probe.PROTOCOL_SHA,
            implementation_sha256={p: hashlib.sha256(probe.Path(p).read_bytes()).hexdigest() for p in probe.BINDINGS})
        tags = []
        def request(self, tag, body):
            self.tags.append(tag)
            self.report['calls'] += 1
            raw = fake()
            (tmp_path/(tag+'.request.json')).write_text(json.dumps(body))
            (tmp_path/(tag+'.response.json')).write_text(json.dumps(raw))
            return raw
    block, public = Block(), probe.cases()
    def endpoints():
        assert block.report['calls'] == 8 and (tmp_path/'forecasts.json').exists()
        return probe.outcomes(public)
    probe.collect(block, public, endpoints)
    (tmp_path/'result.json').write_text(json.dumps(block.report))
    assert block.tags == ['0_control', '0_refresh', '1_refresh', '1_control', '2_control', '2_refresh', '3_refresh', '3_control']
    assert not probe.replay(tmp_path)['gate_passed']
    path = tmp_path/'0_control.request.json'
    request = json.loads(path.read_text())
    request['seed'] += 1
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match='request changed'):
        probe.replay(tmp_path)


def test_gate_requires_control_gain_and_numerical_comparator():
    forecasts = {str(i): {'forecasts': {a: dict(status='complete', mean=[v]*32)
                   for a, v in dict(base=2., control=1., refresh=.1, ridge=.2).items()}} for i in range(4)}
    truth = {str(i): [0.]*32 for i in range(4)}
    assert probe.score(forecasts, truth)['gate_passed']
    for record in forecasts.values():
        record['forecasts']['ridge']['mean'] = [0.]*32
    assert not probe.score(forecasts, truth)['gate_passed']
    forecasts['0']['forecasts']['refresh'] = dict(status='empty_support', mean=None)
    assert probe.score(forecasts, truth)['reason'] == 'empty_support'


def test_bad_schema_does_not_open_targets(tmp_path):
    class Block:
        root = tmp_path
        report = {}
        def request(self, *args):
            raw = fake()
            raw['choices'][0]['message']['content'] = '{"edits":[]}'
            return raw
    with pytest.raises(ValueError):
        probe.collect(Block(), probe.cases(), lambda: pytest.fail('endpoints opened'))
    assert not (tmp_path/'forecasts.json').exists()


def test_banked_qualification_replay():
    result = probe.replay(probe.ROOT)
    assert result['status'] == 'replay_valid' and not result['gate_passed']
    assert result['cost'] == pytest.approx(.01375145)
    assert result['means']['refresh'] > 10*result['means']['ridge']
