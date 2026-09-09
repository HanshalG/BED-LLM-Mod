import hashlib
import json
import pytest
from scripts import physics_paired_repair as probe
from tests.test_physgym_semantic_probe import public, response


def test_paired_payload_no_new_observation_leak():
    case = public()['457']
    case['history'][-1] = [{'a':1.}, 12.345]
    a, b = [probe.body(case, ['1'], arm) for arm in ('control', 'refresh')]
    assert a['seed'] == b['seed']
    assert a['reasoning'] == b['reasoning']
    assert '12.345' not in json.dumps(a)
    assert '12.345' in json.dumps(b)
    assert 'targets' not in json.loads(a['messages'][1]['content'])
    x, y = [json.loads(v['messages'][1]['content']) for v in (a, b)]
    assert x['public_domain'] == y['public_domain']
    assert len(x['history']) == 3 and len(y['history']) == 4


def test_full_replay_and_sealing(tmp_path, monkeypatch):
    case = public()['457']
    data = {'public.json':case, 'forecasts.json':{'pools':{'semantic':['1']}}, 'outcomes.json':[0.]*32}
    monkeypatch.setattr(probe, 'load', lambda name:data[name])

    class Block:
        root = tmp_path
        report = dict(status='incomplete', calls=0, accepted_cost_usd=.002,
                      protocol_sha256=probe.PROTOCOL_SHA,
                      implementation_sha256={p:hashlib.sha256(probe.Path(p).read_bytes()).hexdigest() for p in probe.BINDINGS})
        def request(self, tag, body):
            self.report['calls'] += 1
            raw = response()
            (tmp_path/(tag+'.request.json')).write_text(json.dumps(body))
            (tmp_path/(tag+'.response.json')).write_text(json.dumps(raw))
            return raw

    block = Block()
    def endpoints():
        assert block.report['calls'] == 2
        assert (tmp_path/'forecasts.json').exists()
        return data['outcomes.json']
    probe.collect(block, case, ['1'], endpoints)
    (tmp_path/'result.json').write_text(json.dumps(block.report))
    assert probe.replay(tmp_path)['status'] == 'replay_valid'
    f = json.loads((tmp_path/'forecasts.json').read_text())
    assert f['forecasts']['control'] == f['forecasts']['refresh']
    path = tmp_path/'control.request.json'
    body = json.loads(path.read_text())
    body['seed'] += 1
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='request changed'):
        probe.replay(tmp_path)


def test_schema_failure_does_not_read_targets(tmp_path):
    class Block:
        root = tmp_path
        report = {}
        def request(self, *args):
            raw = response()
            raw['choices'][0]['message']['content'] = '{"expressions":[]}'
            return raw
    with pytest.raises(ValueError):
        probe.collect(Block(), public()['457'], ['1'], lambda:pytest.fail('labels opened'))
    assert not (tmp_path/'forecasts.json').exists()


def test_signal_null_and_empty():
    forecasts = {a:dict(status='complete', mean=[v]*32)
                 for a, v in dict(initial=2., control=1., refresh=.5).items()}
    assert probe.score(forecasts, [0.]*32)['descriptive_repair_signal']
    forecasts['refresh'] = forecasts['control']
    assert not probe.score(forecasts, [0.]*32)['descriptive_repair_signal']
    forecasts['refresh'] = dict(status='empty_support', mean=None)
    result = probe.score(forecasts, [0.]*32)
    assert not result['descriptive_repair_signal'] and not result['depth_authorized']
    assert result['losses']['refresh'] is None
