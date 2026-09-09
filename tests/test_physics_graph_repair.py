import hashlib
import json
import jsonschema
import pytest
from scripts import physics_graph_repair as probe
from tests.test_physgym_semantic_probe import public, response


def graph_response():
    raw = response()
    raw['choices'][0]['message']['content'] = json.dumps({'graphs': [{'nodes': [
        dict(op='constant', args=[], value=1., variable=None)]}]})
    return raw


def test_paired_body_schema_and_history():
    case = public()['457']
    case['history'][-1] = [{'a': 1.}, 12.345]
    a, b = [probe.body(case, ['1'], arm) for arm in ('control', 'refresh')]
    assert a['seed'] == b['seed'] and a['reasoning'] == b['reasoning']
    assert a['messages'][0] == b['messages'][0]
    assert '12.345' not in json.dumps(a) and '12.345' in json.dumps(b)
    assert a['response_format'] == b['response_format']
    jsonschema.validate(json.loads(graph_response()['choices'][0]['message']['content']),
                        a['response_format']['json_schema']['schema'])


def test_full_replay_seals_and_tamper(tmp_path, monkeypatch):
    data = {'public.json': public()['457'], 'forecasts.json': {'pools': {'semantic': ['1']}},
            'outcomes.json': [0.]*32}
    monkeypatch.setattr(probe.previous, 'load', lambda name: data[name])
    class Block:
        root = tmp_path
        report = dict(status='incomplete', calls=0, accepted_cost_usd=.002,
                      protocol_sha256=probe.PROTOCOL_SHA,
                      implementation_sha256={p: hashlib.sha256(probe.Path(p).read_bytes()).hexdigest() for p in probe.BINDINGS})
        def request(self, tag, body):
            self.report['calls'] += 1
            raw = graph_response()
            (tmp_path/(tag+'.request.json')).write_text(json.dumps(body))
            (tmp_path/(tag+'.response.json')).write_text(json.dumps(raw))
            return raw
    block = Block()
    def endpoints():
        assert block.report['calls'] == 2 and (tmp_path/'forecasts.json').exists()
        return data['outcomes.json']
    probe.collect(block, data['public.json'], ['1'], endpoints)
    (tmp_path/'result.json').write_text(json.dumps(block.report))
    assert probe.replay(tmp_path)['status'] == 'replay_valid'
    f = json.loads((tmp_path/'forecasts.json').read_text())
    assert f['forecasts']['control'] == f['forecasts']['refresh']
    path = tmp_path/'refresh.request.json'
    body = json.loads(path.read_text())
    body['seed'] += 1
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='request changed'):
        probe.replay(tmp_path)


def test_graph_semantic_failure_never_reads_targets(tmp_path):
    class Block:
        root = tmp_path
        report = {}
        def request(self, *args):
            raw = graph_response()
            raw['choices'][0]['message']['content'] = json.dumps({'graphs': [{'nodes': [
                dict(op='neg', args=[0], value=None, variable=None)]}]})
            return raw
    with pytest.raises(ValueError, match='reference'):
        probe.collect(Block(), public()['457'], ['1'], lambda: pytest.fail('targets opened'))
    assert not (tmp_path/'forecasts.json').exists()
