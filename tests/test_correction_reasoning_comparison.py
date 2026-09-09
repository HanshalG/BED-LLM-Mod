import hashlib
import json
import pytest
from scripts import correction_reasoning_comparison as probe
from tests.test_compositional_edit_qualification import fake


def test_only_effort_differs():
    case = probe.cases()['0']
    a, b = [probe.body(case, 'refresh', 0, e) for e in ('medium', 'high')]
    assert a['reasoning']['effort'] == 'medium' and b['reasoning']['effort'] == 'high'
    b['reasoning']['effort'] = 'medium'
    assert a == b
    c = probe.body(case, 'control', 0, 'high')
    assert len(json.loads(c['messages'][1]['content'])['history']) == 3
    assert len(json.loads(a['messages'][1]['content'])['history']) == 6
    assert not set(probe.SOURCES) & set(probe.parent.SOURCES)


def test_full_sixteen_call_sealed_replay(tmp_path):
    class Block:
        root = tmp_path
        report = dict(status='incomplete', calls=0, accepted_cost_usd=.016,
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
        assert block.report['calls'] == 16 and (tmp_path/'forecasts.json').exists()
        return probe.outcomes(public)
    probe.collect(block, public, endpoints)
    (tmp_path/'result.json').write_text(json.dumps(block.report))
    assert block.tags[:4] == ['0_medium_control', '0_medium_refresh', '0_high_control', '0_high_refresh']
    assert block.tags[4:8] == ['1_high_refresh', '1_high_control', '1_medium_refresh', '1_medium_control']
    report = probe.replay(tmp_path)
    assert all(not q['gate_passed'] for q in report['qualifications'].values())
    assert report['qualifications']['high']['means'] == report['qualifications']['medium']['means']
    path = tmp_path/'0_high_control.request.json'
    body = json.loads(path.read_text())
    body['reasoning']['effort'] = 'medium'
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='request changed'):
        probe.replay(tmp_path)


def test_bad_response_before_targets(tmp_path):
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


def test_banked_reasoning_comparison_replay():
    report = probe.replay(probe.ROOT)
    assert report['status'] == 'replay_valid'
    assert report['cost'] == pytest.approx(.0443091)
    for result in report['qualifications'].values():
        assert not result['gate_passed'] and result['wins'] == 0
        assert result['means']['refresh'] > result['means']['ridge']
