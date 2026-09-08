from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pytest
from jsonschema import Draft202012Validator

from scripts import deepcoder_luna_transition as g


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root, ledger, protocol = tmp_path/'gate', tmp_path/'ledger.json', tmp_path/'protocol.md'
    protocol.write_text('fixture prospective protocol')
    g.save(ledger, dict(date=datetime.now(ZoneInfo('Europe/London')).date().isoformat(),
                       timezone='Europe/London', daily_cap_usd=5, opening_total_usage_usd=10,
                       recorded_actual_spend_usd=.0402, pending_reservations=dict(g.CARRY),
                       luna_transition_authorization=dict(block=g.AUTH, approved=True, consumed=False, cap_usd=.6)))
    monkeypatch.setattr(g, 'ROOT', root)
    monkeypatch.setattr(g, 'LEDGER', ledger)
    monkeypatch.setattr(g, 'PROTOCOL', protocol)
    monkeypatch.setattr(g, 'route', lambda: {'fixture': True})
    public = {str(i): dict(history=[{'inputs': [[1], [2]], 'output': 1}]*2,
                           query=[[1],[2]], targets=[[[3], [4]]]*32) for i in range(4)}
    monkeypatch.setattr(g, 'cases', lambda _: public)
    def observe(dsl, key, inputs):
        assert (root/(key+'.before.json')).exists()
        assert not (root/'outcomes.json').exists()
        return dict(inputs=inputs, output=1)
    monkeypatch.setattr(g, 'observe', observe)
    reads, posts = [], []

    def credits():
        reads.append(1)
        return dict(total_usage_usd=10.0002, balance_usd=20)

    def execute(body):
        assert not (root/'outcomes.json').exists()
        assert json.loads(ledger.read_text())['pending_reservations']
        tag = json.loads((root/'result.json').read_text())['current']
        shown = json.loads(body['messages'][1]['content'])['history']
        assert len(shown) == (3 if tag.endswith('_regenerated') else 2)
        assert body['seed'] == 25100000+10*int(tag.split('_')[0])+(0 if tag.endswith('_initial') else 1)
        parsed = {'programs': [{'statement': 'x2 = Reverse x0', 'next': {
            'statement': 'x3 = Head x2', 'next': None}}]}
        Draft202012Validator(body['response_format']['json_schema']['schema']).validate(parsed)
        assert body['reasoning']['effort'] == 'medium'
        posts.append(body)
        return dict(model='openai/gpt-5.6-luna', provider='OpenAI', usage=dict(cost=.001,
                    prompt_tokens=1000, completion_tokens=100, completion_tokens_details={'reasoning_tokens': 50}),
                    choices=[dict(finish_reason='stop', message={'content': json.dumps(parsed)})])

    def outcomes(dsl, cases):
        assert len(posts) == 12
        assert (root/'forecasts.json').exists()
        return {i: dict(target_inputs=c['targets'], outputs=[3]*32) for i, c in cases.items()}

    monkeypatch.setattr(g, 'read_live_credits', credits)
    monkeypatch.setattr(g, 'execute', execute)
    monkeypatch.setattr(g, 'outcomes', outcomes)
    return root, ledger, reads, posts


def test_full_constrained_runner(fixture):
    root, ledger, reads, posts = fixture
    result = g.run()
    assert result['status'] == 'transition_screen_complete'
    assert result['endpoints_opened'] and not result['depth_authorized']
    assert result['calls'] == len(posts) == 12 and len(reads) == 25
    assert result['accepted_cost_usd'] == .012 and result['uncertain_exposure_usd'] == 0
    stored = json.loads(ledger.read_text())
    assert stored['recorded_actual_spend_usd'] == .0522
    assert stored['pending_reservations'] == g.CARRY
    assert stored['luna_transition_authorization']['consumed']
    assert set(result['mean_brier']) == set(g.screen.ARMS)
    with pytest.raises(RuntimeError, match='already opened'):
        g.run()


def test_timeout_no_retry_and_no_labels(fixture, monkeypatch):
    root, ledger, _, _ = fixture
    calls = []

    def timeout(body):
        calls.append(1)
        raise TimeoutError()

    monkeypatch.setattr(g, 'execute', timeout)
    r = g.run()
    assert calls == [1] and r['status'] == 'failed_closed'
    assert r['accepted_cost_usd'] == 0 and r['uncertain_exposure_usd'] == .04
    assert not (root/'outcomes.json').exists()
    assert json.loads(ledger.read_text())['pending_reservations']


def test_account_race_prevents_dispatch_without_uncertain_charge(fixture, monkeypatch):
    root, ledger, _, posts = fixture
    reads = []

    def credits():
        reads.append(1)
        return dict(total_usage_usd=15 if len(reads) == 3 else 10.0002, balance_usd=20)

    monkeypatch.setattr(g, 'read_live_credits', credits)
    r = g.run()
    assert r['status'] == 'failed_closed' and r['calls'] == 0 and not posts
    assert r['uncertain_exposure_usd'] == r['accepted_cost_usd'] == 0
    assert json.loads(ledger.read_text())['pending_reservations'] == g.CARRY
    assert json.loads(ledger.read_text())['recorded_actual_spend_usd'] == 5
    assert not (root/'outcomes.json').exists()


def test_without_exact_approval_no_directory_or_model_calls(fixture):
    root, ledger, _, posts = fixture
    stored = json.loads(ledger.read_text())
    stored['luna_transition_authorization']['approved'] = False
    g.save(ledger, stored)
    with pytest.raises(RuntimeError, match='authorization'):
        g.run()
    assert not root.exists() and not posts


def test_source_violation_stops_on_first_paid_response(fixture, monkeypatch):
    root, ledger, _, _ = fixture
    execute = g.execute

    def wrong(body):
        r = execute(body)
        data = json.loads(r['choices'][0]['message']['content'])
        data['programs'][0]['next']['statement'] = 'x3 = Head x1'
        r['choices'][0]['message']['content'] = json.dumps(data)
        return r

    monkeypatch.setattr(g, 'execute', wrong)
    r = g.run()
    assert r['status'] == 'failed_closed' and r['calls'] == 1
    assert r['accepted_cost_usd'] == .001 and r['phase'] == 'validate_response'
    assert not (root/'forecasts.json').exists() and not (root/'outcomes.json').exists()
    assert json.loads(ledger.read_text())['pending_reservations'] == g.CARRY
