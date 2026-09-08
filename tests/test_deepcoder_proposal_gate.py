import hashlib
import json
from decimal import Decimal

import pytest

from scripts import deepcoder_proposal_gate as g


def panel():
    case = dict(history=[dict(inputs=[[1], [2]], output=1)]*2, targets=[[[3], [4]]]*32)
    f = g.pool_forecast([('one', lambda _: 1)], case)
    return dict(version=1, cases={str(i): case for i in range(8)},
                forecasts={str(i): {a: f for a in g.ARMS} for i in range(8)})


def write_panel(tmp_path, p):
    path = tmp_path/'sealed.json'
    g.save(path, p, exclusive=True)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def outcomes(p):
    return {i: dict(target_inputs=c['targets'], outputs=[1]*32) for i, c in p['cases'].items()}


def bomb():
    raise AssertionError('endpoint loader opened')


def test_same_predictions_do_not_pass(tmp_path):
    p = panel()
    path, sha = write_panel(tmp_path, p)
    result = g.evaluate_sealed(path, sha, lambda: outcomes(p))
    assert result['status'] == 'proposal_quality_null'
    assert all(v == 0 for v in result['means'].values())
    assert not result['depth_authorized']


def test_abstentions_not_dropped_or_faked_as_distributions(tmp_path):
    p = panel()
    for arms in p['forecasts'].values():
        arms['history_blind'] = arms['symbolic_search'] = None
    path, sha = write_panel(tmp_path, p)
    result = g.evaluate_sealed(path, sha, lambda: outcomes(p))
    assert result['status'] == 'proposal_screen_pass'
    assert result['means']['history_blind'] == 1
    assert result['coverage']['history_blind'] == 0
    assert result['rows']['0']['history_blind']['abstained']
    assert not result['depth_authorized']


def test_coverage_null_never_opens_targets(tmp_path):
    p = panel()
    for i in range(3):
        p['forecasts'][str(i)]['history_aware'] = None
    path, sha = write_panel(tmp_path, p)
    assert g.evaluate_sealed(path, sha, bomb)['status'] == 'proposal_coverage_null'


@pytest.mark.parametrize('change', ['hash', 'missing_arm', 'target'])
def test_bad_seal_or_controls_fail_before_targets(tmp_path, change):
    p = panel()
    if change == 'missing_arm':
        del p['forecasts']['0']['symbolic_search']
    if change == 'target':
        p['cases']['0'] = dict(history=p['cases']['0']['history'], targets=[[[8], [9]]]*32)
    path, sha = write_panel(tmp_path, p)
    with pytest.raises(ValueError):
        g.evaluate_sealed(path, '0'*64 if change == 'hash' else sha, bomb)


def test_payload_has_no_reasoning_fallback_or_uncapped_provider():
    p = g.payload([dict(role='user', content='test')], 42)
    assert p['reasoning']['enabled'] is False
    assert p['provider']['allow_fallbacks'] is False
    assert p['provider']['max_price'] == dict(prompt=.1, completion=.2)
    assert p['max_tokens'] == 4096
    assert Decimal(65536)*Decimal('.0000001')+Decimal(4096)*Decimal('.0000002') < g.RESERVE
    assert g.RESERVE*16 <= g.CAP
    with pytest.raises(ValueError):
        g.payload([dict(role='user', content='x'*16001)], 42)


def response():
    return dict(usage=dict(cost=.001, prompt_tokens=100, completion_tokens=100,
                           completion_tokens_details={'reasoning_tokens': 0}),
                choices=[dict(finish_reason='stop', message={'content': '{}'})])


@pytest.mark.parametrize('kind', ['cost', 'thinking', 'truncated', 'missing_cost'])
def test_invalid_serving_fails(kind):
    r = response()
    if kind == 'cost':
        r['usage']['cost'] = .016
    if kind == 'thinking':
        r['usage']['completion_tokens_details']['reasoning_tokens'] = 1
    if kind == 'truncated':
        r['choices'][0]['finish_reason'] = 'length'
    if kind == 'missing_cost':
        del r['usage']['cost']
    with pytest.raises((ValueError, KeyError)):
        g.response_content(r)


def test_unknown_pool_errors_not_silently_abstained():
    with pytest.raises(ValueError):
        g.pool_forecast([('bad', lambda _: True)], panel()['cases']['0'])


def test_one_attempt_failure_keeps_reservation_and_no_endpoints(tmp_path, monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    root = tmp_path/'gate'
    ledger = tmp_path/'ledger.json'
    g.save(ledger, dict(date=datetime.now(ZoneInfo('Europe/London')).date().isoformat(),
                       timezone='Europe/London', daily_cap_usd=5,
                       opening_total_usage_usd=10, recorded_actual_spend_usd=0))
    monkeypatch.setattr(g, 'ROOT', root)
    monkeypatch.setattr(g, 'LEDGER', ledger)
    protocol = tmp_path/'protocol.md'
    protocol.write_text('frozen test')
    monkeypatch.setattr(g, 'PROTOCOL', protocol)
    monkeypatch.setattr(g, 'load_dsl', lambda: None)
    monkeypatch.setattr(g, 'read_catalog', lambda: {})
    reads = []

    def credits():
        reads.append('credits')
        return dict(total_usage_usd=10, balance_usd=20)

    monkeypatch.setattr(g, 'read_live_credits', credits)
    monkeypatch.setattr(g, 'public_cases', lambda _: panel()['cases'])
    monkeypatch.setattr(g, 'synthesize', lambda *a, **k: dict(expression=None))
    monkeypatch.setattr(g, 'messages', lambda *a, **k: [dict(role='user', content='test')])
    monkeypatch.setattr(g, 'private_outcomes', lambda *a: bomb())
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fixture-not-a-key')
    attempts = []

    def failed_http(*a, **k):
        assert len(reads) == 3
        assert json.loads(ledger.read_text())['pending_reservations']
        attempts.append(1)
        raise TimeoutError()

    monkeypatch.setattr(g.urllib.request, 'urlopen', failed_http)
    result = g.run()
    assert result['status'] == 'failed_closed'
    assert result['cost_usd'] == .015
    assert attempts == [1]
    assert not (root/'outcomes.json').exists()
    assert json.loads(ledger.read_text())['pending_reservations']
    with pytest.raises(RuntimeError, match='already opened'):
        g.run()


def test_full_runner_seals_before_outcomes_and_reconciles_all_calls(tmp_path, monkeypatch):
    import io
    from datetime import datetime
    from zoneinfo import ZoneInfo
    root, ledger, protocol = tmp_path/'gate', tmp_path/'ledger.json', tmp_path/'protocol.md'
    protocol.write_text('frozen test')
    g.save(ledger, dict(date=datetime.now(ZoneInfo('Europe/London')).date().isoformat(),
                       timezone='Europe/London', daily_cap_usd=5,
                       opening_total_usage_usd=10, recorded_actual_spend_usd=0))
    for key, value in [('ROOT', root), ('LEDGER', ledger), ('PROTOCOL', protocol)]:
        monkeypatch.setattr(g, key, value)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fixture-not-a-key')
    monkeypatch.setattr(g, 'load_dsl', lambda: None)
    monkeypatch.setattr(g, 'read_catalog', lambda: {})
    monkeypatch.setattr(g, 'read_live_credits', lambda: dict(total_usage_usd=10, balance_usd=20))
    monkeypatch.setattr(g, 'public_cases', lambda _: panel()['cases'])
    monkeypatch.setattr(g, 'synthesize', lambda *a, **k: dict(expression=None))
    monkeypatch.setattr(g, 'messages', lambda *a, **k: [dict(role='user', content='test')])
    monkeypatch.setattr(g, 'model_candidates', lambda *a: [('one', lambda _: 1)])
    attempts = []

    def http(*a, **k):
        assert json.loads(ledger.read_text())['pending_reservations']
        assert not (root/'outcomes.json').exists()
        attempts.append(1)
        return io.StringIO(json.dumps(response()))

    def labels(*a):
        assert len(attempts) == 16
        assert (root/'forecasts.json').exists()
        return outcomes(panel())

    monkeypatch.setattr(g.urllib.request, 'urlopen', http)
    monkeypatch.setattr(g, 'private_outcomes', labels)
    result = g.run()
    assert result['status'] == 'proposal_quality_null'
    assert result['calls'] == 16 and result['cost_usd'] == .016
    stored = json.loads(ledger.read_text())
    assert stored['recorded_actual_spend_usd'] == .016
    assert stored['pending_reservations'] == {}


def test_shuffled_source_search_replays_and_honors_cap():
    from scripts.deepcoder_opportunity import load_dsl
    from environments.program_induction.synthesis import synthesize, evaluate_expression
    history = [([[1, 2], [3]], 2), ([[3, 4], [0]], 4)]
    first = synthesize(load_dsl(), history, order_seed=71)
    second = synthesize(load_dsl(), history, order_seed=71)
    assert first['status'] == second['status'] == 'compatible_expression_found'
    assert first['expression'].expression() == second['expression'].expression()
    assert first['operation_attempts'] <= 2048
    assert all(evaluate_expression(first['expression'], x) == y for x, y in history)
