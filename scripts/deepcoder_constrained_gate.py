"""One approved, fresh constrained-interface gate; old run is never reopened."""
import argparse
from datetime import datetime
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from zoneinfo import ZoneInfo

from environments.program_induction import constrained, constrained_request as cr
from environments.program_induction import prediction as pred
from environments.program_induction.synthesis import synthesize, evaluate_expression
from scripts.deepcoder_opportunity import load_dsl, sample_program, sample_input
from scripts import deepcoder_proposal_gate as shared
from scripts.openrouter_daily_budget import read_live_credits, require_budget, budget_status

ROOT = Path('results/nonmyopic/deepcoder_constrained_gate_20260909')
PROTOCOL = Path('results/nonmyopic/DEEPCODER_CONSTRAINED_GATE_PROTOCOL_20260909.md')
LEDGER = Path('results/nonmyopic/openrouter_daily_budget/2026-09-08.json')
AUTH = 'deepcoder_constrained_gate_20260909'
CAP = Decimal('.25')
RESERVE = cr.RESERVATION_USD
save = shared.save
money = shared.money


def record_live(ledger, live):
    status = budget_status(ledger, total_usage_usd=live['total_usage_usd'])
    ledger['recorded_actual_spend_usd'] = status['spent_today_usd']
    save(LEDGER, ledger)


def route():
    url = 'https://openrouter.ai/api/v1/models/'+cr.MODEL+'/endpoints'
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer '+os.environ['OPENROUTER_API_KEY']})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)['data']
    endpoint, = [r for r in data['endpoints'] if r['tag'] == cr.PROVIDER]
    cr.advertised_route(endpoint)
    return endpoint


def cases(dsl):
    result = {}
    for i in range(8):
        program = sample_program(dsl, 12100000+i)
        inputs = [sample_input(13100000+i*100+j) for j in range(34)]
        history = []
        for x in inputs[:2]:
            state = program.run(x)
            history.append(dict(inputs=x, output=None if state is None else state.get_output()))
        result[str(i)] = dict(history=history, targets=inputs[2:])
    return result


def outcomes(dsl, public):
    truth = {}
    for i, case in public.items():
        program = sample_program(dsl, 12100000+int(i))
        outputs = []
        for x in case['targets']:
            state = program.run(x)
            outputs.append(None if state is None else state.get_output())
        truth[i] = dict(target_inputs=case['targets'], outputs=outputs)
    return truth


def candidates(dsl, text):
    def evaluator(program):
        def run(x):
            state = program.run(x)
            return None if state is None else state.get_output()
        return run
    return [(str(p), evaluator(p)) for p in constrained.decode(dsl, text)]


def execute(body):
    request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
        data=pred.canonical(body).encode(), headers={'Authorization': 'Bearer '+os.environ['OPENROUTER_API_KEY'],
                                                   'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def run():
    if ROOT.exists():
        raise RuntimeError('constrained gate already opened; no rerun')
    ledger = json.loads(LEDGER.read_text())
    auth = ledger.get('constrained_gate_authorization', {})
    if (auth.get('block') != AUTH or auth.get('approved') is not True
            or auth.get('consumed') is not False or money(auth.get('cap_usd')) != CAP):
        raise RuntimeError('exact fresh authorization required')
    live = read_live_credits()
    record_live(ledger, live)
    require_budget(ledger, projected_cost_usd=float(CAP), total_usage_usd=live['total_usage_usd'])
    if money(live['balance_usd']) < CAP or ledger.get('pending_reservations'):
        raise RuntimeError('balance or pending reservation gate failed')
    endpoint = route()
    dsl = load_dsl()
    ROOT.mkdir()
    base = money(ledger['recorded_actual_spend_usd'])
    accepted, uncertain = Decimal(0), Decimal(0)
    pending, dispatched = False, False
    report = dict(status='incomplete', phase='prepare', calls=0, endpoints_opened=False, depth_authorized=False,
                  implementation_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                      str(PROTOCOL), __file__, 'scripts/deepcoder_proposal_gate.py',
                      'scripts/deepcoder_opportunity.py', 'environments/program_induction/constrained.py',
                      'environments/program_induction/constrained_request.py',
                      'environments/program_induction/synthesis.py', 'environments/program_induction/prediction.py')})
    try:
        ledger['constrained_gate_authorization']['consumed'] = True
        save(LEDGER, ledger)
        save(ROOT/'route.json', endpoint, exclusive=True)
        public = cases(dsl)
        save(ROOT/'public.json', public, exclusive=True)
        forecasts, work = {i: {} for i in public}, {}
        report['phase'] = 'symbolic_search'
        save(ROOT/'result.json', report)
        for i, case in public.items():
            pool, work[i] = [], []
            for r in range(8):
                found = synthesize(dsl, [(h['inputs'], h['output']) for h in case['history']],
                                   order_seed=15100000+int(i)*10+r)
                expr = found.pop('expression')
                work[i].append(found)
                if expr is not None:
                    pool.append((expr.expression(), lambda x, e=expr: evaluate_expression(e, x)))
            forecasts[i]['symbolic_search'] = shared.pool_forecast(pool, case)
        save(ROOT/'search.json', work, exclusive=True)
        for i, case in public.items():
            order = pred.ARMS[:2] if int(i) % 2 == 0 else pred.ARMS[1::-1]
            for arm in order:
                tag = f'{i}_{arm}'
                report.update(phase='authorize', current=tag)
                body = cr.request(dsl, case['history'], 14100000+int(i), history_blind=arm == 'history_blind')
                save(ROOT/(tag+'.route.json'), route(), exclusive=True)
                save(ROOT/(tag+'.request.json'), body, exclusive=True)
                if accepted+uncertain+RESERVE > CAP:
                    raise RuntimeError('block cap exceeded')
                ledger = json.loads(LEDGER.read_text())
                live = read_live_credits()
                record_live(ledger, live)
                require_budget(ledger, projected_cost_usd=float(RESERVE), total_usage_usd=live['total_usage_usd'])
                if money(live['balance_usd']) < RESERVE or ledger.get('pending_reservations'):
                    raise RuntimeError('account reservation conflict')
                ledger['pending_reservations'] = {AUTH+':'+tag: float(RESERVE)}
                save(LEDGER, ledger)
                pending, dispatched = True, False
                live = read_live_credits()
                record_live(ledger, live)
                require_budget(ledger, projected_cost_usd=float(RESERVE), total_usage_usd=live['total_usage_usd'])
                if money(live['balance_usd']) < RESERVE:
                    raise RuntimeError('balance decreased before dispatch')
                report.update(phase='http_attempt', calls=report['calls']+1)
                save(ROOT/'result.json', report)
                dispatched = True
                raw = execute(body)
                save(ROOT/(tag+'.response.json'), raw, exclusive=True)
                cost = money(raw['usage']['cost'])
                accepted += cost
                pending = False
                ledger['pending_reservations'] = {}
                ledger['recorded_actual_spend_usd'] = float(max(money(ledger['recorded_actual_spend_usd']), base+accepted))
                save(LEDGER, ledger)
                report.update(phase='validate_response', cost_usd=float(accepted))
                save(ROOT/'result.json', report)
                text, _ = shared.response_content(raw)
                if raw.get('model') != cr.MODEL or raw.get('provider') != 'OpenInference':
                    raise ValueError('response model/provider changed')
                forecasts[i][arm] = shared.pool_forecast(candidates(dsl, text), case)
        panel = dict(version=1, cases=public, forecasts=forecasts)
        shared.validate_panel(panel)
        save(ROOT/'forecasts.json', panel, exclusive=True)
        sha = hashlib.sha256((ROOT/'forecasts.json').read_bytes()).hexdigest()
        report.update(phase='sealed_evaluation', forecast_sha256=sha)
        save(ROOT/'result.json', report)

        def load_outcomes():
            report['endpoints_opened'] = True
            save(ROOT/'result.json', report)
            truth = outcomes(dsl, public)
            save(ROOT/'outcomes.json', truth, exclusive=True)
            return truth

        report.update(shared.evaluate_sealed(ROOT/'forecasts.json', sha, load_outcomes))
    except Exception as exc:
        report.update(status='failed_closed', error_type=type(exc).__name__)
    finally:
        if pending:
            if dispatched:
                uncertain = RESERVE
                ledger['pending_reservations'] = {AUTH+':uncertain': float(RESERVE)}
            else:
                ledger['pending_reservations'] = {}
        ledger['recorded_actual_spend_usd'] = float(max(money(ledger['recorded_actual_spend_usd']), base+accepted+uncertain))
        save(LEDGER, ledger)
        report.update(accepted_cost_usd=float(accepted), uncertain_exposure_usd=float(uncertain),
                      cost_usd=float(accepted+uncertain), completed_at=datetime.now(ZoneInfo('Europe/London')).isoformat())
        save(ROOT/'result.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.parse_args()
    with LEDGER.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(pred.canonical(run()))
