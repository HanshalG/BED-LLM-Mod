"""User-requested paired high-reasoning diagnostic on banked observed histories."""
import argparse
import copy
from datetime import datetime
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from zoneinfo import ZoneInfo

from scripts import deepcoder_constrained_gate as base
from scripts.deepcoder_proposal_gate import save, money
from scripts.deepcoder_opportunity import load_dsl
from scripts.openrouter_daily_budget import read_live_credits, budget_status, require_budget
from environments.program_induction import constrained_request as cr, prediction as pred

PARENT = Path('results/nonmyopic/deepcoder_constrained_gate_20260909')
PARENT_SHA = '371918b6a0f8a7170bcacf161aba6b973049df203ac66e746b9682d255ed04dd'
ROOT = Path('results/nonmyopic/deepcoder_high_reasoning_20260909')
LEDGER = base.LEDGER
PROTOCOL = Path('results/nonmyopic/DEEPCODER_HIGH_REASONING_PROTOCOL_20260909.md')
RESERVE, CAP = Decimal('.015'), Decimal('.25')


def bumped(body):
    if body['reasoning'] != {'enabled': False, 'exclude': True} or body['max_tokens'] != 4096:
        raise ValueError('unexpected parent reasoning/token setting')
    changed = copy.deepcopy(body)
    changed['reasoning'] = {'enabled': True, 'effort': 'high', 'exclude': True}
    changed['max_tokens'] = 16384
    if len(pred.canonical(changed).encode()) > 32768:
        raise ValueError('complete request byte cap exceeded')
    return changed


def validate_response(raw):
    usage = raw['usage']
    reasoning = usage['completion_tokens_details']['reasoning_tokens']
    completion = usage['completion_tokens']
    if (type(reasoning) is not int or type(completion) is not int
            or not 0 < reasoning <= completion <= 16384
            or type(usage['prompt_tokens']) is not int or not 0 <= usage['prompt_tokens'] <= 65536
            or money(usage['cost']) > RESERVE):
        raise ValueError('reasoning/token/cost validation failed')
    if raw['model'] != cr.MODEL or raw['provider'] != 'OpenInference':
        raise ValueError('model/provider changed')
    choice, = raw['choices']
    if choice['finish_reason'] != 'stop' or type(choice['message']['content']) is not str:
        raise ValueError('incomplete response')
    return choice['message']['content']


def authorize(ledger, exposure):
    live = read_live_credits()
    status = budget_status(ledger, total_usage_usd=live['total_usage_usd'])
    ledger['recorded_actual_spend_usd'] = status['spent_today_usd']
    save(LEDGER, ledger)
    require_budget(ledger, projected_cost_usd=float(exposure), total_usage_usd=live['total_usage_usd'])
    if money(live['balance_usd']) < exposure:
        raise RuntimeError('balance gate failed')


def route():
    endpoint = base.route()
    if 'reasoning_effort' not in endpoint['supported_parameters'] or endpoint['max_completion_tokens'] < 16384:
        raise ValueError('high-reasoning route unsupported')
    return endpoint


def post(body):
    req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
        data=pred.canonical(body).encode(), headers={'Authorization': 'Bearer '+os.environ['OPENROUTER_API_KEY'],
                                                   'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def run():
    if ROOT.exists():
        raise RuntimeError('probe already opened; no retry')
    data = (PARENT/'forecasts.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != PARENT_SHA:
        raise ValueError('banked panel identity mismatch')
    panel = json.loads(data)
    base.shared.validate_panel(panel)
    ledger = json.loads(LEDGER.read_text())
    auth = ledger.get('high_reasoning_authorization', {})
    if auth != {'approved': True, 'consumed': False, 'cap_usd': .25} or ledger.get('pending_reservations'):
        raise RuntimeError('fresh reasoning authorization required')
    authorize(ledger, CAP)
    route()
    dsl = load_dsl()
    ROOT.mkdir()
    opening = money(ledger['recorded_actual_spend_usd'])
    accepted, uncertain = Decimal(0), Decimal(0)
    pending, dispatched = False, False
    report = dict(status='incomplete', calls=0, results={}, endpoints_opened=False, depth_authorized=False,
                  parent_sha256=PARENT_SHA, protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        ledger['high_reasoning_authorization']['consumed'] = True
        save(LEDGER, ledger)
        for i, case in panel['cases'].items():
            report['results'][i] = {}
            for arm in (pred.ARMS[:2] if int(i)%2 == 0 else pred.ARMS[1::-1]):
                tag = i+'_'+arm
                original = json.loads((PARENT/(tag+'.request.json')).read_text())
                expected = cr.request(dsl, case['history'], 14100000+int(i), history_blind=arm=='history_blind')
                if original != expected:
                    raise ValueError('paired original request identity failed')
                body = bumped(original)
                save(ROOT/(tag+'.request.json'), body, exclusive=True)
                save(ROOT/(tag+'.route.json'), route(), exclusive=True)
                if accepted+RESERVE > CAP:
                    raise RuntimeError('block cap exceeded')
                ledger = json.loads(LEDGER.read_text())
                if ledger.get('pending_reservations'):
                    raise RuntimeError('reservation conflict')
                authorize(ledger, RESERVE)
                ledger['pending_reservations'] = {'high_reasoning:'+tag: float(RESERVE)}
                save(LEDGER, ledger)
                pending, dispatched = True, False
                authorize(ledger, RESERVE)
                report.update(current=tag, calls=report['calls']+1)
                save(ROOT/'result.json', report)
                dispatched = True
                raw = post(body)
                save(ROOT/(tag+'.response.json'), raw, exclusive=True)
                accepted += money(raw['usage']['cost'])
                pending = False
                ledger['pending_reservations'] = {}
                ledger['recorded_actual_spend_usd'] = float(max(money(ledger['recorded_actual_spend_usd']), opening+accepted))
                save(LEDGER, ledger)
                text = validate_response(raw)
                pool = base.candidates(dsl, text)
                matches = []
                for name, evaluate in pool:
                    ys = [evaluate(h['inputs']) for h in case['history']]
                    matches.append(dict(program=name, observed_input_predictions=ys,
                                        compatible=ys==[h['output'] for h in case['history']]))
                report['results'][i][arm] = dict(programs=matches, reasoning_tokens=raw['usage']['completion_tokens_details']['reasoning_tokens'],
                    raw_program_count=len(json.loads(text)['programs']), cost_usd=raw['usage']['cost'])
                report['accepted_cost_usd'] = float(accepted)
                save(ROOT/'result.json', report)
        report['coverage'] = {a: sum(any(p['compatible'] for p in r[a]['programs']) for r in report['results'].values()) for a in pred.ARMS[:2]}
        report['status'] = 'paired_observed_history_diagnostic_complete'
    except Exception as exc:
        report.update(status='failed_closed', error_type=type(exc).__name__)
    finally:
        if pending:
            uncertain = RESERVE if dispatched else Decimal(0)
            ledger['pending_reservations'] = {'high_reasoning:uncertain': float(RESERVE)} if dispatched else {}
        ledger['recorded_actual_spend_usd'] = float(max(money(ledger['recorded_actual_spend_usd']), opening+accepted+uncertain))
        save(LEDGER, ledger)
        report.update(accepted_cost_usd=float(accepted), uncertain_exposure_usd=float(uncertain),
                      completed_at=datetime.now(ZoneInfo('Europe/London')).isoformat())
        save(ROOT/'result.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.parse_args()
    with LEDGER.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run()
        print(pred.canonical({k:v for k,v in result.items() if k!='results'}))
