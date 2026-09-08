"""One-shot, prospectively frozen proposal-only gate; no depth authorization."""
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

from environments.program_induction import prediction as pred
from environments.program_induction.proposals import messages, parse_proposals
from environments.program_induction.synthesis import synthesize, evaluate_expression
from scripts.deepcoder_opportunity import load_dsl, sample_program, sample_input
from scripts.openrouter_daily_budget import read_live_credits, require_budget

MODEL = 'deepseek/deepseek-v4-flash-0731'
ROOT = Path('results/nonmyopic/deepcoder_proposal_gate_20260908')
LEDGER = Path('results/nonmyopic/openrouter_daily_budget/2026-09-08.json')
PROTOCOL = Path('results/nonmyopic/DEEPCODER_PROPOSAL_GATE_PROTOCOL_20260908.md')
RESERVE = Decimal('.015')
CAP = Decimal('.25')
ARMS = pred.ARMS


def save(path, value, *, exclusive=False):
    data = pred.canonical(value)+'\n'
    if exclusive:
        with Path(path).open('x') as stream:
            stream.write(data)
    else:
        temp = Path(str(path)+'.tmp')
        temp.write_text(data)
        temp.replace(path)


def money(value):
    if isinstance(value, bool) or not isinstance(value, (str, float, int)):
        raise ValueError('invalid cost')
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError('invalid cost')
    return result


def public_cases(dsl):
    result = {}
    for i in range(8):
        program = sample_program(dsl, 8100000+i)
        xs = [sample_input(9100000+i*100+j) for j in range(34)]
        history = []
        for x in xs[:2]:
            state = program.run(x)
            history.append(dict(inputs=x, output=None if state is None else state.get_output()))
        result[str(i)] = dict(history=history, targets=xs[2:])
    return result


def private_outcomes(dsl, cases):
    result = {}
    for i, case in cases.items():
        program = sample_program(dsl, 8100000+int(i))
        ys = []
        for x in case['targets']:
            state = program.run(x)
            ys.append(None if state is None else state.get_output())
        result[i] = dict(target_inputs=case['targets'], outputs=ys)
    return result


def model_candidates(dsl, text):
    programs = parse_proposals(dsl, text)

    def evaluator(program):
        def run(x):
            state = program.run(x)
            return None if state is None else state.get_output()
        return run

    return [(str(p), evaluator(p)) for p in programs]


def pool_forecast(candidates, case):
    # Only the explicitly recognized semantic empty-pool failure becomes abstention.
    if not candidates:
        return None
    try:
        return pred.forecast(candidates, case['history'], case['targets'])
    except ValueError as exc:
        if str(exc) == 'no history-compatible candidate; no forecast produced':
            return None
        raise


def validate_panel(panel):
    if type(panel) is not dict or set(panel) != {'version', 'cases', 'forecasts'} or panel['version'] != 1:
        raise ValueError('invalid panel')
    if set(panel['cases']) != {str(i) for i in range(8)} or set(panel['forecasts']) != set(panel['cases']):
        raise ValueError('eight complete cases required')
    for i, case in panel['cases'].items():
        if set(case) != {'history', 'targets'} or len(case['history']) != 2 or len(case['targets']) != 32:
            raise ValueError('case dimensions changed')
        history, targets = pred._history(case['history']), pred._targets(case['targets'])
        arms = panel['forecasts'][i]
        if set(arms) != set(ARMS):
            raise ValueError('missing control')
        for f in arms.values():
            if f is not None:
                pred._validate_forecast(f)
                if f['history_sha256'] != pred.digest(history) or f['target_inputs'] != targets:
                    raise ValueError('history/target identity mismatch')


def evaluate_sealed(path, sha, outcome_loader):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError('forecast seal mismatch')
    panel = json.loads(data, object_pairs_hook=pred._unique_keys)
    validate_panel(panel)
    coverage = {a: sum(fs[a] is not None for fs in panel['forecasts'].values()) for a in ARMS}
    if coverage['history_aware'] < 6:
        return dict(status='proposal_coverage_null', coverage=coverage, endpoints_opened=False,
                    depth_authorized=False)
    truth = outcome_loader()
    if set(truth) != set(panel['cases']):
        raise ValueError('outcome coverage mismatch')
    rows = {}
    for i, case in panel['cases'].items():
        y = truth[i]
        if (set(y) != {'target_inputs', 'outputs'} or y['target_inputs'] != case['targets']
                or type(y['outputs']) is not list or len(y['outputs']) != 32):
            raise ValueError('outcome target mismatch')
        ys = [pred.category(v) for v in y['outputs']]
        rows[i] = {}
        for a in ARMS:
            f = panel['forecasts'][i][a]
            if f is None:
                rows[i][a] = dict(abstained=True, adjusted_brier=1., zero_mass_targets=32)
                continue
            n, losses, zeros = len(f['candidate_keys']), [], 0
            for counts, label in zip(f['counts'], ys):
                p = counts.get(label, 0)/n
                losses.append(.5*(1+sum((c/n)**2 for c in counts.values())-2*p))
                zeros += p == 0
            rows[i][a] = dict(abstained=False, adjusted_brier=sum(losses)/32, zero_mass_targets=zeros)
    means = {a: sum(r[a]['adjusted_brier'] for r in rows.values())/8 for a in ARMS}
    wins = {a: sum(r['history_aware']['adjusted_brier'] < r[a]['adjusted_brier'] for r in rows.values())
            for a in ARMS[1:]}
    passed = all(means['history_aware'] <= .9*means[a] and wins[a] >= 5
                 and sum(r['history_aware']['zero_mass_targets'] for r in rows.values())
                 <= sum(r[a]['zero_mass_targets'] for r in rows.values()) for a in ARMS[1:])
    return dict(status='proposal_screen_pass' if passed else 'proposal_quality_null', coverage=coverage,
                means=means, paired_wins=wins, rows=rows, endpoints_opened=True, depth_authorized=False)


def read_catalog():
    request = urllib.request.Request('https://openrouter.ai/api/v1/models',
                                    headers={'Authorization': 'Bearer '+os.environ['OPENROUTER_API_KEY']})
    with urllib.request.urlopen(request, timeout=30) as response:
        records = json.load(response)['data']
    record, = [r for r in records if r['id'] == MODEL]
    prices = record['pricing']
    if (money(prices['prompt']) > Decimal('.0000001')
            or money(prices['completion']) > Decimal('.0000002')
            or record.get('reasoning', {}).get('mandatory') is not False
            or not {'reasoning', 'seed', 'response_format', 'max_tokens'} <= set(record['supported_parameters'])):
        raise ValueError('model route or price gate failed')
    return record


def payload(msgs, seed):
    if len(pred.canonical(msgs).encode()) > 16000:
        raise ValueError('prompt byte bound exceeded')
    return dict(model=MODEL, messages=msgs, seed=seed, temperature=.7, max_tokens=4096,
                reasoning={'enabled': False, 'exclude': True},
                response_format={'type': 'json_object'},
                provider={'require_parameters': True, 'allow_fallbacks': False,
                          'max_price': {'prompt': .1, 'completion': .2}}, stream=False)


def response_content(raw):
    usage = raw['usage']
    cost = money(usage['cost'])
    if (cost > RESERVE or type(usage['prompt_tokens']) is not int or not 0 <= usage['prompt_tokens'] <= 65536
            or type(usage['completion_tokens']) is not int or not 0 <= usage['completion_tokens'] <= 4096
            or usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0) != 0):
        raise ValueError('serving cost/token/reasoning gate failed')
    choice, = raw['choices']
    if choice['finish_reason'] != 'stop' or choice['message'].get('reasoning'):
        raise ValueError('completion is truncated or reasoning-enabled')
    text = choice['message']['content']
    if type(text) is not str:
        raise ValueError('text response required')
    return text, cost


def run():
    # Exclusive directory is the attempt marker: no resume/retry of this gate.
    if ROOT.exists():
        raise RuntimeError('gate already opened; no rerun')
    ledger = json.loads(LEDGER.read_text())
    live = read_live_credits()
    require_budget(ledger, projected_cost_usd=.25, total_usage_usd=live['total_usage_usd'])
    if money(live['balance_usd']) < CAP or ledger.get('pending_reservations'):
        raise RuntimeError('insufficient balance or outstanding account reservations')
    catalog = read_catalog()
    dsl = load_dsl()
    ROOT.mkdir()
    report = dict(status='incomplete', cost_usd=0., calls=0, endpoints_opened=False,
                  depth_authorized=False, protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest())
    report['implementation_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'scripts/deepcoder_proposal_gate.py', 'scripts/deepcoder_opportunity.py',
        'environments/program_induction/proposals.py', 'environments/program_induction/prediction.py',
        'environments/program_induction/synthesis.py')}
    base = money(ledger['recorded_actual_spend_usd'])
    charges = Decimal(0)
    pending = False
    try:
        save(ROOT/'catalog.json', catalog, exclusive=True)
        cases = public_cases(dsl)
        save(ROOT/'public.json', cases, exclusive=True)
        forecasts = {i: {} for i in cases}
        search_work = {}
        for i, case in cases.items():
            candidates, search_work[i] = [], []
            for restart in range(8):
                result = synthesize(dsl, [(r['inputs'], r['output']) for r in case['history']],
                                    order_seed=10100000+int(i)*10+restart)
                expr = result.pop('expression')
                search_work[i].append(result)
                if expr is not None:
                    candidates.append((expr.expression(), lambda x, e=expr: evaluate_expression(e, x)))
            forecasts[i]['symbolic_search'] = pool_forecast(candidates, case)
        save(ROOT/'search.json', search_work, exclusive=True)
        for i, case in cases.items():
            order = ARMS[:2] if int(i) % 2 == 0 else ARMS[1::-1]
            for a in order:
                tag = f'{i}_{a}'
                body = payload(messages(dsl, case['history'], history_blind=a == 'history_blind'),
                               11100000+int(i))
                catalog = read_catalog()
                save(ROOT/(tag+'.catalog.json'), catalog, exclusive=True)
                save(ROOT/(tag+'.request.json'), body, exclusive=True)
                if charges+RESERVE > CAP:
                    raise RuntimeError('block cap exceeded')
                ledger = json.loads(LEDGER.read_text())
                live = read_live_credits()
                require_budget(ledger, projected_cost_usd=float(RESERVE), total_usage_usd=live['total_usage_usd'])
                if money(live['balance_usd']) < RESERVE or ledger.get('pending_reservations'):
                    raise RuntimeError('account reservation conflict')
                ledger['pending_reservations'] = {tag: float(RESERVE)}
                save(LEDGER, ledger)
                pending = True
                # Fresh authorization immediately precedes each single HTTP attempt.
                live = read_live_credits()
                require_budget(ledger, projected_cost_usd=float(RESERVE), total_usage_usd=live['total_usage_usd'])
                request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
                    data=pred.canonical(body).encode(), headers={'Authorization': 'Bearer '+os.environ['OPENROUTER_API_KEY'],
                                                               'Content-Type': 'application/json'})
                report['calls'] += 1
                save(ROOT/'result.json', report)
                with urllib.request.urlopen(request, timeout=180) as response:
                    raw = json.load(response)
                save(ROOT/(tag+'.response.json'), raw, exclusive=True)
                # Reconcile accepted cost even if the scientific response later fails.
                cost = money(raw['usage']['cost'])
                charges += cost
                pending = False
                ledger['pending_reservations'] = {}
                live = read_live_credits()
                ledger['recorded_actual_spend_usd'] = float(max(money(ledger['recorded_actual_spend_usd']),
                    base+charges, money(live['total_usage_usd'])-money(ledger['opening_total_usage_usd'])))
                save(LEDGER, ledger)
                text, _ = response_content(raw)
                forecasts[i][a] = pool_forecast(model_candidates(dsl, text), case)
                report['cost_usd'] = float(charges)
                save(ROOT/'result.json', report)
        panel = dict(version=1, cases=cases, forecasts=forecasts)
        validate_panel(panel)
        save(ROOT/'forecasts.json', panel, exclusive=True)
        sha = hashlib.sha256((ROOT/'forecasts.json').read_bytes()).hexdigest()
        report['forecast_sha256'] = sha
        save(ROOT/'result.json', report)

        def outcomes():
            report['endpoints_opened'] = True
            save(ROOT/'result.json', report)
            truth = private_outcomes(dsl, cases)
            save(ROOT/'outcomes.json', truth, exclusive=True)
            return truth

        report.update(evaluate_sealed(ROOT/'forecasts.json', sha, outcomes))
    except Exception as exc:
        report.update(status='failed_closed', error_type=type(exc).__name__)
    finally:
        if pending:
            charges += RESERVE
            ledger['pending_reservations'] = {'uncertain_gate_request': float(RESERVE)}
        ledger['recorded_actual_spend_usd'] = float(max(money(ledger['recorded_actual_spend_usd']), base+charges))
        save(LEDGER, ledger)
        report['cost_usd'] = float(charges)
        report['completed_at'] = datetime.now(ZoneInfo('Europe/London')).isoformat()
        save(ROOT/'result.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.parse_args()
    with LEDGER.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(pred.canonical(run()))
