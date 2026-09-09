"""Fixed synthetic correction qualification, not a policy or depth experiment."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from environments.program_induction import physics_edits, physics_feedback
from environments.program_induction.scalar_expression import ScalarExpression
from scripts import deepcoder_luna_medium_probe as luna
from scripts.deepcoder_proposal_gate import save
from scripts.paid_program_probe import PaidProbe

ROOT = Path('results/nonmyopic/compositional_edit_qualification_20260909')
PROTOCOL = Path('results/nonmyopic/COMPOSITIONAL_EDIT_QUALIFICATION_PROTOCOL_20260909.md')
PROTOCOL_SHA = '295f6559a1eab37ea66c2d9e6c6267e924aa94a31a1ac3785b161c48c0a6b6a9'
BINDINGS = [__file__, str(PROTOCOL), 'environments/program_induction/physics_edits.py',
    'environments/program_induction/scalar_tree.py', 'environments/program_induction/scalar_expression.py',
    'environments/program_induction/physics_feedback.py', 'environments/program_induction/physics_proposals.py',
    'scripts/deepcoder_luna_medium_probe.py', 'scripts/deepcoder_proposal_gate.py',
    'scripts/deepcoder_luna_transition.py', 'scripts/paid_program_probe.py', 'scripts/openrouter_daily_budget.py']
BASES = ('x0/(1+x0)', 'sqrt(x0)', 'x0', 'sqrt(x0)')
SOURCES = ('x0/(1+x0)*exp(-.7*x1)', 'sqrt(x0)+.4*x1**2', 'x0/(1+x1**2)', 'sqrt(x0)+.6*sin(x1)')
NAMES = ['x0', 'x1']
HISTORY_POINTS = ((.6, 1), (1.2, 1), (1.8, 1), (.8, .5), (1.4, 1.4), (1.1, 2))


def cases():
    result = {}
    for i in range(4):
        f = ScalarExpression(SOURCES[i], NAMES)
        noise, inputs = random.Random(59100000+i), random.Random(59200000+i)
        history = [[dict(zip(NAMES, xy)), math.log(f(dict(zip(NAMES, xy))))+noise.gauss(0, .05)] for xy in HISTORY_POINTS]
        targets = [{n: math.exp(inputs.uniform(math.log(.5), math.log(2))) for n in NAMES} for _ in range(32)]
        result[str(i)] = dict(base=BASES[i], history=history, targets=targets)
    return result


def outcomes(public):
    result = {}
    for i in range(4):
        noise = random.Random(59100000+i)
        for _ in range(6):
            noise.gauss(0, .05)
        f = ScalarExpression(SOURCES[i], NAMES)
        result[str(i)] = [math.log(f(p))+noise.gauss(0, .05) for p in public[str(i)]['targets']]
    return result


def body(case, arm, index):
    if arm not in ('control', 'refresh'):
        raise ValueError('unknown arm')
    return luna.bumped(dict(model='unused', temperature=0, max_tokens=4096,
        reasoning={'enabled': False, 'exclude': True}, seed=59300000+index,
        messages=physics_edits.messages(NAMES, 'Synthetic positive response with two independently controllable inputs.',
            {'x0': 'Primary input', 'x1': 'Secondary input'}, case['history'][:3 if arm == 'control' else 6], case['base']),
        response_format={'type': 'json_schema', 'json_schema': {'name': 'corrections', 'strict': True, 'schema': physics_edits.schema(2)}}))


def features(points):
    return np.array([[1., math.log(p['x0']), math.log(p['x1']), p['x0'], p['x1'],
                      p['x0']**2, p['x0']*p['x1'], p['x1']**2] for p in points])


def ridge(history, targets):
    x = features([p for p, _ in history])
    penalty = np.eye(8)*.01
    penalty[0, 0] = 0
    beta = np.linalg.solve(x.T@x+penalty, x.T@np.array([y for _, y in history]))
    return dict(status='complete', mean=(features(targets)@beta).tolist())


def panel(public, request):
    result = {}
    for i in range(4):
        case, pools = public[str(i)], {}
        for arm in (('control', 'refresh') if i % 2 == 0 else ('refresh', 'control')):
            raw = request(f'{i}_{arm}', body(case, arm, i))
            pools[arm] = physics_edits.decode(luna.validate_response(raw), 2, case['base'])
        forecasts = {'base': physics_feedback.calibrated_prediction([case['base']], NAMES, case['history'], case['targets']),
                     'ridge': ridge(case['history'], case['targets'])}
        for arm, edits in pools.items():
            forecasts[arm] = physics_feedback.calibrated_prediction(
                [case['base']]+[e['expression'] for e in edits], NAMES, case['history'], case['targets'])
        result[str(i)] = dict(pools=pools, forecasts=forecasts, diagnostics={
            a: physics_feedback.feedback([e['expression'] for e in p], NAMES, case['history']) for a, p in pools.items()})
    return result


def score(forecasts, truth):
    if set(forecasts) != set(map(str, range(4))) or set(truth) != set(forecasts):
        raise ValueError('incomplete cohort')
    losses = {}
    for t, record in forecasts.items():
        if len(truth[t]) != 32 or not all(math.isfinite(y) for y in truth[t]):
            raise ValueError('invalid endpoints')
        losses[t] = {}
        for arm in ('base', 'ridge', 'control', 'refresh'):
            f = record['forecasts'][arm]
            if f['status'] == 'empty_support':
                losses[t][arm] = None
            elif f['status'] != 'complete' or len(f['mean']) != 32 or not all(math.isfinite(y) for y in f['mean']):
                raise ValueError('invalid prediction')
            else:
                losses[t][arm] = sum((p-y)**2 for p, y in zip(f['mean'], truth[t]))/32
    result = dict(status='complete', losses=losses, gate_passed=False, depth_authorized=False)
    if any(v is None for r in losses.values() for v in r.values()):
        return dict(result, reason='empty_support')
    means = {a: sum(r[a] for r in losses.values())/4 for a in ('base', 'ridge', 'control', 'refresh')}
    differences = {t: r['control']-r['refresh'] for t, r in losses.items()}
    wins = sum(v > .01 for v in differences.values())
    result.update(means=means, paired_differences=differences, wins=wins,
        gate_passed=means['refresh'] <= .9*means['control'] and wins >= 2 and means['refresh'] <= means['ridge'])
    return result


def collect(block, public, endpoints):
    save(block.root/'public.json', public, exclusive=True)
    predictions = panel(public, block.request)
    path = block.root/'forecasts.json'
    save(path, predictions, exclusive=True)
    block.report['forecast_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(block.root/'result.json', block.report)
    truth = endpoints()
    save(block.root/'outcomes.json', truth, exclusive=True)
    block.report.update(endpoints_opened=True, **score(predictions, truth))


def run(ledger):
    public = cases()
    with PaidProbe(ROOT, ledger, .32, PROTOCOL, PROTOCOL_SHA, BINDINGS) as block:
        collect(block, public, lambda: outcomes(public))
    return block.report


def replay(root):
    report = json.loads((root/'result.json').read_text())
    if report['status'] != 'complete' or report['protocol_sha256'] != PROTOCOL_SHA or set(report['implementation_sha256']) != set(BINDINGS):
        raise ValueError('not complete frozen run')
    for p, digest in report['implementation_sha256'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest:
            raise ValueError('binding changed')
    public = cases()
    if public != json.loads((root/'public.json').read_text()):
        raise ValueError('public changed')
    receipts = []
    def request(tag, expected):
        if json.loads((root/(tag+'.request.json')).read_text()) != expected:
            raise ValueError('request changed')
        raw = json.loads((root/(tag+'.response.json')).read_text())
        receipts.append(raw['usage']['cost'])
        return raw
    forecasts = panel(public, request)
    path = root/'forecasts.json'
    if forecasts != json.loads(path.read_text()) or hashlib.sha256(path.read_bytes()).hexdigest() != report['forecast_sha256']:
        raise ValueError('forecast changed')
    truth = outcomes(public)
    if truth != json.loads((root/'outcomes.json').read_text()):
        raise ValueError('targets changed')
    if any(report.get(k) != v for k, v in score(forecasts, truth).items()) or report['calls'] != 8 or len(receipts) != 8 or abs(sum(receipts)-report['accepted_cost_usd']) > 1e-10:
        raise ValueError('result/receipt mismatch')
    return dict(status='replay_valid', means=report.get('means'), gate_passed=report['gate_passed'], cost=report['accepted_cost_usd'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(ROOT), sort_keys=True))
    else:
        if args.ledger is None:
            parser.error('--ledger required')
        with args.ledger.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(run(args.ledger), sort_keys=True))
