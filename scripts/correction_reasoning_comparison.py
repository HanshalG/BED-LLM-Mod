"""Fresh-source effort comparison; no policy or old endpoint reopening."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random

from scripts import compositional_edit_qualification as parent
from scripts.deepcoder_proposal_gate import save
from scripts.paid_program_probe import PaidProbe

ROOT = Path('results/nonmyopic/correction_reasoning_comparison_20260909')
PROTOCOL = Path('results/nonmyopic/CORRECTION_REASONING_COMPARISON_PROTOCOL_20260909.md')
PROTOCOL_SHA = 'ad01397536149e2f852881ae849238f1b4c1039a9508de6f08eaca67d4e6ad9f'
BINDINGS = list(dict.fromkeys([__file__, str(PROTOCOL)] + parent.BINDINGS))
BASES = ('x0', 'sqrt(x0)', 'x0', 'x0/(1+x0)')
SOURCES = ('x0*(1+.3*x1)', 'sqrt(x0)/(1+.4*x1)', 'x0+.2*exp(x1)', 'x0/(1+x0)*(1+.5*cos(x1))')


def cases():
    public = {}
    for i in range(4):
        f = parent.ScalarExpression(SOURCES[i], parent.NAMES)
        noise, inputs = random.Random(60100000+i), random.Random(60200000+i)
        history = [[dict(zip(parent.NAMES, xy)), math.log(f(dict(zip(parent.NAMES, xy))))+noise.gauss(0, .05)] for xy in parent.HISTORY_POINTS]
        targets = [{n: math.exp(inputs.uniform(math.log(.5), math.log(2))) for n in parent.NAMES} for _ in range(32)]
        public[str(i)] = dict(base=BASES[i], history=history, targets=targets)
    return public


def outcomes(public):
    result = {}
    for i in range(4):
        noise = random.Random(60100000+i)
        for _ in range(6):
            noise.gauss(0, .05)
        f = parent.ScalarExpression(SOURCES[i], parent.NAMES)
        result[str(i)] = [math.log(f(p))+noise.gauss(0, .05) for p in public[str(i)]['targets']]
    return result


def body(case, arm, index, effort):
    if effort not in ('medium', 'high'):
        raise ValueError('invalid effort')
    result = parent.body(case, arm, index)
    result['seed'] = 60300000+index
    result['reasoning']['effort'] = effort
    return result


def panel(public, request):
    results = {'medium': {}, 'high': {}}
    for i in range(4):
        case = public[str(i)]
        for effort in (('medium', 'high') if i % 2 == 0 else ('high', 'medium')):
            pools = {}
            for arm in (('control', 'refresh') if i % 2 == 0 else ('refresh', 'control')):
                raw = request(f'{i}_{effort}_{arm}', body(case, arm, i, effort))
                pools[arm] = parent.physics_edits.decode(parent.luna.validate_response(raw), 2, case['base'])
            forecasts = {'base': parent.physics_feedback.calibrated_prediction([case['base']], parent.NAMES, case['history'], case['targets']),
                         'ridge': parent.ridge(case['history'], case['targets'])}
            for arm, edits in pools.items():
                forecasts[arm] = parent.physics_feedback.calibrated_prediction(
                    [case['base']]+[e['expression'] for e in edits], parent.NAMES, case['history'], case['targets'])
            results[effort][str(i)] = dict(pools=pools, forecasts=forecasts, diagnostics={
                a: parent.physics_feedback.feedback([e['expression'] for e in p], parent.NAMES, case['history']) for a, p in pools.items()})
    return results


def score(forecasts, truth):
    if set(forecasts) != {'medium', 'high'}:
        raise ValueError('missing effort')
    return dict(status='complete', qualifications={e: parent.score(f, truth) for e, f in forecasts.items()}, depth_authorized=False)


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
    with PaidProbe(ROOT, ledger, .64, PROTOCOL, PROTOCOL_SHA, BINDINGS) as block:
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
    predictions = panel(public, request)
    path = root/'forecasts.json'
    if predictions != json.loads(path.read_text()) or hashlib.sha256(path.read_bytes()).hexdigest() != report['forecast_sha256']:
        raise ValueError('forecast changed')
    truth = outcomes(public)
    if truth != json.loads((root/'outcomes.json').read_text()):
        raise ValueError('targets changed')
    if any(report.get(k) != v for k, v in score(predictions, truth).items()) or report['calls'] != 16 or len(receipts) != 16 or abs(sum(receipts)-report['accepted_cost_usd']) > 1e-10:
        raise ValueError('result/receipt mismatch')
    return dict(status='replay_valid', cost=report['accepted_cost_usd'],
                qualifications={e: {k: r.get(k) for k in ('gate_passed', 'means', 'wins')} for e, r in report['qualifications'].items()})


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
