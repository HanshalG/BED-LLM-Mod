"""One-shot development repair diagnostic; no depth authorization."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path

from environments.program_induction import physics_feedback as feedback
from scripts import physgym_semantic_probe as parent
from scripts.deepcoder_proposal_gate import save
from scripts.paid_program_probe import PaidProbe

ROOT = Path('results/nonmyopic/physics_paired_repair_20260909')
PROTOCOL = Path('results/nonmyopic/PHYSICS_PAIRED_REPAIR_PROTOCOL_20260909.md')
PROTOCOL_SHA = 'ae09fe8c7d46b308ae2395d157b903aee6e9c8ed9dbc0793954802fcc97231cc'
INPUTS = {
    'public.json': '0322598d54ab38864cd996dc225818122622fe39b58d2128f238a22f081bdda5',
    'forecasts.json': '630a1eda7068e0087cbec526da4b929f3bb125efd1e4f13e9123e98882b54a27',
    'outcomes.json': 'ec94fe2d94a80176c00fa37f2e190971ac5026371ffc71368032abec9aea575e',
}
BINDINGS = list(dict.fromkeys([__file__, str(PROTOCOL),
    'environments/program_induction/physics_feedback.py',
    'scripts/deepcoder_proposal_gate.py'] + parent.BINDINGS))


def load(name):
    raw = (parent.ROOT / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUTS[name]:
        raise ValueError('banked input changed')
    return json.loads(raw)['457']


def body(case, initial, arm):
    if arm not in ('control', 'refresh'):
        raise ValueError('unknown arm')
    value = parent.body(case, 'refresh', 2)
    value['seed'] = 56100002
    value['messages'] = feedback.revision_messages(
        case['names'], case['context'], case['descriptions'],
        case['history'][:3 if arm == 'control' else 4], initial)
    return value


def panel(case, initial, request):
    pools = {}
    for arm in ('control', 'refresh'):
        raw = request(arm, body(case, initial, arm))
        pools[arm] = parent.model.decode(parent.luna.validate_response(raw), len(case['names']))
    forecasts = {}
    for arm, expressions in dict(initial=initial, **{
            a: initial + p for a, p in pools.items()}).items():
        forecasts[arm] = feedback.calibrated_prediction(
            expressions, case['names'], case['history'], case['targets'])
    return dict(pools=pools, forecasts=forecasts, diagnostics={
        a: feedback.feedback(p, case['names'], case['history']) for a, p in pools.items()})


def score(forecasts, truth):
    if len(truth) != 32 or not all(math.isfinite(y) for y in truth):
        raise ValueError('invalid targets')
    losses = {}
    for arm in ('initial', 'control', 'refresh'):
        f = forecasts[arm]
        if f['status'] == 'empty_support':
            losses[arm] = None
        elif f['status'] != 'complete' or len(f['mean']) != 32 or not all(math.isfinite(y) for y in f['mean']):
            raise ValueError('invalid prediction')
        else:
            losses[arm] = sum((p-y)**2 for p, y in zip(f['mean'], truth))/32
    complete = all(v is not None for v in losses.values())
    signal = complete and all(losses['refresh'] < losses[a] and
                             losses['refresh'] <= .9*losses[a] for a in ('initial', 'control'))
    return dict(status='complete', losses=losses, descriptive_repair_signal=signal,
                depth_authorized=False, fresh_evidence=False)


def collect(block, case, initial, endpoints):
    result = panel(case, initial, block.request)
    path = block.root / 'forecasts.json'
    save(path, result, exclusive=True)
    block.report['forecast_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(block.root / 'result.json', block.report)
    truth = endpoints()
    save(block.root / 'outcomes.json', truth, exclusive=True)
    block.report.update(endpoints_opened=True, **score(result['forecasts'], truth))


def run(ledger):
    case, initial = load('public.json'), load('forecasts.json')['pools']['semantic']
    with PaidProbe(ROOT, ledger, .08, PROTOCOL, PROTOCOL_SHA, BINDINGS) as block:
        collect(block, case, initial, lambda: load('outcomes.json'))
    return block.report


def replay(root):
    report = json.loads((root / 'result.json').read_text())
    if report['status'] != 'complete' or report['protocol_sha256'] != PROTOCOL_SHA:
        raise ValueError('not complete frozen run')
    if set(report['implementation_sha256']) != set(BINDINGS):
        raise ValueError('missing binding')
    for path, digest in report['implementation_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('binding changed')
    receipts = []

    def request(tag, expected):
        if json.loads((root / (tag+'.request.json')).read_text()) != expected:
            raise ValueError('request changed')
        raw = json.loads((root / (tag+'.response.json')).read_text())
        receipts.append(raw['usage']['cost'])
        return raw

    result = panel(load('public.json'), load('forecasts.json')['pools']['semantic'], request)
    path = root / 'forecasts.json'
    if result != json.loads(path.read_text()) or hashlib.sha256(path.read_bytes()).hexdigest() != report['forecast_sha256']:
        raise ValueError('forecast changed')
    truth = load('outcomes.json')
    if truth != json.loads((root / 'outcomes.json').read_text()):
        raise ValueError('target changed')
    measured = score(result['forecasts'], truth)
    if any(report.get(k) != v for k, v in measured.items()) or report['calls'] != 2 or len(receipts) != 2:
        raise ValueError('result changed')
    if abs(sum(receipts)-report['accepted_cost_usd']) > 1e-10:
        raise ValueError('receipt mismatch')
    return dict(status='replay_valid', **{k: report[k] for k in (
        'losses', 'descriptive_repair_signal', 'accepted_cost_usd')})


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
