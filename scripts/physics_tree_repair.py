"""Frozen recursive-tree repair diagnostic on opened development data."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from environments.program_induction import scalar_tree
from scripts import physics_paired_repair as previous
from scripts.deepcoder_proposal_gate import save
from scripts.paid_program_probe import PaidProbe

ROOT = Path('results/nonmyopic/physics_tree_repair_20260909')
PROTOCOL = Path('results/nonmyopic/PHYSICS_TREE_REPAIR_PROTOCOL_20260909.md')
PROTOCOL_SHA = 'c1bf5cd10f19bd38c885f251c0748e5f7a5626a1e96513911505e2d47248b058'
BINDINGS = list(dict.fromkeys([__file__, str(PROTOCOL),
    'environments/program_induction/scalar_tree.py'] + previous.BINDINGS))


def body(case, initial, arm):
    value = previous.body(case, initial, arm)
    value['messages'] = scalar_tree.messages(case['names'], case['context'], case['descriptions'],
        case['history'][:3 if arm == 'control' else 4], initial)
    value['response_format']['json_schema'].update(name='scalar_trees', schema=scalar_tree.schema(len(case['names'])))
    return value


def panel(case, initial, request):
    pools = {}
    for arm in ('control', 'refresh'):
        raw = request(arm, body(case, initial, arm))
        pools[arm] = scalar_tree.decode(previous.parent.luna.validate_response(raw), len(case['names']))
    forecasts = {a: previous.feedback.calibrated_prediction(p, case['names'], case['history'], case['targets'])
                 for a, p in dict(initial=initial, **{a: initial+p for a, p in pools.items()}).items()}
    return dict(pools=pools, forecasts=forecasts, diagnostics={
        a: previous.feedback.feedback(p, case['names'], case['history']) for a, p in pools.items()})


def collect(block, case, initial, endpoints):
    result = panel(case, initial, block.request)
    path = block.root/'forecasts.json'
    save(path, result, exclusive=True)
    block.report['forecast_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(block.root/'result.json', block.report)
    truth = endpoints()
    save(block.root/'outcomes.json', truth, exclusive=True)
    block.report.update(endpoints_opened=True, **previous.score(result['forecasts'], truth))


def run(ledger):
    case = previous.load('public.json')
    initial = previous.load('forecasts.json')['pools']['semantic']
    with PaidProbe(ROOT, ledger, .08, PROTOCOL, PROTOCOL_SHA, BINDINGS) as block:
        collect(block, case, initial, lambda: previous.load('outcomes.json'))
    return block.report


def replay(root):
    report = json.loads((root/'result.json').read_text())
    if report['status'] != 'complete' or report['protocol_sha256'] != PROTOCOL_SHA:
        raise ValueError('not complete frozen run')
    if set(report['implementation_sha256']) != set(BINDINGS):
        raise ValueError('missing binding')
    for path, digest in report['implementation_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('binding changed')
    receipts = []
    def request(tag, expected):
        if json.loads((root/(tag+'.request.json')).read_text()) != expected:
            raise ValueError('request changed')
        raw = json.loads((root/(tag+'.response.json')).read_text())
        receipts.append(raw['usage']['cost'])
        return raw
    result = panel(previous.load('public.json'), previous.load('forecasts.json')['pools']['semantic'], request)
    path = root/'forecasts.json'
    if result != json.loads(path.read_text()) or hashlib.sha256(path.read_bytes()).hexdigest() != report['forecast_sha256']:
        raise ValueError('forecast changed')
    truth = previous.load('outcomes.json')
    if truth != json.loads((root/'outcomes.json').read_text()):
        raise ValueError('target changed')
    measured = previous.score(result['forecasts'], truth)
    if any(report.get(k) != v for k, v in measured.items()) or report['calls'] != 2 or len(receipts) != 2:
        raise ValueError('result changed')
    if abs(sum(receipts)-report['accepted_cost_usd']) > 1e-10:
        raise ValueError('receipt mismatch')
    return dict(status='replay_valid', losses=report['losses'],
                descriptive_repair_signal=report['descriptive_repair_signal'],
                accepted_cost_usd=report['accepted_cost_usd'])


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
