"""Source-only paired copy probe, with no scientific endpoint loader."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from environments.program_induction import scalar_tree
from scripts import deepcoder_luna_medium_probe as luna
from scripts.deepcoder_proposal_gate import save
from scripts.paid_program_probe import PaidProbe

ROOT = Path('results/nonmyopic/tree_copy_fidelity_20260909')
PROTOCOL = Path('results/nonmyopic/TREE_COPY_FIDELITY_PROTOCOL_20260909.md')
PROTOCOL_SHA = '2bd73d9a5356ece4f55e3b36f792a36f7bf51543c44818713030b895ab287499'
BINDINGS = [__file__, str(PROTOCOL), 'environments/program_induction/scalar_tree.py',
    'environments/program_induction/scalar_expression.py', 'environments/program_induction/physics_feedback.py',
    'environments/program_induction/physics_proposals.py', 'scripts/paid_program_probe.py',
    'scripts/deepcoder_proposal_gate.py', 'scripts/deepcoder_luna_medium_probe.py',
    'scripts/deepcoder_luna_transition.py', 'scripts/openrouter_daily_budget.py']
TARGET = {'trees': [{'op': 'add', 'left': {'op': 'variable', 'name': 'x0'},
    'right': {'op': 'mul', 'left': {'op': 'constant', 'value': 2},
              'right': {'op': 'sin', 'arg': {'op': 'variable', 'name': 'x1'}}}}]}


def body(arm):
    if arm not in ('recursive', 'string'):
        raise ValueError('unknown arm')
    schema = scalar_tree.schema(2) if arm == 'recursive' else dict(type='object',
        properties={'payload': {'type': 'string'}}, required=['payload'], additionalProperties=False)
    instruction = 'Return the supplied target object exactly, as JSON.' if arm == 'recursive' else 'Return a JSON object with exactly one payload string, containing a JSON serialization of the supplied target object.'
    return luna.bumped(dict(model='unused', temperature=0, max_tokens=4096,
        reasoning={'enabled': False, 'exclude': True}, seed=57100001,
        messages=[dict(role='system', content='This is a lossless data-copying test. Do not simplify, evaluate or change the target. '+instruction),
                  dict(role='user', content=json.dumps(TARGET, sort_keys=True))],
        response_format={'type': 'json_schema', 'json_schema': {'name': 'tree_copy', 'strict': True, 'schema': schema}}))


def measure(text, arm):
    try:
        if len(text.encode()) > 32768:
            raise ValueError('response cap')
        if arm == 'string':
            wrapper = json.loads(text, object_pairs_hook=scalar_tree._object)
            if type(wrapper) is not dict or set(wrapper) != {'payload'} or type(wrapper['payload']) is not str:
                raise ValueError('invalid wrapper')
            text = wrapper['payload']
        expressions = scalar_tree.decode(text, 2)
        data = json.loads(text, object_pairs_hook=scalar_tree._object)
        return dict(format_valid=True, exact_copy=data == TARGET, expressions=expressions)
    except (ValueError, RecursionError) as error:
        return dict(format_valid=False, exact_copy=False, error_type=type(error).__name__)


def collect(block):
    results = {}
    for arm in ('recursive', 'string'):
        raw = block.request(arm, body(arm))
        results[arm] = measure(luna.validate_response(raw), arm)
        block.report['results'] = results
        save(block.root/'result.json', block.report)
    block.report.update(status='complete', both_copy=all(r['exact_copy'] for r in results.values()),
                        endpoints_opened=False, depth_authorized=False)


def run(ledger):
    with PaidProbe(ROOT, ledger, .08, PROTOCOL, PROTOCOL_SHA, BINDINGS) as block:
        collect(block)
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
    results, cost = {}, 0.
    for arm in ('recursive', 'string'):
        if json.loads((root/(arm+'.request.json')).read_text()) != body(arm):
            raise ValueError('request changed')
        raw = json.loads((root/(arm+'.response.json')).read_text())
        results[arm] = measure(luna.validate_response(raw), arm)
        cost += raw['usage']['cost']
    if results != report['results'] or report['calls'] != 2 or abs(cost-report['accepted_cost_usd']) > 1e-10:
        raise ValueError('result/receipt mismatch')
    if report['both_copy'] != all(r['exact_copy'] for r in results.values()) or report['endpoints_opened'] or report['depth_authorized']:
        raise ValueError('invalid disposition')
    return dict(status='replay_valid', results=results, cost=cost)


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
