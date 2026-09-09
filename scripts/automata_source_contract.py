"""Isolated pinned upstream API audit, not a benchmark or learned policy."""
import ast
import hashlib
import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

SOURCE = Path('/private/tmp/bed-aalpy-source-audit')
COMMIT = 'ca24a8a2ae224bec8e229753bc9f6a55b6297870'
SUL_SHA = '4d80f444833a59525f3c82f224110456b8c8b0fe997d23173050f44677f885b9'
OUT = Path('results/nonmyopic/AUTOMATA_QUERY_CONTRACT_AUDIT_20260909.json')


def load_sul(raw):
    if hashlib.sha256(raw).hexdigest() != SUL_SHA:
        raise ValueError('upstream SUL source changed')
    tree = ast.parse(raw)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SUL']
    if len(classes) != 1:
        raise ValueError('SUL class coverage')
    # Only the inspected upstream base class executes, not imports or oracle code.
    scope = {'ABC': ABC, 'abstractmethod': abstractmethod, 'Any': Any}
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<pinned-SUL>', 'exec'), scope)
    return scope['SUL']


def measure(base):
    class Fixture(base):
        def __init__(self):
            super().__init__()
            self.resets = self.physical_steps = 0
            self.armed = False

        def pre(self):
            self.resets += 1
            self.armed = False

        def post(self):
            pass

        def step(self, letter):
            self.physical_steps += 1
            if letter == 'fail':
                raise RuntimeError('fixture failure after attempted step')
            if letter == 'arm':
                self.armed = True
                return 'ack'
            return 'armed' if self.armed else 'idle'

    def counts(f):
        return {'membership_queries': f.num_queries, 'reported_steps': f.num_steps,
                'physical_steps': f.physical_steps, 'resets': f.resets}
    whole = Fixture()
    output = whole.query(('arm', 'probe'))
    split = Fixture()
    split_output = [split.query((a,)) for a in ('arm', 'probe')]
    failed = Fixture()
    try:
        failed.query(('arm', 'fail'))
    except RuntimeError:
        pass
    else:
        raise AssertionError('fixture must fail')
    return {'whole_word': {'outputs': output, **counts(whole)},
            'split_words': {'outputs': split_output, **counts(split)},
            'interrupted_word': counts(failed)}


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    raw = subprocess.check_output(['git', '-C', str(SOURCE), 'show',
                                   COMMIT+':aalpy/base/SUL.py'], timeout=20)
    result = {'status': 'isolated_source_contract', 'source_commit': COMMIT,
              'source_sha256': SUL_SHA, 'measurements': measure(load_sul(raw)),
              'benchmark_instances_loaded': 0, 'model_calls': 0, 'cost_usd': 0,
              'planning_opportunity_established': False, 'paid_authorized': False}
    with OUT.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == '__main__': main()
