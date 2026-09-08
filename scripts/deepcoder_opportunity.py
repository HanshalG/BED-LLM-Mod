"""One frozen program-induction pilot, with no model transport or dataset reads."""
import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path
import random
from time import monotonic
import types
import urllib.request

from environments.program_induction.reference import ProgramReference


PIN = 'ef046ce2cc3fcd024e32f5dfe00e69700dac82ed'
DSL_SHA = 'b5564c6956ea1a49225c8e8d24b4a5118cda399ecfdadd7f2abca4e7960f4c42'


@lru_cache(maxsize=1)
def load_dsl():
    url = f'https://raw.githubusercontent.com/google-deepmind/exedec/{PIN}/tasks/deepcoder/deepcoder_dsl.py'
    with urllib.request.urlopen(url, timeout=30) as response:
        source = response.read(100000)
    if hashlib.sha256(source).hexdigest() != DSL_SHA:
        raise ValueError('pinned DSL identity failed')
    module = types.ModuleType('pinned_exedec_deepcoder')
    # Execute only hash-verified upstream implementation, never generated Python.
    exec(compile(source, url, 'exec'), module.__dict__)
    from absl import flags
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(['deepcoder-reference'])
    if module.deepcoder_max_list_length() != 5 or module.deepcoder_max_int() != 50:
        raise ValueError('DSL configuration changed')
    return module


def sample_program(dsl, seed):
    rng = random.Random(seed)
    variables = [('x0', list), ('x1', list)]
    statements = []
    for index in range(rng.choice((2, 3, 4))):
        choices = []
        for op in dsl.OPERATIONS:
            args = []
            for typ in op.inputs_type:
                if isinstance(typ, tuple):
                    args.append([lam for lam in dsl.LAMBDAS
                                 if (lam.inputs_type, lam.output_type) == typ])
                else:
                    args.append([name for name, value_type in variables if value_type == typ])
            for actual in product(*args):
                if not statements or variables[-1][0] in actual:
                    choices.append((op, actual))
        if not choices:
            raise ValueError('grammar has no valid continuation')
        op, args = rng.choice(choices)
        name = f'x{index+2}'
        statements.append(dsl.Statement(name, op, args))
        variables.append((name, op.output_type))
    return dsl.Program(['x0', 'x1'], statements)


def sample_input(seed):
    rng = random.Random(seed)
    return [[rng.randint(-10, 10) for _ in range(rng.randint(1, 5))] for _ in range(2)]


def outcome(program, inputs):
    state = program.run(inputs)
    if state is None:
        return 'ERROR'
    value = state.get_output()
    return json.dumps({'type': 'int' if type(value) is int else 'list', 'value': value},
                      sort_keys=True, separators=(',', ':'))


def panel(index, dsl):
    start = monotonic()
    programs = [sample_program(dsl, 3100000+index*1000+i) for i in range(128)]
    inputs = [sample_input(4100000+index*1000+i) for i in range(40)]
    rows = [[outcome(program, inp) for inp in inputs] for program in programs]
    ref = ProgramReference(rows, 8, seconds=120)
    menu, state = tuple(range(8)), ref.initial_state
    try:
        initial = ref.risk(state)
        optimal = ref.planner.plan(state, 4, available=menu)
        values = {f'h{h}': ref.deployed(state, menu, 4, h) for h in (1, 2, 3)}
        values['receding_openloop_h3'] = ref.deployed(state, menu, 4, 3, 'open_loop')
        values['random'] = ref.random(state, menu, 4)
        values['exact_myopic_control'] = values['h1']
        roots = {f'h{h}': asdict(ref.planner.plan(state, h, available=menu)) for h in (1, 2, 3)}
        return dict(index=index, initial_risk=initial, optimal_B4=optimal.root.expected_risk,
                    values=values, initial_plans=roots,
                    matrix_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
                    input_sha256=hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
                    program_sha256=hashlib.sha256(json.dumps([str(p) for p in programs]).encode()).hexdigest(),
                    distinct_programs=len({str(p) for p in programs}),
                    distinct_extensions=len({tuple(row) for row in rows}),
                    error_fraction=sum(v == 'ERROR' for row in rows for v in row)/(128*40),
                    constant_extensions=sum(len(set(row)) == 1 for row in rows),
                    elapsed_seconds=monotonic()-start)
    finally:
        ref.clear()


def run(path):
    path = Path(path)
    result = dict(status='incomplete', panels=[], source_pin=PIN, dsl_sha256=DSL_SHA,
                  model_calls=0, model_cost_usd=0, paid_calls_authorized=False,
                  scope='finite empirical program prior; not full grammar or LLM efficacy')
    with path.open('x') as stream:
        json.dump(result, stream)
    try:
        dsl = load_dsl()
        for i in range(4):
            result['panels'].append(panel(i, dsl))
            path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
        means = {key: sum(p['values'][key] for p in result['panels'])/4
                 for key in result['panels'][0]['values']}
        optimal = sum(p['optimal_B4'] for p in result['panels'])/4
        ordered = sum(p['values']['h3'] <= p['values']['h2']+1e-12
                      <= p['values']['h1']+2e-12 for p in result['panels'])
        passed = (means['h1'] > 0 and means['h2'] > 0
                  and means['h2'] <= .95*means['h1']
                  and means['h3'] <= .95*means['h2']
                  and means['h3'] < means['receding_openloop_h3']-1e-12 and ordered >= 3)
        result.update(status='finite_opportunity_pass' if passed else 'finite_opportunity_null',
                      means=means, optimal_B4_mean=optimal, ordered_panels=ordered)
    except Exception as exc:
        result.update(status='failed_closed', error_type=type(exc).__name__)
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    if result['status'] == 'failed_closed':
        raise RuntimeError('failure banked; no automatic rerun')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
