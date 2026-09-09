"""Initial-only exact conditional syntax sampling, empirical planning diagnostic."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

from scripts.robustfill_coverage_audit import load, atoms, SHA
from environments.program_induction.reference import ProgramReference

ROOT = Path('results/nonmyopic/robustfill_initial_opportunity_20260909')
INPUT = Path('results/nonmyopic/author_strings_opportunity_20260909/forecasts.json')
INPUT_SHA = '0e567b7429a6e4f7c37e836074701ebea6541fc0bc5bd54f5fa7a6829f866ffe'


def draw_rows(groups, target, total_atoms, rng, count=256):
    """Uniform length 1..3 then IID uniform atoms, conditioned on initial output."""
    totals = {o: sum(c for row, c in entries) for o, entries in groups.items()}

    @lru_cache(None)
    def ways(position, left):
        if left == 0:
            return int(position == len(target))
        return sum(n * ways(position + len(o), left - 1)
                   for o, n in totals.items() if target.startswith(o, position))

    def pick(options):
        total = sum(w for item, w in options)
        if total <= 0:
            raise ValueError('empty conditional support')
        z = rng.randrange(total)
        for item, w in options:
            z -= w
            if z < 0:
                return item
        raise AssertionError('unreachable')

    # Clear the common denominator 3*N**3 for the three length masses.
    lengths = [(k, ways(0, k) * total_atoms ** (3-k)) for k in (1, 2, 3)]
    if not sum(w for k, w in lengths):
        return [], dict(length_integer_weights=[w for k, w in lengths])
    rows = []
    for _ in range(count):
        left, position, selected = pick(lengths), 0, []
        while left:
            options = [(o, n * ways(position + len(o), left-1))
                       for o, n in totals.items() if target.startswith(o, position)]
            output = pick(options)
            selected.append(pick(groups[output]))
            position += len(output)
            left -= 1
        rows.append([''.join(p[j] for p in selected) for j in range(len(selected[0]))])
    return rows, dict(length_integer_weights=[w for k, w in lengths])


def support(dsl, initial, inputs, seed):
    groups = defaultdict(Counter)
    total = 0
    for atom in atoms(dsl):
        variants = [atom]
        if not isinstance(atom, dsl.ConstStr):
            variants += [dsl.Compose(dsl.ToCase(c), atom) for c in dsl.Case]
        for expression in variants:
            total += 1
            observed = expression(initial[0])
            if observed in initial[1]:
                groups[observed][tuple(expression(x) for x in inputs)] += 1
    rows, meta = draw_rows({o: list(c.items()) for o, c in groups.items()},
                           initial[1], total, random.Random(seed))
    return dict(rows=rows, total_atoms=total, distinct_rows=len(set(map(tuple, rows))), **meta)


def measure(rows):
    if not rows:
        return dict(status='empty_support')
    ref = ProgramReference(rows, 6, seconds=120)
    try:
        state, menu = ref.initial_state, tuple(range(6))
        values = {f'h{h}': ref.deployed(state, menu, 3, h) for h in (1, 2, 3)}
        values['random'] = ref.random(state, menu, 3)
        values['openloop'] = ref.planner.plan(state, 3, available=menu, mode='open_loop').root.expected_risk
        return dict(status='complete', initial_risk=ref.risk(state), expected_risk=values)
    except (TimeoutError, RuntimeError) as error:
        return dict(status='incomplete', error=type(error).__name__)
    finally:
        ref.clear()


def save(path, data):
    with path.open('x') as output:
        json.dump(data, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')


def main():
    if ROOT.exists() or hashlib.sha256(INPUT.read_bytes()).hexdigest() != INPUT_SHA:
        raise ValueError('already opened or input binding mismatch')
    ROOT.mkdir()
    cases = json.loads(INPUT.read_text())
    dsl = load()
    panels = {task: support(dsl, cases[task]['initial'], cases[task]['inputs'], 50100000+i)
              for i, task in enumerate(('1', '10'))}
    save(ROOT/'panels.json', panels)
    report = dict(source_sha256=SHA, input_sha256=INPUT_SHA, model_calls=0, cost_usd=0,
                  paid_authority=False, full_prior_reference=False,
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  panel_sha256=hashlib.sha256((ROOT/'panels.json').read_bytes()).hexdigest(),
                  results={t: measure(p['rows']) for t, p in panels.items()})
    save(ROOT/'result.json', report)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
