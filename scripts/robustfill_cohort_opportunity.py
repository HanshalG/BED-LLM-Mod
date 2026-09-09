"""Fixed development cohort; later labels validate source, never score forecasts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from scripts.author_strings_audit import COMMIT, inspect, parse
from scripts.author_strings_overlap_audit import get, choose
from scripts.robustfill_initial_opportunity import support, measure, save
from scripts.robustfill_coverage_audit import load

ROOT = Path('results/nonmyopic/robustfill_cohort_opportunity_20260909')
COHORT = ('185', '104', '88', '98', '272', '195', '29', '110', '80', '12', '70', '91')


def public_case(bk, exs):
    metadata = inspect(bk, exs)
    xs, ys = defaultdict(dict), defaultdict(dict)
    for name, args in parse(bk):
        if name == 'in':
            e, p, c = args
            xs[e][p] = c
    for name, args in parse(exs):
        if name == 'pos':
            e, p, c = args[0][1]
            ys[e][p] = c
    if any(not e.startswith('e') or not e[1:].isdigit() for e in xs):
        raise ValueError('invalid example ID')
    ids = sorted(xs, key=lambda e: int(e[1:]))[:10]
    string = lambda chars: ''.join(chars[i] for i in sorted(chars))
    inputs = [string(xs[e]) for e in ids]
    if len(ids) != 10 or len(set(inputs)) != 10:
        raise ValueError('first ten IDs must have ten distinct inputs')
    # Only the initial output crosses the source-validation boundary.
    return dict(initial=[inputs[0], string(ys[ids[0]])], inputs=inputs[1:],
                example_ids=ids, source_metadata=metadata,
                extra_examples=metadata['examples']-10)


def summarize(results):
    completed = {t: r for t, r in results.items() if r['status']=='complete'}
    return dict(total_tasks=len(results), complete_tasks=len(completed),
                strict_h2_improvements=sum(r['expected_risk']['h2'] < r['expected_risk']['h1']-1e-12
                                           for r in completed.values()),
                strict_h3_improvements=sum(r['expected_risk']['h3'] < r['expected_risk']['h2']-1e-12
                                           for r in completed.values()),
                nonmonotonic_tasks=[t for t,r in completed.items()
                                   if r['expected_risk']['h2']>r['expected_risk']['h1']+1e-12
                                   or r['expected_risk']['h3']>r['expected_risk']['h2']+1e-12],
                full_cohort_means=({k:sum(r['expected_risk'][k] for r in completed.values())/len(results)
                                    for k in ('h1','h2','h3','random','openloop')}
                                   if len(completed)==len(results) and results else None))


def main():
    if ROOT.exists():
        raise ValueError('already opened')
    selection = json.loads(Path('results/nonmyopic/author_strings_overlap_20260909/selection.json').read_text())
    bindings = json.loads(Path('results/nonmyopic/author_strings_overlap_20260909/result.json').read_text())['records']
    if tuple(selection['ids']) != COHORT or choose(COHORT) != list(COHORT):
        raise ValueError('cohort binding mismatch')
    ROOT.mkdir()
    dsl = load()
    results = {}
    for i, task in enumerate(COHORT):
        try:
            base = f'https://huggingface.co/datasets/andrewcropper/ilp-datasets/resolve/{COMMIT}/strings/{task}/train/'
            bk = get(base+'bk.pl', 1000000)
            if hashlib.sha256(bk).hexdigest() != bindings[task]['bk_sha256']:
                raise ValueError('input binding mismatch')
            exs = get(base+'exs.pl', 1000000)
            public = public_case(bk.decode(), exs.decode())
            public['exs_sha256'] = hashlib.sha256(exs).hexdigest()
            save(ROOT/f'{task}.public.json', public)
            panel = support(dsl, public['initial'], public['inputs'], 51100000+i)
            save(ROOT/f'{task}.panel.json', panel)
            results[task] = measure(panel['rows'])
            results[task]['distinct_rows'] = panel['distinct_rows']
        except Exception as error:
            results[task] = dict(status='failed_closed', error_type=type(error).__name__)
        save(ROOT/f'{task}.result.json', results[task])
        print(task, results[task]['status'], flush=True)
    result = dict(results=results, summary=summarize(results), model_calls=0, cost_usd=0,
                  paid_authority=False, full_prior_reference=False,
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(ROOT/'result.json', result)
    print(json.dumps(result['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
