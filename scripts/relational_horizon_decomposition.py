"""Retrospective exact root/continuation diagnosis on hash-verified banked worlds."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import monotonic

from concept_synth.sexpr_parser import parse_sexpr_formula
from concept_synth.fol.model import evaluate
from environments.relational_concepts.generative import sample_concept, sample_world
from scripts.relational_concept_opportunity import ParticleHorizon

BANK = Path('results/nonmyopic/RELATIONAL_CONCEPT_OPPORTUNITY_20260908.json')
BANK_SHA = '86b5ad180d2f5a3e4dc6e13907a4827d2eda4cb166e88281000b7fa9f61786f3'


def decompose(extensions, banked, query_count=8, budget=4):
    solver = ParticleHorizon(extensions, query_count, max_seconds=5, max_states=100000)
    optimum, optimal_query = solver.plan(solver.full, budget)
    roots = {h: [solver.action_value(solver.full, h, q) for q in range(query_count)]
             for h in range(1, budget+1)}
    parts = {}
    for horizon in (1, 2, 3):
        name = f'h{horizon}'
        query = banked['first_queries'][name]
        expected_query = solver.plan(solver.full, min(horizon, budget))[1]
        if query != expected_query:
            raise ValueError('banked root action does not replay')
        achieved = Fraction(banked['exact_values'][name])
        root_best = optimum if query is None else roots[budget][query]
        root_regret = root_best-optimum
        continuation_excess = achieved-root_best
        if root_regret < 0 or continuation_excess < 0:
            raise ValueError('negative optimality decomposition')
        if horizon >= budget-1 and continuation_excess:
            raise ValueError('full remaining-horizon continuation must be optimal')
        if root_regret+continuation_excess != achieved-optimum:
            raise ArithmeticError('decomposition identity failed')
        parts[name] = dict(first_query=query, exact_achieved=str(achieved),
                           exact_root_regret=str(root_regret),
                           exact_continuation_excess=str(continuation_excess),
                           root_regret=float(root_regret),
                           continuation_excess=float(continuation_excess))
    return dict(exact_optimal_budget_risk=str(optimum), optimal_budget_risk=float(optimum),
                optimal_first_query=optimal_query, decomposition=parts,
                exact_root_values={f'h{h}': [str(v) for v in values] for h, values in roots.items()},
                reference_seconds=monotonic()-solver.started)


def replay_extensions(banked):
    index = banked['panel']
    started = monotonic()
    models = [sample_world(2000000+index*1000+i).to_model() for i in range(72)]
    rows = []
    for i in range(512):
        formula = parse_sexpr_formula(sample_concept(1000000+index*1000+i).formula)
        values = []
        for model in models:
            if monotonic()-started > 120:
                raise TimeoutError('reconstruction time cap')
            values.append(bool(evaluate(formula, model, {'x': 0})))
        rows.append(tuple(values))
    digest = hashlib.sha256(b''.join(bytes(row) for row in rows)).hexdigest()
    if digest != banked['extension_sha256']:
        raise ValueError('banked likelihood matrix mismatch')
    return rows


def run(output):
    path = Path(output)
    if path.exists():
        raise FileExistsError('diagnostic already banked')
    raw = BANK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BANK_SHA:
        raise ValueError('bank binding mismatch')
    banked = json.loads(raw)
    if banked['status'] != 'finite_reference_opportunity_null' or [r['panel'] for r in banked['rows']] != list(range(4)):
        raise ValueError('unexpected banked panel')
    report = dict(status='incomplete', bank_sha256=BANK_SHA, rows=[],
                  retrospective=True, original_status=banked['status'],
                  paid_calls_authorized=False, model_calls=0, paid_cost_usd=0)
    with path.open('x') as stream:
        json.dump(report, stream)
    try:
        for row in banked['rows']:
            extensions = replay_extensions(row)
            result = decompose(extensions, row)
            result.update(panel=row['panel'], verified_extension_sha256=row['extension_sha256'])
            report['rows'].append(result)
            path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
        report['status'] = 'retrospective_decomposition_complete'
    except Exception as exc:
        report.update(status='failed_closed', error_type=type(exc).__name__)
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    if report['status'] == 'failed_closed':
        raise RuntimeError('diagnostic failed closed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
