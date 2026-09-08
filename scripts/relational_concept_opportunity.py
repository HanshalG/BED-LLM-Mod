"""Complete exact finite-particle opportunity pilot for the frozen new grammar."""
import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from time import monotonic

from concept_synth.sexpr_parser import parse_sexpr_formula
from concept_synth.fol.model import evaluate as evaluate_formula

from environments.relational_concepts.generative import sample_concept, sample_world
from scripts.semantic_scene_opportunity import MenuHorizon
from scripts.number_game_initial_openloop_audit import OpenLoopMembership


class ParticleHorizon(MenuHorizon):
    def __init__(self, extensions, query_count, **kwargs):
        rows = [tuple(row) for row in extensions]
        super().__init__(list(dict.fromkeys(rows)), range(query_count), **kwargs)
        if not 0 < query_count < self.size:
            raise ValueError('disjoint nonempty query and target columns required')
        # Preserve each prior draw as one bit, including identical functions.
        self.full = (1 << len(rows)) - 1
        self.columns = [sum(int(row[q]) << i for i, row in enumerate(rows))
                        for q in range(self.size)]
        self.target_columns = self.columns[query_count:]

    def risk(self, mask):
        if mask not in self.risks:
            self.check()
            n = mask.bit_count()
            if not n:
                raise ValueError('empty posterior')
            counts = [(mask & col).bit_count() for col in self.target_columns]
            self.risks[mask] = Fraction(sum(k*(n-k) for k in counts),
                                        n*n*len(self.target_columns))
        return self.risks[mask]


class RecedingOpenLoop(OpenLoopMembership):
    @lru_cache(maxsize=None)
    def deployed(self, mask, budget, horizon):
        self.solver.check()
        if not budget:
            return self.solver.risk(mask)
        _, sequence = self.plan(mask, min(budget, horizon))
        if not sequence:
            return self.solver.risk(mask)
        positive = mask & self.solver.columns[sequence[0]]
        return sum((Fraction(child.bit_count(), mask.bit_count())
                    * self.deployed(child, budget-1, horizon)
                    for child in (positive, mask ^ positive) if child), Fraction(0))


def compare(rows, query_count=8, budget=4):
    solver = ParticleHorizon(rows, query_count, max_seconds=5, max_states=100000)
    values = {f'h{h}': solver.deployed(solver.full, budget, h) for h in (1, 2, 3)}
    control = RecedingOpenLoop(solver, max_sets=100000)
    values['receding_openloop_h3'] = control.deployed(solver.full, budget, 3)
    values['random'] = solver.random_policy(solver.full, tuple(range(query_count)), budget)
    # Exact one-step optimization is the saturated myopic computation control.
    values['exact_myopic_control'] = values['h1']
    return dict(exact_values={k: str(v) for k, v in values.items()},
                values={k: float(v) for k, v in values.items()},
                first_queries={f'h{h}': solver.plan(solver.full, h)[1] for h in (1, 2, 3)},
                initial_risk=str(solver.risk(solver.full)),
                planner_seconds=monotonic()-solver.started,
                cached_risks=len(solver.risks), cached_plans=len(solver.plans),
                openloop_sets=control.sets)


def panel(index):
    started = monotonic()
    models = [sample_world(2000000 + index*1000 + i).to_model() for i in range(72)]
    rows = []
    for i in range(512):
        formula = parse_sexpr_formula(sample_concept(1000000 + index*1000 + i).formula)
        row = []
        for model in models:
            if monotonic()-started > 120:
                raise TimeoutError('likelihood construction cap')
            row.append(bool(evaluate_formula(formula, model, {'x': 0})))
        rows.append(tuple(row))
    unique = len(set(rows))
    constant = sum(len(set(row)) == 1 for row in rows)
    result = compare(rows)
    result.update(panel=index, prior_draws=512, distinct_observed_extensions=unique,
                  constant_on_all_72_worlds=constant,
                  extension_sha256=hashlib.sha256(b''.join(bytes(row) for row in rows)).hexdigest(),
                  elapsed_seconds=monotonic()-started)
    return result


def run(output):
    path = Path(output)
    if path.exists():
        raise FileExistsError('result already banked')
    report = dict(status='incomplete', rows=[], model_calls=0, paid_cost_usd=0,
                  paid_calls_authorized=False, whole_grammar_accuracy_verified=False)
    # Reserve the result before any outcome computation; a crash cannot reopen it.
    with path.open('x') as stream:
        json.dump(report, stream)
    try:
        for index in range(4):
            report['rows'].append(panel(index))
            path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
        means = {key: sum((Fraction(r['exact_values'][key]) for r in report['rows']), Fraction(0))/4
                 for key in report['rows'][0]['exact_values']}
        ordered = sum(Fraction(r['exact_values']['h3']) <= Fraction(r['exact_values']['h2'])
                      <= Fraction(r['exact_values']['h1']) for r in report['rows'])
        passed = (means['h1'] > 0 and means['h2'] > 0
                  and means['h2'] <= Fraction(95, 100)*means['h1']
                  and means['h3'] <= Fraction(95, 100)*means['h2']
                  and means['h3'] < means['receding_openloop_h3'] and ordered >= 3)
        report.update(status='finite_reference_opportunity_pass' if passed else 'finite_reference_opportunity_null',
                      exact_means={k: str(v) for k, v in means.items()},
                      means={k: float(v) for k, v in means.items()}, ordered_panels=ordered)
    except Exception as exc:
        report.update(status='failed_closed', error_type=type(exc).__name__)
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    if report['status'] == 'failed_closed':
        raise RuntimeError('banked pilot failure; no rerun')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
