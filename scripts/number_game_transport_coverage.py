"""Retrospective coverage decomposition, with unchanged transported policies."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from scripts.number_game_initial_transport import (
    INPUT, INPUT_SHA, COMPILER, COMPILER_SHA, OUT as PARENT, route,
)
from scripts.number_game_initial_horizon_audit import ExactMembershipHorizon, digest, load_compiler

OUT = Path('results/nonmyopic/number_game_transport_coverage_20260909.json')


def decompose(support, donors):
    solver = ExactMembershipHorizon(support)
    groups = {name: {'mass': Fraction(0), 'loss': [Fraction(0)]*3,
                     'failure': [Fraction(0)]*3} for name in ('covered', 'uncovered')}
    for donor in donors:
        weight = Fraction(1, len(donors)*len(donor))
        for truth in donor:
            group = groups['covered' if truth in support else 'uncovered']
            group['mass'] += weight
            for j in range(3):
                value, failure = route(solver, truth, j+1)
                if failure is None:
                    group['loss'][j] += weight*value
                else:
                    group['failure'][j] += weight
    return groups


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    assert digest(INPUT) == INPUT_SHA and digest(COMPILER) == COMPILER_SHA
    records = json.loads(INPUT.read_text())
    assert [r['tree_seed'] for r in records] == list(range(28300, 28332))
    compiler, supports = load_compiler(COMPILER), []
    for record in records:
        support = []
        for rule in record['initial']:
            extension = compiler(rule['expression'])
            assert hashlib.sha256(bytes(extension)).hexdigest() == rule['extension_sha256']
            if extension not in support:
                support.append(extension)
        supports.append(support)
    parent = json.loads((PARENT/'RESULT.json').read_text())
    result = {'status': 'complete', 'cost_usd': 0, 'model_calls': 0,
              'initial_sha256': INPUT_SHA, 'parent_sha256': digest(PARENT/'RESULT.json'),
              'interpretation': 'Retrospective strata; not causal or held-out validation', 'rows': []}
    for i, support in enumerate(supports):
        groups = decompose(support, supports[:i]+supports[i+1:])
        reference = parent['rows'][i]
        assert reference['tree_seed'] == records[i]['tree_seed']
        for j in range(3):
            for field, old in [('loss', 'unconditional_brier_lower_bound'), ('failure', 'failure_mass')]:
                assert abs(float(sum(g[field][j] for g in groups.values()))-reference[old][j]) < 1e-12
        assert groups['covered']['failure'] == [0]*3
        result['rows'].append({'seed': records[i]['tree_seed'], 'groups': {
            name: {key: list(map(float, value)) if isinstance(value, list) else float(value)
                   for key, value in group.items()} for name, group in groups.items()}})
    result['aggregate'] = {}
    for name in ('covered', 'uncovered'):
        groups = [r['groups'][name] for r in result['rows']]
        mass = sum(g['mass'] for g in groups)/32
        loss = [sum(g['loss'][j] for g in groups)/32 for j in range(3)]
        failure = [sum(g['failure'][j] for g in groups)/32 for j in range(3)]
        result['aggregate'][name] = {'mass': mass, 'loss_contribution': loss,
            'conditional_brier_lower': [v/mass for v in loss],
            'conditional_brier_upper': [(v+f)/mass for v, f in zip(loss, failure)],
            'h3_minus_h2_contribution': loss[2]-loss[1]}
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['aggregate'], indent=2))


if __name__ == '__main__':
    main()
