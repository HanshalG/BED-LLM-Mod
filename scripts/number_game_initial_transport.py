"""Cross-generation diagnostic using only the banked initial-rule projection."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from scripts.number_game_initial_horizon_audit import ExactMembershipHorizon, digest, load_compiler

INPUT = Path('results/nonmyopic/number_game_initial_horizon_audit/20260908-v1/INITIAL_ONLY.json')
INPUT_SHA = '00109f2712fa303bbea6a8ae7c8ae4819d8843c7ee51fdd1e1405066ffd2efae'
COMPILER = Path('scripts/number_game_generator_aware_bed.py')
COMPILER_SHA = '1df1eab2d6e7dbbebb89e31d6c9863160d15ce519c8d9c55724de577a500a35a'
OUT = Path('results/nonmyopic/number_game_initial_transport_20260909')


def route(solver, truth, depth):
    mask, queries = solver.full, []
    for remaining in (3, 2, 1):
        query = solver.plan(mask, min(depth, remaining))[1]
        if query is None:
            query = next(q for q in range(solver.size) if q not in queries)
        queries.append(query)
        positives = mask & solver.columns[query]
        mask = positives if truth[query] else mask ^ positives
        if not mask:
            return None, len(queries)
    count = mask.bit_count()
    loss = sum((Fraction((mask & col).bit_count(), count)-int(y))**2
               for col, y in zip(solver.columns, truth))/solver.size
    return loss, None


def main():
    if digest(INPUT) != INPUT_SHA or digest(COMPILER) != COMPILER_SHA:
        raise ValueError('initial projection/compiler mismatch')
    OUT.mkdir(exist_ok=False)
    rows = json.loads(INPUT.read_text())
    if [r['tree_seed'] for r in rows] != list(range(28300, 28332)):
        raise ValueError('panel identity')
    compiler, supports = load_compiler(COMPILER), []
    for row in rows:
        support = []
        for rule in row['initial']:
            extension = compiler(rule['expression'])
            if hashlib.sha256(bytes(extension)).hexdigest() != rule['extension_sha256']:
                raise ValueError('extension mismatch')
            if extension not in support:
                support.append(extension)
        supports.append(support)
    result = {'initial_projection_sha256': INPUT_SHA, 'rows': [], 'model_calls': 0,
              'cost_usd': 0, 'future_supports_opened': False, 'historical_targets_opened': False}
    try:
        for index, support in enumerate(supports):
            solver = ExactMembershipHorizon(support)
            own = solver.compare()
            failure, loss = [Fraction(0) for _ in range(3)], [Fraction(0) for _ in range(3)]
            exact_covered = Fraction(0)
            for donor, truths in enumerate(supports):
                if donor == index:
                    continue
                weight = Fraction(1, (len(supports)-1)*len(truths))
                for truth in truths:
                    exact_covered += weight*int(truth in support)
                    for depth in (1, 2, 3):
                        value, failed_round = route(solver, truth, depth)
                        if failed_round is not None:
                            failure[depth-1] += weight
                        else:
                            loss[depth-1] += weight*value
            row = {'tree_seed': rows[index]['tree_seed'], 'own_prior_brier': own['full_budget_values'],
                   'exact_rule_coverage': float(exact_covered),
                   'failure_mass': list(map(float, failure)),
                   'unconditional_brier_lower_bound': list(map(float, loss)),
                   'unconditional_brier_upper_bound': [float(a+b) for a, b in zip(loss, failure)],
                   'successful_mass_brier': [float(a/(1-b)) if b < 1 else None for a,b in zip(loss,failure)]}
            result['rows'].append(row)
            (OUT/f'tree_{row["tree_seed"]}.json').write_text(json.dumps(row, indent=2)+'\n')
        result['status'] = 'complete'
        result['mean'] = {key: [sum(r[key][j] for r in result['rows'])/32 for j in range(3)]
                          for key in ('failure_mass', 'unconditional_brier_lower_bound', 'unconditional_brier_upper_bound')}
        result['mean_exact_rule_coverage'] = sum(r['exact_rule_coverage'] for r in result['rows'])/32
    except Exception as error:
        result.update(status='failed_closed', error=f'{type(error).__name__}: {error}')
    (OUT/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
