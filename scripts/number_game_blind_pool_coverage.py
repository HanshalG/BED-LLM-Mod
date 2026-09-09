"""Exact descriptive coverage of random blind pools on opened initial supports."""
from fractions import Fraction
from math import comb
import json
from pathlib import Path

from scripts.number_game_initial_transport import INPUT, INPUT_SHA
from scripts.number_game_initial_horizon_audit import digest


def inclusion_probability(population, occurrences, width):
    if not 0 <= occurrences <= population or not 1 <= width <= population:
        raise ValueError('invalid hypergeometric counts')
    misses = comb(population-occurrences, width) if width <= population-occurrences else 0
    return 1-Fraction(misses, comb(population, width))


def main():
    output = Path('results/nonmyopic/number_game_blind_pool_coverage_20260909.json')
    if output.exists():
        raise FileExistsError(output)
    assert digest(INPUT) == INPUT_SHA
    records = json.loads(INPUT.read_text())
    assert [r['tree_seed'] for r in records] == list(range(28300, 28332))
    supports = [set(rule['extension_sha256'] for rule in r['initial']) for r in records]
    rows = []
    for width in (1, 2, 4, 8, 16, 31):
        by_donor = []
        for index, support in enumerate(supports):
            counts = [sum(rule in other for j, other in enumerate(supports) if j != index)
                      for rule in support]
            by_donor.append(sum(inclusion_probability(31, n, width) for n in counts)/len(support))
        rows.append({'width': width, 'mean_coverage': float(sum(by_donor)/32),
                     'coverage_by_donor': list(map(float, by_donor))})
    parent = json.loads(Path('results/nonmyopic/number_game_initial_transport_20260909/RESULT.json').read_text())
    assert abs(rows[0]['mean_coverage']-parent['mean_exact_rule_coverage']) < 1e-12
    result = {'initial_sha256': INPUT_SHA, 'rows': rows, 'model_calls': 0, 'cost_usd': 0,
              'scope': 'opened correlated LLM generations; coverage only, not predictive validation'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k != 'coverage_by_donor'} for row in rows]))


if __name__ == '__main__':
    main()
