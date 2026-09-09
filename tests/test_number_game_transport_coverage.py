from fractions import Fraction
import json
from pathlib import Path

from scripts.number_game_transport_coverage import decompose


def test_coverage_is_truth_identity_not_successful_conditioning():
    support = [(False, False, False, False)]
    groups = decompose(support, [[support[0], (False, False, False, True)]])
    assert groups['covered']['mass'] == groups['uncovered']['mass'] == Fraction(1, 2)
    assert groups['covered']['loss'] == [0]*3
    assert groups['uncovered']['failure'] == [0]*3
    assert groups['uncovered']['loss'] == [Fraction(1, 8)]*3


def test_donor_weighting_and_failure_not_dropped():
    truth = (False, False, False, False)
    groups = decompose([truth], [[truth], [truth, (True, True, True, True)]])
    assert groups['covered']['mass'] == Fraction(3, 4)
    assert groups['uncovered']['mass'] == Fraction(1, 4)
    assert groups['uncovered']['failure'] == [Fraction(1, 4)]*3


def test_banked_strata_reconstruct_every_parent_row():
    result = json.loads(Path('results/nonmyopic/number_game_transport_coverage_20260909.json').read_text())
    parent = json.loads(Path('results/nonmyopic/number_game_initial_transport_20260909/RESULT.json').read_text())
    assert result['status'] == 'complete' and len(result['rows']) == 32
    for row, old in zip(result['rows'], parent['rows']):
        assert row['seed'] == old['tree_seed']
        assert abs(sum(g['mass'] for g in row['groups'].values())-1) < 1e-12
        for j in range(3):
            for field, reference in [('loss', 'unconditional_brier_lower_bound'), ('failure', 'failure_mass')]:
                assert abs(sum(g[field][j] for g in row['groups'].values())-old[reference][j]) < 1e-12
