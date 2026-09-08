from fractions import Fraction

import numpy as np
import pytest

from environments.chembench_mopen.horizon import FiniteBeliefModel, HorizonPlanner

from scripts.relational_concept_opportunity import ParticleHorizon, compare


def test_duplicate_prior_mass_and_disjoint_targets():
    solver = ParticleHorizon([(False, False), (False, False), (True, True)], 1)
    assert solver.full.bit_count() == 3
    assert solver.risk(solver.full) == Fraction(2, 9)
    assert solver.plan(solver.full, 1)[0] == 0
    flat = ParticleHorizon([(False, True), (True, True)], 1)
    assert flat.risk(flat.full) == 0


def test_complete_tiny_comparison():
    result = compare([(False, False, False), (False, True, True),
                      (True, False, True), (True, True, False)], query_count=2, budget=2)
    assert all(v == 0 for v in result['values'].values())
    assert result['initial_risk'] == '1/4'


def test_constant_particles_are_not_removed():
    result = compare([(False, False, False)]*3+[(True, True, True)], query_count=2, budget=2)
    assert result['initial_risk'] == '3/16'
    assert result['values']['h3'] == 0


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_adapter_matches_independent_planner(seed):
    rows = (np.random.default_rng(seed).random((7, 7)) > .5).tolist()
    rows += [rows[0]]*2
    exact = ParticleHorizon(rows, 3)
    array = np.array(rows, dtype=float)
    model = FiniteBeliefModel(np.stack((1-array[:, :3], array[:, :3]), axis=-1),
                             array[:, 3:], np.full(len(rows), 1/len(rows)))
    planner = HorizonPlanner(model)
    for h in (1, 2, 3):
        assert float(exact.plan(exact.full, h)[0]) == pytest.approx(
            planner.plan(model.initial_state, h).root.expected_risk, abs=1e-14)
