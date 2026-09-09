from itertools import product
import math
import random
import pytest

from scripts.number_game_persistent_discrepancy import PersistentDiscrepancyBelief


def test_full_world_enumeration_matches_predictive_conditionals():
    rules = ((False, False, True), (True, False, False))
    belief = PersistentDiscrepancyBelief(rules, 1, 3)
    worlds = list(product((False, True), repeat=3))
    weights = []
    for world in worlds:
        mass = 0
        for rule in rules:
            e = sum(a != b for a, b in zip(rule, world))
            mass += math.exp(math.lgamma(1+e)+math.lgamma(6-e)-math.lgamma(7)
                             -(math.lgamma(1)+math.lgamma(3)-math.lgamma(4)))/2
        weights.append(mass)
    assert sum(weights) == pytest.approx(1)
    assert all(w > 0 for w in weights)
    for answer in (False, True):
        mass = sum(w for world, w in zip(worlds, weights) if world[0] == answer)
        expected = [sum(w*world[q] for world,w in zip(worlds,weights) if world[0] == answer)/mass
                    for q in range(3)]
        assert belief.observe(0, answer).predict((0,1,2)) == pytest.approx(expected)


def test_repeated_observation_is_not_new_noise_or_evidence():
    belief = PersistentDiscrepancyBelief(((False, False, False),), 1, 3).observe(0, True)
    assert belief.observe(0, True) == belief
    assert belief.predict((0,))[0] == 1
    rng = random.Random(8)
    assert all(belief.sample_world(rng)[0] for _ in range(100))
    with pytest.raises(ValueError, match='contradiction'):
        belief.observe(0, False)


def test_tower_identity_and_query_risk():
    belief = PersistentDiscrepancyBelief(((False, False, True), (True, True, False)), 1, 4)
    before = belief.predict((0,1,2))
    p = before[0]
    after = [belief.observe(0, y).predict((0,1,2)) for y in (False, True)]
    assert [(1-p)*a+p*b for a,b in zip(*after)] == pytest.approx(before)
    assert belief.expected_query_brier(0, (0,1,2)) <= sum(v*(1-v) for v in before)/3


def test_tempering_deterministic_likelihood_does_not_restore_support():
    for alpha in (.1, .5, 1):
        assert [0**alpha, 1**alpha] == [0, 1]
