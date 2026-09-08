import json

import pytest
from concept_synth.sexpr_parser import parse_sexpr_formula

from environments.relational_concepts import generative as g


def test_reproducible_scoped_concepts():
    for seed in range(64):
        a = g.sample_concept(seed)
        assert a == g.sample_concept(seed)
        assert parse_sexpr_formula(a.formula).free_vars() == {'x'}
        assert 1 <= a.structural_draws <= 64


def test_world_sampler_does_not_evaluate_concepts(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('world generation accessed semantics')
    monkeypatch.setattr(g, 'evaluate', forbidden)
    monkeypatch.setattr(g, 'sample_concept', forbidden)
    world = g.sample_world(1304)
    assert world == g.sample_world(1304)
    assert 7 <= world.size <= 13
    assert set(world.public_payload()) == {'domain', 'unary', 'binary'}
    assert 'formula' not in json.dumps(world.public_payload())


def test_upstream_semantics_hand_world():
    world = g.PublicWorld(2, (('P', (1,)), ('Q', ())), (('R', ((0, 1),)), ('S', ())))
    c = g.PrivateConcept('(exists y (and (R x y) (P y)))', 1)
    assert c.label(world, 0)
    assert not c.label(world, 1)
    with pytest.raises(ValueError):
        c.label(world, -1)


def test_support_count_matches_height_one_expansion():
    assert g.derivation_count(0, 1) == len(g.atoms(('x',))) == 4
    assert g.derivation_count(1, 1) == 4 + 4 + 2*4**2 + 2*len(g.atoms(('x', 'y')))
    assert g.derivation_count(4, 1) > 10**12


@pytest.mark.parametrize('seed', [-1, True, 1.2, '1'])
def test_seed_rejection(seed):
    with pytest.raises(ValueError):
        g.sample_concept(seed)
    with pytest.raises(ValueError):
        g.sample_world(seed)
