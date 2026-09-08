import pytest
import z3
import json
from concept_synth.sexpr_parser import parse_sexpr_formula
from scripts import induction_equivalence_audit as module
from scripts.induction_equivalence_audit import pair_status, translate


def compare(a, b):
    domain, predicates = z3.DeclareSort('TestObject'), {}
    return pair_status(translate(parse_sexpr_formula(a), domain, predicates),
                       translate(parse_sexpr_formula(b), domain, predicates), 1000)


@pytest.mark.parametrize('a,b', [
    ('(exists y (R x y))', '(exists z (R x z))'),
    ('(and (P x) (Q x))', '(and (Q x) (P x))'),
    ('(and (P x) (P x))', '(P x)'),
    ('(not (forall y (R x y)))', '(exists z (not (R x z)))'),
    ('(forall y (exists y (R x y)))', '(exists z (R x z))'),
])
def test_proven_equivalences(a, b):
    assert compare(a, b) == 'equivalent'


@pytest.mark.parametrize('a,b', [
    ('(P x)', '(Q x)'),
    ('(exists y (R x y))', '(forall y (R x y))'),
    ('(exists y (R x y))', '(exists y (R y x))'),
    ('(exists y (R x y))', '(exists x (R x x))'),
])
def test_binding_and_predicate_distinctions(a, b):
    assert compare(a, b) == 'distinguishable'


def test_unknown_never_merges_and_private_values_not_emitted(monkeypatch):
    rows = [dict(schemaVersion='induction_benchmark_record_v1', task='FullObs',
                 instanceId=f'private-id-{i}',
                 problemDescription=dict(hiddenTarget=dict(formula='(P x)' if i % 2 else '(Q x)')),
                 problem=dict(worlds=[dict(domain=['secret-object'], targetExtension='do-not-read')]))
            for i in range(375)]
    monkeypatch.setattr(module, 'pair_status', lambda *args: 'unknown')
    result = module.audit(rows)
    assert result['pair_status_counts'] == {'unknown': 1}
    assert result['proven_equivalence_components'] == 2
    assert not result['final_split_authorized']
    for private in ['private-id', 'secret-object', 'do-not-read', '(P x)', '(Q x)']:
        assert private not in json.dumps(result)
