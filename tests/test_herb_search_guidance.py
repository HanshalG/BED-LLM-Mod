import json
from pathlib import Path
import pytest
from scripts.herb_search_guidance import weights


def metadata():
    root = Path(__file__).resolve().parents[1] / 'results/nonmyopic/herb_full_grammar_20260909'
    data = json.loads((root / 'source.json').read_text())
    data['rules'] = [line.strip() for line in (root / 'grammar.jl').read_text().splitlines()
                     if line.strip().startswith('Value = ')]
    return data


def test_weights_preserve_all_rules_and_do_not_mutate_base():
    data = metadata()
    proposal = {'steps': [{'id': 'x0', 'op': 'vmirror', 'args': ['I']}], 'output': 'x0'}
    base, guide = weights(data, [proposal])
    assert len(base) == len(guide) == 353
    assert sum(base) == pytest.approx(1)
    assert sum(guide) == pytest.approx(1)
    assert all(g >= .5*b > 0 for b, g in zip(base, guide))
    index = data['rules'].index('Value = vmirror(Value)')
    assert guide[index] > base[index]
    assert weights(data, [])[1] == base


def test_invalid_guide_fails_instead_of_partial_salvage():
    with pytest.raises(ValueError):
        weights(metadata(), [{'steps': [{'id': 'x0', 'op': 'eval', 'args': ['I']}], 'output': 'x0'}])
