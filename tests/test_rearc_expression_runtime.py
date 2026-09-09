import json
from pathlib import Path
import pytest
from scripts.herb_search_runtime import search
from scripts.rearc_expression_examples import ExpressionExamples


def test_search_rejects_unfrozen_budget_before_container(monkeypatch):
    monkeypatch.setattr('scripts.herb_search_runtime.subprocess.run', lambda *args,**kwargs: pytest.fail('container opened'))
    with pytest.raises(ValueError, match='budget'):
        search([],64,50000)


def test_public_channel_schedule_and_hash_rejection():
    source = ExpressionExamples()
    source.tasks = ['a','b','c','d']
    source.cohort = {'demo_seeds':[34100,34101,34102], 'target_seeds':list(range(34200,34208))}
    calls = []
    def one(task, seed, mode):
        calls.append((task,seed,mode))
        if mode == 'input':
            return {'input':[[0]],'output_sha256':'sealed'}
        return {'input':[[0]],'output':[[0]],'output_sha256':'different'}
    source.one = one
    cases = source.public()
    assert len(cases) == 4 and len(calls) == 44
    assert sum(mode=='demonstration' for _,_,mode in calls) == 12
    assert sum(mode=='input' for _,_,mode in calls) == 32
    assert not any(mode=='output' for _,_,mode in calls)
    with pytest.raises(ValueError, match='target identity'):
        source.targets(cases)


def test_actual_symbolic_preflight():
    path = Path(__file__).resolve().parents[1] / 'results/nonmyopic/REARC_EXPRESSION_SYMBOLIC_PREFLIGHT_20260909.json'
    result = json.loads(path.read_text())
    assert result['status'] == 'complete' and len(result['expressions']) == 128
    assert result['expansions'] <= 100000 and result['no_api_key']
