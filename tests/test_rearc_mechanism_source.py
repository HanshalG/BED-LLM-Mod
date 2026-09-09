import hashlib
import json
from pathlib import Path
import pytest
from scripts.rearc_mechanism_cohort import select
from scripts.rearc_mechanism_examples import MechanismExamples,SMOKE_SHA
from scripts.rearc_mechanism_source_smoke import COHORT_SHA

ROOT = Path(__file__).resolve().parents[1]/'results/nonmyopic'


def test_frozen_cohort_and_ordered_source_gate():
    raw = (ROOT/'REARC_MECHANISM_COHORT_20260909.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==COHORT_SHA
    cohort = json.loads(raw)
    inventory = json.loads((ROOT/'REARC_SOURCE_SCOPE_20260909.json').read_text())['all_ids']
    assert cohort['selected_ids']==select(inventory,cohort['excluded_ids'])
    assert len(cohort['selected_ids'])==6 and len(cohort['excluded_ids'])==20
    assert not set(cohort['selected_ids']) & set(cohort['excluded_ids'])
    raw = (ROOT/'REARC_MECHANISM_SOURCE_SMOKE_20260909.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SMOKE_SHA
    result = json.loads(raw)
    assert result['status']=='passed' and result['calls']==0
    assert [(r['task'],r['seed']) for r in result['rows']]==[
        (task,seed) for task in cohort['selected_ids'] for seed in cohort['source_seeds']]
    assert all(r['status']=='ok' and r['returncode']==0 for r in result['rows'])


def test_public_channels_and_combined_query_target_hashes():
    source = MechanismExamples()
    source.cohort = json.loads((ROOT/'REARC_MECHANISM_COHORT_20260909.json').read_text())
    source.tasks = source.cohort['selected_ids']
    calls = []
    def one(task,seed,mode):
        calls.append((task,seed,mode))
        if mode=='input':
            return {'input':[[seed%10]],'output_sha256':str(seed)}
        return {'input':[[0]],'output':[[0]],'output_sha256':'wrong'}
    source.one=one
    cases=source.public()
    assert len(calls)==66
    assert sum(mode=='demonstration' for _,_,mode in calls)==6
    assert sum(mode=='input' for _,_,mode in calls)==60
    assert not any(mode=='output' for _,_,mode in calls)
    for case in cases:
        assert len(case['outputs'])==1 and len(case['query_inputs'])==2 and len(case['target_inputs'])==8
        assert case['target_hashes']==[str(s) for s in source.cohort['query_seeds']+source.cohort['target_seeds']]
    with pytest.raises(ValueError,match='target identity'):
        source.targets(cases)
