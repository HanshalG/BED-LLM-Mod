import hashlib
import json
from pathlib import Path
import pytest
from scripts.rearc_slot_cohort import select
from scripts.rearc_slot_examples import SlotExamples, SMOKE_SHA
from scripts.rearc_slot_source_smoke import COHORT_SHA

ROOT = Path(__file__).resolve().parents[1]/'results/nonmyopic'


def test_cohort_and_source_exact_bindings():
    raw = (ROOT/'REARC_SLOT_COHORT_20260909.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == COHORT_SHA
    cohort = json.loads(raw)
    inventory = json.loads((ROOT/'REARC_SOURCE_SCOPE_20260909.json').read_text())['all_ids']
    assert select(inventory, cohort['excluded_ids']) == cohort['selected_ids']
    assert not set(cohort['selected_ids']) & set(cohort['excluded_ids'])
    raw = (ROOT/'REARC_SLOT_SOURCE_SMOKE_20260909.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SMOKE_SHA
    smoke = json.loads(raw)
    assert smoke['status'] == 'passed' and smoke['calls'] == 0
    assert [(r['task'], r['seed']) for r in smoke['rows']] == [
        (task, seed) for task in cohort['selected_ids'] for seed in cohort['source_seeds']]
    assert all(r['status'] == 'ok' and r['returncode'] == 0 for r in smoke['rows'])


def test_source_channel_schedule_and_hash_rejection():
    source = SlotExamples()
    source.tasks = ['a', 'b', 'c', 'd']
    source.cohort = {'demo_seeds': [36100,36101,36102], 'target_seeds': list(range(36200,36208))}
    calls = []
    def one(task, seed, mode):
        calls.append((task, seed, mode))
        if mode == 'input':
            return {'input': [[0]], 'output_sha256': 'sealed'}
        return {'input': [[0]], 'output': [[0]], 'output_sha256': 'different'}
    source.one = one
    cases = source.public()
    assert len(calls) == 44
    assert sum(mode == 'demonstration' for _,_,mode in calls) == 12
    assert sum(mode == 'input' for _,_,mode in calls) == 32
    assert not any(mode == 'output' for _,_,mode in calls)
    with pytest.raises(ValueError, match='target identity'):
        source.targets(cases)
