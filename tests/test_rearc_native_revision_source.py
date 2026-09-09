import hashlib
import json
import pytest
from scripts import rearc_native_revision_source as source


def test_exact_candidate_rule_and_seeds():
    c=json.loads(source.COHORT.read_text())
    old=json.loads((source.BASE/'REARC_QUALIFIED_REPRESENTATION_CANDIDATES_20260909.json').read_text())
    excluded=set(old['excluded_ids']+old['selected_ids'])
    inventory=json.loads((source.BASE/'REARC_SOURCE_SCOPE_20260909.json').read_text())['all_ids']
    assert len(excluded)==74 and sorted(excluded)==c['excluded_ids']
    expected=sorted(set(inventory)-excluded,key=lambda k:hashlib.sha256(('bed-rearc-native-revision-v1:'+k).encode()).digest())[:24]
    assert c['selected_ids']==expected
    rows=source.schedule(c)
    assert len(rows)==264
    assert [r['seed'] for r in rows[:11]]==[50100,50200,50201,50202]+list(range(50300,50307))


def test_public_channel_mapping_never_opens_second_output(monkeypatch):
    original={'inputs':[[[0]]],'outputs':[[[9]]],'query_inputs':[[[1]],[[2]]],
              'target_inputs':[[[i]] for i in range(3,11)],'target_hashes':list(range(10))}
    monkeypatch.setattr(source,'pool_cases',lambda p:[original])
    c,=source.revision_cases(None)
    assert c['reveal_input']==[[1]] and c['reveal_hash']==0
    assert c['query_inputs']==[[[2]],[[3]]] and len(c['target_inputs'])==7
    assert c['target_hashes']==list(range(1,10))
    assert 'reveal_output' not in c


def test_endpoints_reject_candidate_list_before_execution():
    s=object.__new__(source.RevisionExamples)
    s.tasks=['reserved']*24
    with pytest.raises(ValueError):s.reveal(0)
    with pytest.raises(ValueError):s.targets([{}]*4)
