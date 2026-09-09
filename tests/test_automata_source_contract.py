import subprocess
import pytest
from scripts.automata_source_contract import SOURCE, COMMIT, load_sul, measure


def test_hash_failure_precedes_execution():
    with pytest.raises(ValueError, match='source changed'):
        load_sul(b'raise AssertionError("must not execute")')


def test_pinned_query_reset_and_measurement_contract():
    raw=subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':aalpy/base/SUL.py'])
    r=measure(load_sul(raw))
    assert r['whole_word']=={'outputs':['ack','armed'],'membership_queries':1,
        'reported_steps':2,'physical_steps':2,'resets':1}
    assert r['split_words']=={'outputs':[['ack'],['idle']],'membership_queries':2,
        'reported_steps':2,'physical_steps':2,'resets':2}
    assert r['interrupted_word']=={'membership_queries':0,'reported_steps':0,
        'physical_steps':2,'resets':1}
