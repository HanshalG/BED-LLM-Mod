import hashlib
import json
import pytest
from scripts import rearc_scene_source as source


def test_channel_mapping_and_second_answer_binding(monkeypatch):
    y=[[8]]
    h=hashlib.sha256(json.dumps(y,separators=(',',':')).encode()).hexdigest()
    c={'inputs':[[[0]]],'outputs':[[[1]]],'query_inputs':[[[2]],[[3]]],
       'target_inputs':[[[i]] for i in range(8)],'target_hashes':[h]+['future']*9}
    monkeypatch.setattr(source,'pool_cases',lambda p:[c]*4)
    cases=source.scene_cases(None,[y]*4)
    assert all(len(c['inputs'])==len(c['outputs'])==2 for c in cases)
    assert cases[0]['outputs']==[[[1]],[[8]]]
    assert cases[0]['query_inputs']==[[[3]],[[0]]]
    assert len(cases[0]['target_inputs'])==7
    assert cases[0]['target_hashes']==['future']*9
    with pytest.raises(ValueError,match='binding'):
        source.scene_cases(None,[[[9]]]*4)


def test_exact_source_schedule():
    c={'selected_ids':['task'],'demo_seeds':[52100],
       'query_seeds':[52200,52201,52202],'target_seeds':list(range(52300,52307))}
    rows=source.schedule(c)
    assert len(rows)==11
    assert [r['mode'] for r in rows]==['demonstration']+['input']*10
    assert [r['seed'] for r in rows]==[52100,52200,52201,52202]+list(range(52300,52307))
