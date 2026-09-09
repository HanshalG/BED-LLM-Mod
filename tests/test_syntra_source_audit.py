import json

import pytest

from scripts.syntra_source_audit import summarize


def test_duplicate_inputs_and_no_label_dependency():
    row = dict(idx=0,name='fixture',train=[dict(input=str(i),output='SECRET') for i in range(3)],
               test=[dict(input='2',output='SECRET'),dict(input='3',output='SECRET')])
    a = summarize('playgol_v2.jsonl',json.dumps(row).encode())
    assert a['distinct_input_counts']=={'4':1}
    assert a['eligible_for_one_initial_one_target_four_candidates']==0
    for p in row['train']+row['test']:
        p['output']={'different':'not a label assertion'}
    b = summarize('playgol_v2.jsonl',json.dumps(row).encode())
    a.pop('sha256')
    b.pop('sha256')
    assert a==b and 'SECRET' not in json.dumps(a)


def test_source_schema_and_no_evaluation():
    row = dict(code='raise AssertionError()',idx=0,prompt='hidden',source_file='x',task_id=0,
               train='[1]',test='[1, 2]')
    result = summarize('mbpp_plus_51_cases.jsonl',json.dumps(row).encode())
    assert result['split_lengths']=={'train':{'1':1},'test':{'2':1}}
    row['test']='__import__("os").system("false")'
    with pytest.raises(ValueError):
        summarize('mbpp_plus_51_cases.jsonl',json.dumps(row).encode())
    with pytest.raises(ValueError):
        summarize('playgol_v2.jsonl',b'{}')
