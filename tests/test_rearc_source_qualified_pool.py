import hashlib
import json
import subprocess
import pytest
from scripts.rearc_source_qualified_pool import qualify,replay_pool,public_cases


def schedules(n):
    return [[{'task':f'{i:08x}','seed':j,'mode':'demonstration' if j==0 else 'input'}
             for j in range(11)] for i in range(n)]


def response(request):
    value={'input':[[1]],'output_sha256':hashlib.sha256(b'[[1]]').hexdigest()}
    if request['mode']=='demonstration':value['output']=[[1]]
    return subprocess.CompletedProcess([],0,json.dumps(value).encode(),b'')


def test_select_first_valid_bounded_and_replay_without_dispatch(tmp_path):
    calls=[]
    def dispatch(r):
        calls.append(r)
        if r['task']=='00000000' and r['seed']==2:
            return subprocess.CompletedProcess([],1,json.dumps({'status':'source_failed',
                'phase':'verify','error_type':'ValueError'}).encode(),b'')
        return response(r)
    root=tmp_path/'pool'
    result=qualify(root,schedules(4),dispatch,needed=2)
    assert result['status']=='source_pool_qualified' and len(calls)==25
    assert [r['task'] for r in result['accepted']]==['00000001','00000002']
    assert len(result['rejected'])==1 and replay_pool(root)==result
    assert len(public_cases(root))==2 and len(calls)==25
    with pytest.raises(FileExistsError):qualify(root,schedules(4),dispatch,needed=2)
    assert len(calls)==25
    (root/'03').mkdir()
    with pytest.raises(ValueError,match='unexpected'):replay_pool(root)


@pytest.mark.parametrize('kind',['timeout','generate','malformed'])
def test_nonreference_failure_stops_pool(tmp_path,kind):
    calls=[]
    def dispatch(r):
        calls.append(r)
        if kind=='timeout':raise subprocess.TimeoutExpired('worker',15)
        raw=b'bad' if kind=='malformed' else json.dumps({'status':'source_failed',
            'phase':'generate','error_type':'ValueError'}).encode()
        return subprocess.CompletedProcess([],1,raw,b'')
    root=tmp_path/'pool'
    result=qualify(root,schedules(3),dispatch,needed=2)
    assert result['status']=='failed_closed' and len(calls)==1
    assert replay_pool(root)==result
    with pytest.raises(ValueError,match='closed'):public_cases(root)


def test_exhaustion_is_not_qualification(tmp_path):
    def dispatch(r):
        return subprocess.CompletedProcess([],1,json.dumps({'status':'source_failed',
            'phase':'verify','error_type':'ValueError'}).encode(),b'')
    root=tmp_path/'pool'
    result=qualify(root,schedules(2),dispatch,needed=2)
    assert result['status']=='insufficient_source_valid_tasks'
    assert replay_pool(root)==result


def test_invalid_manifest_makes_no_directory(tmp_path):
    root=tmp_path/'pool'
    duplicate=schedules(2)
    duplicate[1]=duplicate[0]
    with pytest.raises(ValueError):qualify(root,duplicate,None,needed=2)
    assert not root.exists()


def test_prospective_candidate_manifest_matches_metadata_rule():
    from scripts.rearc_qualified_representation_source import COHORT,BASE,schedule
    cohort=json.loads(COHORT.read_text())
    previous=json.loads((BASE/'REARC_REPRESENTATION_COHORT_20260909.json').read_text())
    inventory=json.loads((BASE/'REARC_SOURCE_SCOPE_20260909.json').read_text())['all_ids']
    excluded=set(previous['excluded_ids']+previous['selected_ids'])
    assert len(excluded)==50 and cohort['excluded_ids']==sorted(excluded)
    expected=sorted(set(inventory)-excluded,
        key=lambda k:hashlib.sha256(('bed-rearc-source-qualified-v1:'+k).encode()).digest())[:24]
    assert cohort['selected_ids']==expected
    rows=schedule(cohort)
    assert len(rows)==264
    assert [r['seed'] for r in rows[:11]]==[48100,48200,48201]+list(range(48300,48308))
