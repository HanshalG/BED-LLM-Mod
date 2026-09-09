import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from scripts.rearc_public_source_journal import collect,replay


def raw():
    return json.dumps({'input':[[1]],'output_sha256':hashlib.sha256(b'[[9]]').hexdigest()}).encode()


@pytest.mark.parametrize('failure',['exit','oversized','hidden_output','timeout','duplicate','malformed',None])
def test_preserved_prefix_and_no_retry(tmp_path,failure):
    root=tmp_path/'bank'
    schedule=[{'task':'1234abcd','seed':i,'mode':'input'} for i in range(3)]
    calls=[]
    def dispatch(request):
        assert (root/f'{request["seed"]:03d}.request.json').exists()
        calls.append(request)
        data=raw()
        if request['seed']==1:
            if failure=='timeout': raise subprocess.TimeoutExpired('source',15,output=b'SECRET')
            if failure=='exit': return subprocess.CompletedProcess([],1,b'SECRET',b'SECRET')
            if failure=='oversized': data=b'X'*16385
            if failure=='hidden_output': data=b'{"input":[[1]],"output":[[987654321]],"output_sha256":"SECRET"}'
            if failure=='duplicate': data=b'{"input":[],"input":[]}'
            if failure=='malformed': data=b'SECRET'
        return subprocess.CompletedProcess([],0,data,b'')
    result=collect(root,schedule,dispatch)
    assert len(calls)==(3 if failure is None else 2)
    assert result['completed']==(3 if failure is None else 1)
    assert json.loads((root/'000.public.json').read_text())['input']==[[1]]
    assert replay(root)['new_source_calls']==0
    for file in root.iterdir():
        assert 'SECRET' not in file.read_text() and '987654321' not in file.read_text()
    with pytest.raises(FileExistsError): collect(root,schedule,dispatch)


def test_hidden_schedule_forbidden_before_dispatch(tmp_path):
    with pytest.raises(ValueError):
        collect(tmp_path/'bank',[{'task':'1234abcd','seed':0,'mode':'output'}],None)
    assert not (tmp_path/'bank').exists()


def test_public_tamper_rejected(tmp_path):
    root=tmp_path/'bank'
    collect(root,[{'task':'1234abcd','seed':0,'mode':'input'}],
        lambda request:subprocess.CompletedProcess([],0,raw(),b''))
    p=root/'000.public.json'
    value=json.loads(p.read_text())
    value['input']=[[2]]
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='public identity'):
        replay(root)
