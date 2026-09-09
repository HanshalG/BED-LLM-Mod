import json
import subprocess
import pytest
from scripts import rearc_public_source_transport as transport
from scripts.rearc_public_source_journal import collect


@pytest.mark.parametrize('timeout',[False,True])
def test_limits_and_cleanup(tmp_path,monkeypatch,timeout):
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if args[1]=='run':
            assert '--network=none' in args and '--read-only' in args
            assert '--memory=256m' in args and '--user=65534:65534' in args
            assert kwargs['timeout']==15
            if timeout: raise subprocess.TimeoutExpired(args,15)
        return subprocess.CompletedProcess(args,0,b'{}',b'')
    monkeypatch.setattr(transport.subprocess,'run',run)
    request={'task':'1234abcd','seed':0,'mode':'input'}
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired):transport.dispatch(tmp_path,request)
    else:
        transport.dispatch(tmp_path,request)
    assert calls[-1][1:3]==['rm','-f']


def test_sanitized_worker_error_record(tmp_path):
    failure={'status':'source_failed','phase':'verify','error_type':'ValueError'}
    collect(tmp_path/'bank',[{'task':'1234abcd','seed':0,'mode':'input'}],
        lambda r:subprocess.CompletedProcess([],1,json.dumps(failure).encode(),b''))
    saved=json.loads((tmp_path/'bank/000.failure.json').read_text())
    assert saved['worker_failure']==failure
