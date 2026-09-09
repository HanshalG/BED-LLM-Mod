import subprocess
import pytest
from scripts.rearc_python_contract import validate
from scripts import rearc_python_runtime as runtime


@pytest.mark.parametrize('code',[
    'import os\ndef transform(g): return g',
    'def transform(g): return g.__class__',
    'def transform(g,x): return g',
    'def other(g): return g',
    'from collections import *\ndef transform(g): return g'])
def test_bad_contract(code):
    with pytest.raises(ValueError):validate(code)


def test_native_loops_and_helpers():
    validate('from collections import deque\ndef transform(g):\n q=deque()\n return [[v for v in row] for row in g]')


def test_host_never_executes_candidate_and_cleans_up(tmp_path,monkeypatch):
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if args[1]=='run':
            assert '--network=none' in args and '--read-only' in args and '--user=65534:65534' in args
            assert '--memory=256m' in args and kwargs['timeout']==15
            raise subprocess.TimeoutExpired(args,15)
        return subprocess.CompletedProcess(args,0,b'',b'')
    monkeypatch.setattr(runtime.subprocess,'run',run)
    code=f"open({str(tmp_path/'host-marker')!r},'w')\ndef transform(g): return g"
    assert runtime.execute(code,[[1]])['status']=='runtime_failed'
    assert not (tmp_path/'host-marker').exists()
    assert calls[-1][1:3]==['rm','-f']
