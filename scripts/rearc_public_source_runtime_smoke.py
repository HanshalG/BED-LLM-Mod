"""Actual isolated source-channel checks using synthetic functions, not tasks."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from scripts.rearc_public_source_journal import collect,replay,save
from scripts.rearc_public_source_transport import dispatch
from scripts.rearc_graph_runtime import IMAGE

SOURCE = '''
def generate_aaaaaa00(a,b):
    return {'input':((1,),),'output':((1,),)}
def verify_aaaaaa00(x): return x
def generate_aaaaaa01(a,b): raise ValueError('PRIVATE_PAYLOAD')
def verify_aaaaaa01(x): return x
def generate_aaaaaa02(a,b):
    return {'input':((1,),),'output':((2,),)}
def verify_aaaaaa02(x): return x
def generate_aaaaaa03(a,b):
    return {'input':((1,),),'output':((99,),)}
def verify_aaaaaa03(x): return x
def generate_aaaaaa04(a,b):
    while True: pass
def verify_aaaaaa04(x): return x
'''


def main():
    out=Path('results/nonmyopic/rearc_public_source_runtime_smoke_20260909')
    out.mkdir(exist_ok=False)
    names=('rearc_public_source_worker.py','rearc_graph_worker.py','rearc_public_source_transport.py','rearc_public_source_journal.py')
    save(out/'bindings.json',{'image':IMAGE,'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),
        'implementation':{n:hashlib.sha256(Path('scripts',n).read_bytes()).hexdigest() for n in names}})
    rows=[]
    try:
        with tempfile.TemporaryDirectory(prefix='bed-public-source-synthetic-') as tmp:
            root=Path(tmp)
            (root/'selected_source.py').write_text(SOURCE)
            for name in names[:2]:shutil.copyfile(Path('scripts',name),root/name)
            root.chmod(0o755)
            for p in root.iterdir():p.chmod(0o444)
            cases=[('input','aaaaaa00','input',None),('demo','aaaaaa00','demonstration',None),
                ('generate','aaaaaa01','input','generate'),('verify','aaaaaa02','input','verify'),
                ('grid','aaaaaa03','input','grid_validation'),('cpu','aaaaaa04','input','resource')]
            for name,task,mode,expected in cases:
                result=collect(out/name,[{'task':task,'seed':0,'mode':mode}],lambda r:dispatch(root,r))
                replay(out/name)
                assert result['status']==('public_schedule_complete' if expected is None else 'failed_closed')
                if expected is not None:
                    failure=json.loads((out/name/'000.failure.json').read_text())
                    if expected=='resource':assert failure.get('returncode',0)!=0 or failure['error_type']=='TimeoutExpired'
                    else:assert failure['worker_failure']['phase']==expected
                for file in (out/name).iterdir():assert 'PRIVATE_PAYLOAD' not in file.read_text()
                rows.append({'case':name,'status':'passed','new_model_calls':0})
                print(name,'passed',flush=True)
        save(out/'result.json',{'status':'passed','rows':rows,'benchmark_examples':0,'model_calls':0,'cost_usd':0})
    except Exception as error:
        save(out/'result.json',{'status':'failed','rows':rows,'error_type':type(error).__name__,'model_calls':0})
        raise


if __name__=='__main__':main()
