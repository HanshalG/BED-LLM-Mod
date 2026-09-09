"""Fresh nonroot/no-network container for each Python program and input."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
from scripts.rearc_graph_runtime import IMAGE
from scripts.rearc_graph_worker import grid
from scripts.rearc_python_contract import validate


def execute(code,input_grid):
    validate(code)  # Parse only on host; never execute candidate code here.
    grid(input_grid)
    payload=json.dumps({'code':code,'input':input_grid}).encode()
    if len(payload)>65536:raise ValueError('request size')
    name='bed-native-python-'+uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='bed-native-python-') as directory:
        root=Path(directory)
        for file in ('rearc_python_worker.py','rearc_python_contract.py','rearc_graph_worker.py'):
            shutil.copyfile(Path(__file__).with_name(file),root/file)
        root.chmod(0o755)
        for p in root.iterdir():p.chmod(0o444)
        args=['docker','run','--rm','-i','--name',name,'--network=none','--read-only',
            '--user=65534:65534','--cap-drop=ALL','--security-opt=no-new-privileges',
            '--pids-limit=32','--memory=256m','--memory-swap=256m','--cpus=1',
            '--env=PYTHONDONTWRITEBYTECODE=1','--env=PYTHONHASHSEED=0',
            '--mount',f'type=bind,src={root},dst=/app,readonly',IMAGE,'python','/app/rearc_python_worker.py']
        try:
            response=subprocess.run(args,input=payload,capture_output=True,timeout=15)
            if response.returncode:
                return {'status':'runtime_failed','returncode':response.returncode}
            if len(response.stdout)>16384:raise ValueError('response size')
            value=json.loads(response.stdout)
            if value.get('status')=='ok' and set(value)=={'status','output'}:
                grid(value['output'])
            elif value.get('status')!='failed' or set(value)!={'status','phase','error_type','candidate_line'}:
                raise ValueError('response envelope')
            return value
        except subprocess.TimeoutExpired:
            return {'status':'runtime_failed','reason':'wall_timeout'}
        finally:
            subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)
