"""Pinned isolated runtime for the declared symbolic comparator."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
from scripts.rearc_graph_runtime import IMAGE,DSL_SHA


def execute_symbolic(inputs,outputs):
    payload=json.dumps({'inputs':inputs,'outputs':outputs},allow_nan=False).encode()
    if len(payload)>65536:
        raise ValueError('request size')
    dsl=subprocess.check_output(['git','-C','/private/tmp/bed-rearc-source-audit','show',
        'e5b7f1d06362a76f9d3b8c25154ff1fafca897ce:dsl.py'],timeout=20)
    if hashlib.sha256(dsl).hexdigest()!=DSL_SHA:
        raise ValueError('source binding')
    name='bed-rearc-symbolic-'+uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='bed-rearc-symbolic-') as directory:
        root=Path(directory)
        (root/'dsl.py').write_bytes(dsl)
        (root/'scripts').mkdir()
        (root/'scripts/__init__.py').write_text('')
        for file in ('rearc_graph_worker.py','rearc_symbolic_beam.py','rearc_symbolic_worker.py'):
            shutil.copyfile(Path(__file__).with_name(file),root/'scripts'/file)
        root.chmod(0o755)
        (root/'scripts').chmod(0o755)
        for file in root.rglob('*.py'): file.chmod(0o444)
        args=['docker','run','--rm','-i','--name',name,'--network=none','--read-only',
              '--user=65534:65534','--cap-drop=ALL','--security-opt=no-new-privileges',
              '--pids-limit=32','--memory=512m','--memory-swap=512m','--cpus=1',
              '--mount',f'type=bind,src={root},dst=/app,readonly','--workdir=/app',
              '--env=PYTHONDONTWRITEBYTECODE=1',IMAGE,'python','-m','scripts.rearc_symbolic_worker']
        try:
            response=subprocess.run(args,input=payload,capture_output=True,timeout=60)
            if response.returncode:
                return {'status':'failed','returncode':response.returncode}
            if len(response.stdout)>131072:
                raise ValueError('response size')
            result=json.loads(response.stdout)
            if result['uid']!=65534 or not result['no_api_key']:
                raise ValueError('isolation')
            return result
        finally:
            subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)
