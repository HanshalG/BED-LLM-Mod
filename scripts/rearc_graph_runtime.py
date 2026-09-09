"""Resource-bounded container wrapper; mounts only DSL, validator and worker."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

IMAGE = 'sha256:989bb9480c980a0e6722f366bf7dbf68bf6c097e66ed1fe450885989e5ef0b49'
DSL_SHA = 'c60faf5730a966fc1689c08bc96794cd97fd0311b8036e0f9b0c80e2d12f0f11'


def execute(graph, input_grid):
    payload = json.dumps({'graph': graph, 'input': input_grid}, allow_nan=False).encode()
    if len(payload) > 65536:
        raise ValueError('request size')
    dsl = subprocess.check_output(['git', '-C', '/private/tmp/bed-rearc-source-audit', 'show',
                                  'e5b7f1d06362a76f9d3b8c25154ff1fafca897ce:dsl.py'], timeout=20)
    if hashlib.sha256(dsl).hexdigest() != DSL_SHA:
        raise ValueError('DSL binding')
    name = 'bed-rearc-'+uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='bed-rearc-worker-') as directory:
        root = Path(directory)
        (root/'dsl.py').write_bytes(dsl)
        for file in ('rearc_program_graph.py', 'rearc_graph_worker.py'):
            shutil.copyfile(Path(__file__).with_name(file), root/file)
        root.chmod(0o755)
        for file in root.iterdir():
            file.chmod(0o444)
        args = ['docker', 'run', '--rm', '-i', '--name', name, '--network=none', '--read-only',
                '--user=65534:65534', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                '--pids-limit=32', '--memory=256m', '--memory-swap=256m', '--cpus=1',
                '--mount', f'type=bind,src={root},dst=/app,readonly',
                '--env=PYTHONDONTWRITEBYTECODE=1', IMAGE, 'python', '/app/rearc_graph_worker.py']
        try:
            response = subprocess.run(args, input=payload, capture_output=True, timeout=15)
            if len(response.stdout) > 16384:
                raise ValueError('response size')
            if response.returncode:
                return {'status': 'failed', 'returncode': response.returncode}
            value = json.loads(response.stdout)
            if value['status'] != 'ok' or value['uid'] != 65534 or not all(value[k] for k in ('read_only', 'network_denied', 'no_api_key')):
                raise ValueError('runtime isolation checks')
            return value
        finally:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=10)
