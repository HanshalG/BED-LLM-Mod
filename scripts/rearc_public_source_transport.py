"""Bounded public-source container transport; caller owns its frozen source mount."""
import json
from pathlib import Path
import subprocess
import uuid
from scripts.rearc_graph_runtime import IMAGE
from scripts.rearc_public_source_journal import validate_schedule


def dispatch(root, request):
    validate_schedule([request])
    root=Path(root).resolve(strict=True)
    name='bed-public-source-'+uuid.uuid4().hex
    args=['docker','run','--rm','-i','--name',name,'--network=none','--read-only',
        '--user=65534:65534','--cap-drop=ALL','--security-opt=no-new-privileges',
        '--pids-limit=32','--memory=256m','--memory-swap=256m','--cpus=1',
        '--mount',f'type=bind,src={root},dst=/app,readonly',
        '--env=PYTHONDONTWRITEBYTECODE=1','--env=PYTHONHASHSEED=0',IMAGE,
        'python','/app/rearc_public_source_worker.py']
    try:
        return subprocess.run(args,input=json.dumps(request).encode(),capture_output=True,timeout=15)
    finally:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)
