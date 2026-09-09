"""Zero-model, pre-fix Black runtime qualification; no fixed source or test bank."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from scripts.bugsinpy_contract_audit import fetch

REVISION = '026c81b83454f176a9f9253cbfb70be2c159d822'
OUTPUT = Path('results/nonmyopic/BLACK_PREFIX_RUNTIME_SMOKE_20260909.json')
IMAGE = 'bed-black-prefixed-smoke:20260909'
BASE = 'python:3.8.20-slim-bookworm'
SMOKE = '''import json, os, socket
import black
checks = []
for source, expected in [('x=1\\n', 'x = 1\\n'), ('f( 1,2 )\\n', 'f(1, 2)\\n')]:
    output = black.format_file_contents(source, fast=False, mode=black.FileMode())
    checks.append({'source': source, 'output': output, 'expected': expected, 'ok': output == expected})
try:
    black.format_file_contents('def :\\n', fast=False, mode=black.FileMode())
    invalid = 'accepted'
except black.InvalidInput:
    invalid = 'InvalidInput'
try:
    open('/write-probe', 'w').close()
    write_denied = False
except OSError:
    write_denied = True
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(('1.1.1.1', 443))
    network_denied = False
except OSError:
    network_denied = True
finally:
    sock.close()
no_key = 'OPENROUTER_API_KEY' not in os.environ
ok = all(c['ok'] for c in checks) and invalid == 'InvalidInput' and write_denied and network_denied and no_key
print(json.dumps(dict(status='passed' if ok else 'failed', checks=checks,
    invalid_input=invalid, write_denied=write_denied, network_denied=network_denied,
    no_api_key=no_key, uid=os.getuid(), black_version=black.__version__)))
raise SystemExit(0 if ok else 1)
'''


def source_paths(tree):
    if tree.get('truncated') is not False:
        raise ValueError('truncated tree')
    paths = []
    for row in tree['tree']:
        p = row['path']
        if row['type'] != 'blob':
            continue
        if p in ('black.py', 'LICENSE') or (p.startswith('blib2to3/') and
                                            p.endswith(('.py', '.txt'))):
            if '..' in Path(p).parts or Path(p).is_absolute():
                raise ValueError('unsafe source path')
            paths.append(p)
    if 'black.py' not in paths or 'LICENSE' not in paths:
        raise ValueError('missing source')
    return sorted(paths)


def command(args, timeout):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout).stdout


def run():
    if OUTPUT.exists():
        raise RuntimeError('already banked')
    result = {'revision': REVISION, 'model_calls': 0, 'cost_usd': 0,
              'fixed_source_read': False, 'upstream_tests_read': False}
    try:
        base = json.loads(command(['docker', 'image', 'inspect', BASE], 20))[0]
        digest = base['RepoDigests'][0]
        result['base_digest'] = digest
        tree_raw = fetch(f'https://api.github.com/repos/psf/black/git/trees/{REVISION}?recursive=1')
        paths = source_paths(json.loads(tree_raw))
        result['tree_sha256'] = hashlib.sha256(tree_raw).hexdigest()
        result['source_hashes'] = {}
        with tempfile.TemporaryDirectory(prefix='bed-black-prefix-') as directory:
            root = Path(directory)
            for path in paths:
                raw = fetch(f'https://raw.githubusercontent.com/psf/black/{REVISION}/{path}')
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                result['source_hashes'][path] = hashlib.sha256(raw).hexdigest()
            if result['source_hashes']['black.py'] != 'a2a3a17242908ce2db5c562eb2ef36c15e9f1fa5fd0ecedac1d978181119c55e':
                raise ValueError('source binding changed')
            (root / 'smoke.py').write_text(SMOKE)
            dockerfile = f'''FROM {digest}
RUN pip install --no-cache-dir --only-binary=:all: click==7.1.2 attrs==19.3.0 appdirs==1.4.4 toml==0.10.2
WORKDIR /app
COPY . /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app HOME=/tmp
USER 65534:65534
CMD ["python", "/app/smoke.py"]
'''
            (root / 'Dockerfile').write_text(dockerfile)
            result['dockerfile'] = dockerfile
            command(['docker', 'build', '-t', IMAGE, str(root)], 240)
            result['image_id'] = json.loads(command(['docker', 'image', 'inspect', IMAGE], 20))[0]['Id']
            args = ['docker', 'run', '--rm', '--network=none', '--read-only', '--cap-drop=ALL',
                    '--security-opt=no-new-privileges', '--pids-limit=32', '--memory=256m',
                    '--cpus=1', '--tmpfs=/tmp:rw,noexec,nosuid,size=16m', IMAGE]
            result['container_args'] = args
            result['smoke'] = json.loads(command(args, 30))
            result['status'] = result['smoke']['status']
    except Exception as exc:
        result['status'] = 'failed_closed'
        result['error_type'] = type(exc).__name__
        result['error'] = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            result['stderr'] = exc.stderr[-5000:]
            result['stdout'] = exc.stdout[-5000:]
    with OUTPUT.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
