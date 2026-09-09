"""Frozen all-task source checks, one attempt per task/seed, no replacement."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
from scripts.rearc_graph_runtime import IMAGE, DSL_SHA
from scripts.rearc_source_scope import COMMIT, SOURCE

SEEDS = (31000, 31001, 31002, 31003)
OUT = Path('results/nonmyopic/REARC_SOURCE_SMOKE_20260909.json')


def selected_functions(source, names):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError('missing selected source function')
    return '\n\n'.join(ast.get_source_segment(source, node) for node in functions)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    scope = json.loads(Path('results/nonmyopic/REARC_SOURCE_SCOPE_20260909.json').read_text())
    assert scope['source_commit'] == COMMIT
    files = {name: subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':'+name], timeout=20)
             for name in ('dsl.py','utils.py','generators.py','verifiers.py')}
    assert hashlib.sha256(files['dsl.py']).hexdigest() == DSL_SHA
    assert hashlib.sha256(files['verifiers.py']).hexdigest() == scope['verifiers_sha256']
    module = 'from dsl import *\nfrom random import choice, randint, sample, shuffle, uniform\nrng = []\n'
    module += selected_functions(files['utils.py'].decode(), {'unifint'})+'\n'
    for filename, prefix in (('generators.py','generate_'),('verifiers.py','verify_')):
        module += selected_functions(files[filename].decode(), {prefix+key for key in scope['selected_ids']})+'\n'
    result = {'source_commit':COMMIT,'image':IMAGE,'seeds':SEEDS,'rows':[],
              'source_hashes':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()},
              'selected_module_sha256':hashlib.sha256(module.encode()).hexdigest(),
              'model_calls':0,'cost_usd':0,'examples_emitted':False}
    with tempfile.TemporaryDirectory(prefix='bed-rearc-source-worker-') as directory:
        root = Path(directory)
        (root/'dsl.py').write_bytes(files['dsl.py'])
        (root/'selected_source.py').write_text(module)
        for name in ('rearc_graph_worker.py','rearc_source_worker.py'):
            shutil.copyfile(Path(__file__).with_name(name), root/name)
        root.chmod(0o755)
        for file in root.iterdir(): file.chmod(0o444)
        for task in scope['selected_ids']:
            for seed in SEEDS:
                name = 'bed-rearc-source-'+uuid.uuid4().hex
                args = ['docker','run','--rm','-i','--name',name,'--network=none','--read-only',
                        '--user=65534:65534','--cap-drop=ALL','--security-opt=no-new-privileges',
                        '--pids-limit=32','--memory=256m','--memory-swap=256m','--cpus=1',
                        '--mount',f'type=bind,src={root},dst=/app,readonly',
                        '--env=PYTHONDONTWRITEBYTECODE=1','--env=PYTHONHASHSEED=0',
                        IMAGE,'python','/app/rearc_source_worker.py']
                row = {'task':task,'seed':seed}
                try:
                    call = subprocess.run(args, input=json.dumps(row).encode(), capture_output=True, timeout=15)
                    row.update(json.loads(call.stdout))
                    row['returncode'] = call.returncode
                except Exception as error:
                    row.update(status='failed', error_type=type(error).__name__)
                finally:
                    subprocess.run(['docker','rm','-f',name], capture_output=True, timeout=10)
                result['rows'].append(row)
                print(task,seed,row['status'],flush=True)
    result['status'] = 'passed' if len(result['rows']) == 16 and all(r['status']=='ok' and r['returncode']==0 for r in result['rows']) else 'failed'
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'])


if __name__ == '__main__':
    main()
