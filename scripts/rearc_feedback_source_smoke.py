"""Exactly 24 frozen hash-only checks for the new feedback cohort."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
from scripts.rearc_source_smoke import selected_functions
from scripts.rearc_graph_runtime import IMAGE, DSL_SHA
from scripts.rearc_source_scope import COMMIT, SOURCE

OUT=Path('results/nonmyopic/REARC_FEEDBACK_SOURCE_SMOKE_20260909.json')


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    path=Path('results/nonmyopic/REARC_FEEDBACK_COHORT_20260909.json')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='33f55e646da0e56835fd157278d941560697af9e6cfc438c0ee418b5394a0b40':
        raise ValueError('cohort binding')
    cohort=json.loads(raw)
    files={name:subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':'+name],timeout=20)
           for name in ('dsl.py','utils.py','generators.py','verifiers.py')}
    if hashlib.sha256(files['dsl.py']).hexdigest()!=DSL_SHA: raise ValueError('DSL binding')
    module='from dsl import *\nfrom random import choice,randint,sample,shuffle,uniform\nrng=[]\n'
    module+=selected_functions(files['utils.py'].decode(),{'unifint'})+'\n'
    for filename,prefix in (('generators.py','generate_'),('verifiers.py','verify_')):
        module+=selected_functions(files[filename].decode(),{prefix+t for t in cohort['selected_ids']})+'\n'
    report={'status':'running','source_commit':COMMIT,'image':IMAGE,'rows':[],
            'cohort_sha256':hashlib.sha256(raw).hexdigest(),
            'module_sha256':hashlib.sha256(module.encode()).hexdigest(),
            'source_hashes':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()},
            'model_calls':0,'cost_usd':0,'examples_emitted':False,'paid_authorized':False}
    with OUT.open('x') as f: json.dump(report,f,indent=2)
    try:
        with tempfile.TemporaryDirectory(prefix='bed-rearc-feedback-source-') as directory:
            root=Path(directory)
            (root/'dsl.py').write_bytes(files['dsl.py'])
            (root/'selected_source.py').write_text(module)
            for name in ('rearc_graph_worker.py','rearc_source_worker.py'):
                shutil.copyfile(Path(__file__).with_name(name),root/name)
            root.chmod(0o755)
            for path in root.iterdir(): path.chmod(0o444)
            for task in cohort['selected_ids']:
                for seed in range(33000,33003):
                    name='bed-rearc-feedback-source-'+uuid.uuid4().hex
                    row={'task':task,'seed':seed}
                    try:
                        run=subprocess.run(['docker','run','--rm','-i','--name',name,
                            '--network=none','--read-only','--user=65534:65534','--cap-drop=ALL',
                            '--security-opt=no-new-privileges','--pids-limit=32',
                            '--memory=256m','--memory-swap=256m','--cpus=1',
                            '--env=PYTHONDONTWRITEBYTECODE=1','--env=PYTHONHASHSEED=0',
                            '--mount',f'type=bind,src={root},dst=/app,readonly',IMAGE,
                            'python','/app/rearc_source_worker.py'],input=json.dumps(row).encode(),
                            capture_output=True,timeout=15)
                        if len(run.stdout)>16384: raise ValueError('response size')
                        row.update(json.loads(run.stdout),returncode=run.returncode)
                    except Exception as exc:
                        row.update(status='failed',error_type=type(exc).__name__)
                    finally:
                        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)
                    report['rows'].append(row)
                    OUT.write_text(json.dumps(report,indent=2)+'\n')
                    print(task,seed,row['status'],flush=True)
        report['status']='passed' if len(report['rows'])==24 and all(
            r['status']=='ok' and r.get('returncode')==0 for r in report['rows']) else 'failed'
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__)
    finally:
        OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'])


if __name__=='__main__': main()
