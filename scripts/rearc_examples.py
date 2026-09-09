"""Source collection keeps target labels inside the isolated source worker."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
from scripts.rearc_source_scope import COMMIT,SOURCE
from scripts.rearc_source_smoke import selected_functions
from scripts.rearc_graph_runtime import IMAGE,DSL_SHA


class Examples:
    def __enter__(self):
        self.temp=tempfile.TemporaryDirectory(prefix='bed-rearc-examples-')
        self.root=Path(self.temp.name)
        scope=json.loads(Path('results/nonmyopic/REARC_SOURCE_SCOPE_20260909.json').read_text())
        self.tasks=scope['selected_ids']
        assert scope['source_commit']==COMMIT
        files={name:subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':'+name],timeout=20)
               for name in ('dsl.py','utils.py','generators.py','verifiers.py')}
        assert hashlib.sha256(files['dsl.py']).hexdigest()==DSL_SHA
        assert hashlib.sha256(files['verifiers.py']).hexdigest()==scope['verifiers_sha256']
        self.dsl=files['dsl.py'].decode()
        module='from dsl import *\nfrom random import choice,randint,sample,shuffle,uniform\nrng=[]\n'
        module+=selected_functions(files['utils.py'].decode(),{'unifint'})+'\n'
        for filename,prefix in (('generators.py','generate_'),('verifiers.py','verify_')):
            module+=selected_functions(files[filename].decode(),{prefix+t for t in self.tasks})+'\n'
        (self.root/'dsl.py').write_text(self.dsl)
        (self.root/'selected_source.py').write_text(module)
        for name in ('rearc_graph_worker.py','rearc_example_worker.py'):
            shutil.copyfile(Path(__file__).with_name(name),self.root/name)
        self.root.chmod(0o755)
        for file in self.root.iterdir(): file.chmod(0o444)
        return self

    def __exit__(self,*args):
        self.temp.cleanup()

    def one(self,task,seed,mode):
        name='bed-rearc-example-'+uuid.uuid4().hex
        args=['docker','run','--rm','-i','--name',name,'--network=none','--read-only',
              '--user=65534:65534','--cap-drop=ALL','--security-opt=no-new-privileges',
              '--pids-limit=32','--memory=256m','--memory-swap=256m','--cpus=1',
              '--mount',f'type=bind,src={self.root},dst=/app,readonly',
              '--env=PYTHONDONTWRITEBYTECODE=1','--env=PYTHONHASHSEED=0',IMAGE,
              'python','/app/rearc_example_worker.py']
        try:
            call=subprocess.run(args,input=json.dumps(dict(task=task,seed=seed,mode=mode)).encode(),capture_output=True,timeout=15)
            if call.returncode or len(call.stdout)>16384:
                raise RuntimeError('source generation failed')
            value=json.loads(call.stdout)
            expected={'output_sha256'}|({'input'} if mode!='output' else set())|({'output'} if mode!='input' else set())
            if set(value)!=expected: raise ValueError('source channel violation')
            return value
        finally:
            subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)

    def public(self):
        cases=[]
        for task in self.tasks:
            demos=[self.one(task,seed,'demonstration') for seed in range(31100,31103)]
            targets=[self.one(task,seed,'input') for seed in range(31200,31208)]
            cases.append({'inputs':[r['input'] for r in demos],'outputs':[r['output'] for r in demos],
                          'target_inputs':[r['input'] for r in targets],'target_hashes':[r['output_sha256'] for r in targets]})
        return cases

    def targets(self,cases):
        result=[]
        for task,case in zip(self.tasks,cases):
            outputs=[]
            for j,seed in enumerate(range(31200,31208)):
                row=self.one(task,seed,'output')
                if row['output_sha256']!=case['target_hashes'][j]:
                    raise ValueError('target reconstruction mismatch')
                outputs.append(row['output'])
            result.append(outputs)
        return result
