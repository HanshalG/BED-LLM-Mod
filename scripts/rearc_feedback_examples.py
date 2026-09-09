"""New cohort collector; reuses frozen isolated channel transport without editing it."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from scripts.rearc_examples import Examples
from scripts.rearc_source_scope import COMMIT,SOURCE
from scripts.rearc_source_smoke import selected_functions


class FeedbackExamples(Examples):
    def __enter__(self):
        smoke_path=Path('results/nonmyopic/REARC_FEEDBACK_SOURCE_SMOKE_20260909.json')
        if hashlib.sha256(smoke_path.read_bytes()).hexdigest()!='05600069258e2b66a82452a8f81721c8be1bc89adf6d490d098a17bff14d9f14':
            raise ValueError('source smoke binding')
        smoke=json.loads(smoke_path.read_text())
        path=Path('results/nonmyopic/REARC_FEEDBACK_COHORT_20260909.json')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=smoke['cohort_sha256']:
            raise ValueError('cohort binding')
        self.tasks=json.loads(path.read_text())['selected_ids']
        if smoke['status']!='passed': raise ValueError('source gate closed')
        files={name:subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':'+name],timeout=20)
               for name in ('dsl.py','utils.py','generators.py','verifiers.py')}
        if {k:hashlib.sha256(v).hexdigest() for k,v in files.items()}!=smoke['source_hashes']:
            raise ValueError('source identity')
        module='from dsl import *\nfrom random import choice,randint,sample,shuffle,uniform\nrng=[]\n'
        module+=selected_functions(files['utils.py'].decode(),{'unifint'})+'\n'
        for filename,prefix in (('generators.py','generate_'),('verifiers.py','verify_')):
            module+=selected_functions(files[filename].decode(),{prefix+t for t in self.tasks})+'\n'
        if hashlib.sha256(module.encode()).hexdigest()!=smoke['module_sha256']:
            raise ValueError('selected module identity')
        self.temp=tempfile.TemporaryDirectory(prefix='bed-rearc-feedback-examples-')
        self.root=Path(self.temp.name); self.dsl=files['dsl.py'].decode()
        (self.root/'dsl.py').write_bytes(files['dsl.py'])
        (self.root/'selected_source.py').write_text(module)
        for name in ('rearc_graph_worker.py','rearc_example_worker.py'):
            shutil.copyfile(Path(__file__).with_name(name),self.root/name)
        self.root.chmod(0o755)
        for p in self.root.iterdir(): p.chmod(0o444)
        return self

    def public(self):
        cases=[]
        for task in self.tasks:
            demos=[self.one(task,s,'demonstration') for s in range(33100,33103)]
            targets=[self.one(task,s,'input') for s in range(33200,33208)]
            cases.append({'inputs':[d['input'] for d in demos],'outputs':[d['output'] for d in demos],
                          'target_inputs':[t['input'] for t in targets],'target_hashes':[t['output_sha256'] for t in targets]})
        return cases

    def targets(self,cases):
        if len(cases)!=8: raise ValueError('target coverage')
        result=[]
        for task,case in zip(self.tasks,cases):
            rows=[self.one(task,s,'output') for s in range(33200,33208)]
            if [r['output_sha256'] for r in rows]!=case['target_hashes']: raise ValueError('target identity')
            result.append([r['output'] for r in rows])
        return result
