"""Frozen new-cohort source channels, reusing the existing isolated transport."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from scripts.rearc_examples import Examples
from scripts.rearc_source_scope import COMMIT, SOURCE
from scripts.rearc_source_smoke import selected_functions
from scripts.rearc_paired_repair_source_smoke import COHORT_SHA

SMOKE_SHA = '05f2c64083470dec0d27a7765d8f6bd55c583a041b00ed4a3adbb6bb74d560e3'


class PairedRepairExamples(Examples):
    def __enter__(self):
        root = Path('results/nonmyopic')
        raw = (root/'REARC_PAIRED_REPAIR_SOURCE_SMOKE_20260909.json').read_bytes()
        cohort_raw = (root/'REARC_PAIRED_REPAIR_COHORT_20260909.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != SMOKE_SHA or hashlib.sha256(cohort_raw).hexdigest() != COHORT_SHA:
            raise ValueError('source gate binding')
        smoke, self.cohort = json.loads(raw), json.loads(cohort_raw)
        if smoke['status'] != 'passed' or smoke['cohort_sha256'] != COHORT_SHA:
            raise ValueError('source gate closed')
        self.tasks = self.cohort['selected_ids']
        files = {name: subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':'+name],timeout=20)
                 for name in ('dsl.py','utils.py','generators.py','verifiers.py')}
        if {k:hashlib.sha256(v).hexdigest() for k,v in files.items()} != smoke['source_hashes']:
            raise ValueError('source identity')
        module = 'from dsl import *\nfrom random import choice,randint,sample,shuffle,uniform\nrng=[]\n'
        module += selected_functions(files['utils.py'].decode(), {'unifint'})+'\n'
        for filename,prefix in (('generators.py','generate_'),('verifiers.py','verify_')):
            module += selected_functions(files[filename].decode(),{prefix+t for t in self.tasks})+'\n'
        if hashlib.sha256(module.encode()).hexdigest() != smoke['module_sha256']:
            raise ValueError('selected module identity')
        self.temp = tempfile.TemporaryDirectory(prefix='bed-mechanism-examples-')
        self.root = Path(self.temp.name)
        self.dsl = files['dsl.py'].decode()
        (self.root/'dsl.py').write_bytes(files['dsl.py'])
        (self.root/'selected_source.py').write_text(module)
        for name in ('rearc_graph_worker.py','rearc_example_worker.py'):
            shutil.copyfile(Path(__file__).with_name(name),self.root/name)
        self.root.chmod(0o755)
        for file in self.root.iterdir():
            file.chmod(0o444)
        return self

    def public(self):
        cases = []
        for task in self.tasks:
            demos = [self.one(task,seed,'demonstration') for seed in self.cohort['demo_seeds']]
            queries = [self.one(task,seed,'input') for seed in self.cohort['query_seeds']]
            targets = [self.one(task,seed,'input') for seed in self.cohort['target_seeds']]
            cases.append({'inputs':[d['input'] for d in demos], 'outputs':[d['output'] for d in demos],
                          'query_inputs':[q['input'] for q in queries],
                          'target_inputs':[t['input'] for t in targets], 'target_hashes':[t['output_sha256'] for t in queries+targets]})
        return cases

    def targets(self, cases):
        if len(cases) != len(self.tasks):
            raise ValueError('case coverage')
        result = []
        for task,case in zip(self.tasks,cases):
            rows = [self.one(task,seed,'output') for seed in self.cohort['query_seeds']+self.cohort['target_seeds']]
            if [r['output_sha256'] for r in rows] != case['target_hashes']:
                raise ValueError('target identity')
            result.append([r['output'] for r in rows])
        return result
