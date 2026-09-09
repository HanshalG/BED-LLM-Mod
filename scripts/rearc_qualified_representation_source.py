"""Fresh exact-schedule source gate; no extrapolation from separate smoke seeds."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from scripts.rearc_examples import Examples
from scripts.rearc_paired_repair_examples import PairedRepairExamples
from scripts.rearc_source_scope import SOURCE,COMMIT
from scripts.rearc_source_smoke import selected_functions
from scripts.rearc_graph_runtime import DSL_SHA
from scripts.rearc_public_source_journal import collect,replay,save
from scripts.rearc_public_source_transport import dispatch
from scripts.rearc_source_qualified_pool import qualify, public_cases as pool_cases

BASE=Path('results/nonmyopic')
COHORT=BASE/'REARC_QUALIFIED_REPRESENTATION_CANDIDATES_20260909.json'
ROOT=BASE/'rearc_qualified_representation_source_20260909'


def make_cohort():
    inventory=json.loads((BASE/'REARC_SOURCE_SCOPE_20260909.json').read_text())['all_ids']
    old=json.loads((BASE/'REARC_REPRESENTATION_COHORT_20260909.json').read_text())
    excluded=sorted(set(old['excluded_ids']+old['selected_ids']))
    if len(excluded)!=50:raise ValueError('closed cohort coverage')
    selected=sorted(set(inventory)-set(excluded),key=lambda k:hashlib.sha256(('bed-rearc-source-qualified-v1:'+k).encode()).digest())[:24]
    value={'source_commit':COMMIT,'excluded_ids':excluded,'selected_ids':selected,
        'demo_seeds':[48100],'query_seeds':[48200,48201],'target_seeds':list(range(48300,48308)),
        'selection':'first24 SHA256(bed-rearc-source-qualified-v1:+id), no replacement'}
    save(COHORT,value)
    return value


def schedule(cohort):
    return [{'task':task,'seed':seed,'mode':mode} for task in cohort['selected_ids']
        for mode,seeds in [('demonstration',cohort['demo_seeds']),
            ('input',cohort['query_seeds']+cohort['target_seeds'])] for seed in seeds]


class QualifiedExamples(Examples):
    targets=PairedRepairExamples.targets

    def __enter__(self):
        self.cohort=json.loads(COHORT.read_text())
        self.tasks=self.cohort['selected_ids']
        if self.cohort['source_commit']!=COMMIT:raise ValueError('source commit')
        files={name:subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':'+name],timeout=20)
            for name in ('dsl.py','utils.py','generators.py','verifiers.py')}
        if hashlib.sha256(files['dsl.py']).hexdigest()!=DSL_SHA:raise ValueError('DSL identity')
        module='from dsl import *\nfrom random import choice,randint,sample,shuffle,uniform\nrng=[]\n'
        module+=selected_functions(files['utils.py'].decode(),{'unifint'})+'\n'
        for filename,prefix in [('generators.py','generate_'),('verifiers.py','verify_')]:
            module+=selected_functions(files[filename].decode(),{prefix+t for t in self.tasks})+'\n'
        self.bindings={'source_commit':COMMIT,'cohort_sha256':hashlib.sha256(COHORT.read_bytes()).hexdigest(),
            'source_hashes':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()},
            'module_sha256':hashlib.sha256(module.encode()).hexdigest()}
        self.temp=tempfile.TemporaryDirectory(prefix='bed-representation-source-')
        self.root=Path(self.temp.name)
        self.dsl=files['dsl.py'].decode()
        (self.root/'dsl.py').write_bytes(files['dsl.py'])
        (self.root/'selected_source.py').write_text(module)
        for name in ('rearc_graph_worker.py','rearc_public_source_worker.py','rearc_example_worker.py'):
            shutil.copyfile(Path('scripts',name),self.root/name)
        self.root.chmod(0o755)
        for p in self.root.iterdir():p.chmod(0o444)
        return self


def main():
    ROOT.mkdir(exist_ok=False)
    with QualifiedExamples() as source:
        save(ROOT/'bindings.json',source.bindings)
        all_rows=schedule(source.cohort)
        schedules=[all_rows[i*11:(i+1)*11] for i in range(24)]
        result=qualify(ROOT/'pool',schedules,lambda r:dispatch(source.root,r),needed=6)
        if result['status']=='source_pool_qualified':
            save(ROOT/'public.json',pool_cases(ROOT/'pool'))
            selected={**source.cohort,'selected_ids':[r['task'] for r in result['accepted']],
                'candidate_manifest_sha256':source.bindings['cohort_sha256'],
                'selection':'first six exact-schedule reference-valid tasks in frozen candidate order'}
            save(ROOT/'selected_cohort.json',selected)
        save(ROOT/'result.json',result)
        print(json.dumps(result))


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--select']:print(json.dumps(make_cohort()))
    elif not sys.argv[1:]:main()
    else:raise ValueError('unknown argument')
