"""Source-only expressivity audit of the frozen disjoint feedback cohort."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from scripts.rearc_program_graph import exports, source_graph


def main():
    cohort_path=Path('results/nonmyopic/REARC_FEEDBACK_COHORT_20260909.json')
    cohort=json.loads(cohort_path.read_text())
    files={name:subprocess.check_output(['git','-C','/private/tmp/bed-rearc-source-audit',
           'show',cohort['source_commit']+':'+name],timeout=20)
           for name in ('dsl.py','generators.py','verifiers.py')}
    functions,constants=exports(files['dsl.py'].decode())
    trees={name:{n.name:n for n in ast.parse(raw.decode()).body if isinstance(n,ast.FunctionDef)}
           for name,raw in files.items()}
    rows=[]
    for task in cohort['selected_ids']:
        verifier=ast.get_source_segment(files['verifiers.py'].decode(),trees['verifiers.py']['verify_'+task])
        generator=ast.get_source_segment(files['generators.py'].decode(),trees['generators.py']['generate_'+task])
        graph=source_graph(verifier,functions,constants)
        rows.append({'task':task,'reference_steps':len(graph['steps']),
            'first_operation':graph['steps'][0]['op'],
            'generator_sha256':hashlib.sha256(generator.encode()).hexdigest(),
            'verifier_sha256':hashlib.sha256(verifier.encode()).hexdigest(),
            'graph_compatible':True})
    result={'status':'source_graphs_compatible','rows':rows,
        'cohort_sha256':hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        'source_hashes':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},
        'examples_generated':0,'model_calls':0,'cost_usd':0,'paid_authorized':False,
        'scope':'syntax/reference expressivity only; source generation and calibration untested'}
    with Path('results/nonmyopic/REARC_FEEDBACK_SOURCE_AUDIT_20260909.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__': main()
