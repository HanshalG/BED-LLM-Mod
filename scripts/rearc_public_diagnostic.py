"""Retrospective executable diagnostics on the first already-public example only."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
from scripts.rearc_graph_runtime import IMAGE, DSL_SHA


def diagnose(graph, input_grid):
    source = subprocess.check_output(['git','-C','/private/tmp/bed-rearc-source-audit','show',
        'e5b7f1d06362a76f9d3b8c25154ff1fafca897ce:dsl.py'],timeout=20)
    if hashlib.sha256(source).hexdigest()!=DSL_SHA:
        raise ValueError('DSL binding')
    with tempfile.TemporaryDirectory(prefix='bed-rearc-public-diagnostic-') as directory:
        root = Path(directory)
        (root/'dsl.py').write_bytes(source)
        for name in ('rearc_public_diagnostic_worker.py','rearc_graph_worker.py','rearc_program_graph.py'):
            shutil.copyfile(Path(__file__).with_name(name),root/name)
        root.chmod(0o755)
        for path in root.iterdir(): path.chmod(0o444)
        name = 'bed-rearc-diagnostic-'+uuid.uuid4().hex
        payload = json.dumps({'graph':graph,'input':input_grid}).encode()
        if len(payload)>65536: raise ValueError('payload size')
        try:
            result = subprocess.run(['docker','run','--rm','-i','--name',name,
                '--network=none','--read-only','--user=65534:65534','--cap-drop=ALL',
                '--security-opt=no-new-privileges','--pids-limit=32','--memory=256m',
                '--memory-swap=256m','--cpus=1','--env=PYTHONDONTWRITEBYTECODE=1',
                '--mount',f'type=bind,src={root},dst=/app,readonly',IMAGE,
                'python','/app/rearc_public_diagnostic_worker.py'],input=payload,capture_output=True,timeout=15)
            if result.returncode or len(result.stdout)>65536:
                return {'status':'runtime_failed','returncode':result.returncode}
            return json.loads(result.stdout)
        finally:
            subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)


def main():
    parent = Path('results/nonmyopic/rearc_luna_qualification_20260909')
    output = Path('results/nonmyopic/REARC_PUBLIC_EXECUTION_DIAGNOSIS_20260909.json')
    if output.exists(): raise FileExistsError(output)
    # Load only the exact paid request and response, never the panel/target files.
    request = (parent/'0_initial.request.json').read_bytes()
    response = (parent/'0_initial.response.json').read_bytes()
    if hashlib.sha256(request).hexdigest()!='9ffcdbe1770b8d54a30e37645874879bcfa8bec0a762a1917380f106713f6403':
        raise ValueError('public request identity')
    if hashlib.sha256(response).hexdigest()!='301a5bf8118092368765f7f57d1f7d1d3c3b12ed7ba7785beb5a7d5bf43eb916':
        raise ValueError('response identity')
    public = json.loads(json.loads(request)['messages'][1]['content'])
    graphs = json.loads(json.loads(response)['choices'][0]['message']['content'])['hypotheses']
    result = {'rows':[diagnose(g,public['public_inputs'][0]) for g in graphs],
              'scope':'retrospective execution trace on already-paid public input0 only',
              'model_calls':0,'cost_usd':0,'target_labels_opened':False,
              'qualification_reopened':False,'depth_authorized':False}
    with output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__': main()
