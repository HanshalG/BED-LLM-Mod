"""Bounded no-network full-DSL search worker for the expression qualification."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import uuid
from scripts.herb_search_guidance import weights
from scripts.herb_candidate_expression import to_graph

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / 'results/nonmyopic/herb_full_grammar_20260909'
MANIFEST = ROOT / 'results/nonmyopic/herb_bridge_runtime_20260909/Manifest.toml'
IMAGE = 'julia@sha256:2323a3445e7701cf8f7190293e360e4f67c7e43de24480183f178aa1062dc99b'


def search(proposals, count, max_expansions):
    if (count, max_expansions) not in ((56,50000),(128,100000)):
        raise ValueError('unfrozen search budget')
    metadata = json.loads((GRAMMAR/'source.json').read_text())
    grammar = (GRAMMAR/'grammar.jl').read_text()
    metadata['rules'] = [r.strip() for r in grammar.splitlines() if r.strip().startswith('Value = ')]
    _, probabilities = weights(metadata, proposals)
    name = 'bed-herb-search-'+uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='bed-herb-search-') as directory:
        root = Path(directory)
        (root/'grammar.jl').write_text(grammar)
        (root/'request.toml').write_text('weights = '+json.dumps(probabilities)+f'\ncount = {count}\nmax_expansions = {max_expansions}\n')
        for file in ('herb_search_worker.jl','herb_ordered_iterator.jl'):
            shutil.copyfile(Path(__file__).with_name(file), root/file)
        root.chmod(0o755)
        for file in root.iterdir():
            file.chmod(0o444)
        args = ['docker','run','--rm','--name',name,'--network=none','--read-only','--user=65534:65534',
                '--memory=2g','--memory-swap=2g','--cpus=2','--pids-limit=128','--cap-drop=ALL',
                '--security-opt=no-new-privileges','-e','JULIA_DEPOT_PATH=/work/depot',
                '-v','bed-herb-runtime-20260909:/work:ro',
                '--mount',f'type=bind,src={root},dst=/app,readonly',
                '--mount',f'type=bind,src={root},dst=/input,readonly',IMAGE,
                'timeout','120','julia','--compiled-modules=no','--startup-file=no','--project=/work/env','/app/herb_search_worker.jl']
        try:
            response = subprocess.run(args,capture_output=True,timeout=135)
            if response.returncode or len(response.stdout)>1048576:
                raise RuntimeError(f'search process failed: {response.returncode}')
            result = tomllib.loads(response.stdout.decode())
            if result['manifest_sha256'] != hashlib.sha256(MANIFEST.read_bytes()).hexdigest() or not result['no_api_key']:
                raise ValueError('runtime binding')
            if result['status'] != 'complete':
                return result
            if len(result['expressions']) != count or result['expansions'] > max_expansions:
                raise ValueError('search coverage')
            scores = result['log_weights']
            if len(scores) != count or any(b>a+1e-12 for a,b in zip(scores,scores[1:])):
                raise ValueError('search order')
            functions = {s['name'] for s in metadata['signatures']}
            terminals = {r.removeprefix('Value = ') for r in metadata['rules'] if '(' not in r}
            for expression in result['expressions']:
                to_graph(expression, functions, terminals-functions-{'I'})
            return result
        finally:
            subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)


if __name__ == '__main__':
    print(json.dumps(search([],128,100000),indent=2))
