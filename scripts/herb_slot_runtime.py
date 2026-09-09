"""Prospective search runtime with immutable raw output and fixed candidate slots."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import uuid

from scripts.herb_candidate_expression import to_graph
from scripts.herb_safe_guidance import prepare
from scripts.herb_search_runtime import GRAMMAR, MANIFEST, IMAGE


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)


def decode(raw, metadata, count, cap):
    if (count, cap) not in ((56, 50000), (128, 100000)):
        raise ValueError('search budget')
    result = tomllib.loads(raw.decode())
    if (result['manifest_sha256'] != hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
            or result['no_api_key'] is not True):
        raise ValueError('runtime binding')
    if result['status'] != 'complete' or result['failure'] != '':
        raise ValueError('incomplete search')
    expressions, scores = result['expressions'], result['log_weights']
    if (not isinstance(expressions, list) or len(expressions) != count
            or not all(isinstance(e, str) for e in expressions)
            or type(result['expansions']) is not int or not 0 <= result['expansions'] <= cap):
        raise ValueError('search coverage')
    if (not isinstance(scores, list) or len(scores) != count
            or any(type(s) not in (int, float) or not math.isfinite(s) or s > 0 for s in scores)
            or any(b > a+1e-12 for a, b in zip(scores, scores[1:]))):
        raise ValueError('search order')
    functions = {s['name'] for s in metadata['signatures']}
    terminals = {r.removeprefix('Value = ') for r in metadata['rules'] if '(' not in r}
    slots = []
    for index, (expression, score) in enumerate(zip(expressions, scores)):
        try:
            graph = to_graph(expression, functions, terminals-functions-{'I'})
        except ValueError as error:
            slot = {'graph': None, 'conversion_error': str(error)}
        else:
            slot = {'graph': graph, 'conversion_error': None}
        slots.append({'slot': index, 'expression': expression, 'log_search_weight': score, **slot})
    return {'status': 'complete', 'slots': slots, 'expansions': result['expansions'],
            'raw_sha256': hashlib.sha256(raw).hexdigest(), 'replacement_candidates': 0}


def search(proposals, count, cap, bank_dir):
    if (count, cap) not in ((56, 50000), (128, 100000)):
        raise ValueError('search budget')
    bank = Path(bank_dir)
    bank.mkdir(parents=True, exist_ok=False)
    grammar = (GRAMMAR/'grammar.jl').read_text()
    metadata = json.loads((GRAMMAR/'source.json').read_text())
    metadata['rules'] = [r.strip() for r in grammar.splitlines() if r.strip().startswith('Value = ')]
    prepared = prepare(metadata, proposals)
    save(bank/'request.json', {'proposals': proposals, 'guidance_slots': prepared['slots'],
        'weights': prepared['guided'], 'count': count, 'cap': cap,
        'grammar_sha256': hashlib.sha256(grammar.encode()).hexdigest(), 'image': IMAGE})
    name = 'bed-herb-slots-'+uuid.uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix='bed-herb-slots-') as directory:
            root = Path(directory)
            (root/'grammar.jl').write_text(grammar)
            (root/'request.toml').write_text('weights = '+json.dumps(prepared['guided'])+
                f'\ncount = {count}\nmax_expansions = {cap}\n')
            for file in ('herb_search_worker.jl', 'herb_ordered_iterator.jl'):
                shutil.copyfile(Path(__file__).with_name(file), root/file)
            root.chmod(0o755)
            for file in root.iterdir():
                file.chmod(0o444)
            args = ['docker', 'run', '--rm', '--name', name, '--network=none', '--read-only',
                '--user=65534:65534', '--memory=2g', '--memory-swap=2g', '--cpus=2', '--pids-limit=128',
                '--cap-drop=ALL', '--security-opt=no-new-privileges', '-e', 'JULIA_DEPOT_PATH=/work/depot',
                '-v', 'bed-herb-runtime-20260909:/work:ro', '--mount', f'type=bind,src={root},dst=/app,readonly',
                '--mount', f'type=bind,src={root},dst=/input,readonly', IMAGE, 'timeout', '120', 'julia',
                '--compiled-modules=no', '--startup-file=no', '--project=/work/env', '/app/herb_search_worker.jl']
            response = subprocess.run(args, capture_output=True, timeout=135)
            # Preserve the exact ordered prefix even if validation or conversion fails.
            (bank/'stdout.toml').write_bytes(response.stdout)
            (bank/'stderr.txt').write_bytes(response.stderr)
            save(bank/'process.json', {'returncode': response.returncode, 'network': False, 'model_calls': 0})
            if response.returncode or len(response.stdout) > 1048576:
                raise RuntimeError('search process failed')
            result = decode(response.stdout, metadata, count, cap)
            save(bank/'result.json', result)
            return result
    except Exception as error:
        save(bank/'failure.json', {'type': type(error).__name__, 'message': str(error)[:200]})
        raise
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=10)


def replay(bank_dir):
    bank = Path(bank_dir)
    request = json.loads((bank/'request.json').read_text())
    grammar = (GRAMMAR/'grammar.jl').read_text()
    if request['grammar_sha256'] != hashlib.sha256(grammar.encode()).hexdigest() or request['image'] != IMAGE:
        raise ValueError('grammar or image binding')
    metadata = json.loads((GRAMMAR/'source.json').read_text())
    metadata['rules'] = [r.strip() for r in grammar.splitlines() if r.strip().startswith('Value = ')]
    prepared = prepare(metadata, request['proposals'])
    if request['weights'] != prepared['guided'] or request['guidance_slots'] != prepared['slots']:
        raise ValueError('guidance replay')
    if json.loads((bank/'process.json').read_text()) != {'returncode': 0, 'network': False, 'model_calls': 0}:
        raise ValueError('process replay')
    result = decode((bank/'stdout.toml').read_bytes(), metadata, request['count'], request['cap'])
    if result != json.loads((bank/'result.json').read_text()):
        raise ValueError('candidate replay')
    return {'status': 'exact_replay', 'slots': len(result['slots']), 'new_calls': 0}
