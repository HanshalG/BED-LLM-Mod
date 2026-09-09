"""Reconstruct only the eight saved agent maps; never run physical endpoints."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np

from scripts.discoverphysics_dark_matter_opportunity import posterior_batch
from scripts.discoverphysics_myopic_risk import myopic_prediction_risk

UPSTREAM = Path('external/DiscoverPhysics-replay').resolve()
REVISION = '33b7fa9df96de9c35744efd181ca7e5a8dd60ad5'
MODEL = Path('results/nonmyopic/discoverphysics_dark_matter_structured_replication_v3/discoverphysics-dark-matter-structured-replication-v3-20260728T063000Z/MODEL_FROZEN.json')
MODEL_SHA = '473cf5c883929a2cf8b6d862bebf69e1bb6b6401bea8c7b0edba955e847b7e1d'
OUT = Path('results/nonmyopic/discoverphysics_candidate_reconstruction_20260909')
HELPERS = {
    'scripts/discoverphysics_dark_matter_executable_support.py': ('compile_hypothesis',),
    'scripts/discoverphysics_dark_matter_grounded_policy.py': (
        'compile_support', 'support_prior', 'simulate_maps', 'immediate_eig', '_entropy_rows'),
    'scripts/discoverphysics_dark_matter_opportunity.py': (
        'active_probe_actions', 'heldout_experiments', '_run_to_times'),
    'scripts/discoverphysics_dark_matter_semantic_smoke.py': ('action_table',),
}
ROOT_ACTIONS = {'A': 'r4.5_a3', 'B': 'center', 'C': 'r4.5_a5', 'D': 'r4.5_a1'}


def load_functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names) or len(nodes) != len(names):
        raise ValueError('numerical helper missing or duplicated')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future]+nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)


def main():
    if OUT.exists():
        raise RuntimeError('reconstruction already started')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UPSTREAM, text=True).strip()
    if revision != REVISION or hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA:
        raise ValueError('source or model mismatch')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=UPSTREAM, check=True,
                   stdout=subprocess.DEVNULL)
    data = json.loads(MODEL.read_text())
    if len(data['initial_support']) != 8:
        raise ValueError('wrong initial support')
    OUT.mkdir()
    result = {'status': 'started', 'model_sha256': MODEL_SHA, 'upstream_commit': revision,
              'model_calls': 0, 'physical_endpoint_runs': 0, 'cost_usd': 0,
              'helper_hashes': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in HELPERS}}
    (OUT/'STARTED.json').write_text(json.dumps(result, indent=2)+'\n')
    def timeout(*_):
        raise TimeoutError('300 second reconstruction cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        namespace = {'np': np, 'math': math, 'posterior_batch': posterior_batch,
                     'OBSERVATION_NOISE_STD': .075,
                     'entropy': lambda p: float(-np.sum(p[p > 0]*np.log(p[p > 0])))}
        for path, names in HELPERS.items():
            load_functions(path, names, namespace)
        sys.path.insert(0, str(UPSTREAM/'PhysicsSchool'))
        executor_file = UPSTREAM/'ScienceAgent/scienceagent/executor.py'
        spec = importlib.util.spec_from_file_location('candidate_only_executor', executor_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        maps = namespace['compile_support'](data['initial_support'])
        original_run = namespace['_run_to_times']
        calls = 0
        def guarded_run(executor, **kwargs):
            nonlocal calls
            if not any(np.array_equal(executor._dark_positions_rel, m) for m in maps):
                raise ValueError('attempt to simulate a non-candidate map')
            calls += 1
            if calls > 216:
                raise ValueError('candidate trajectory cap')
            return original_run(executor, **kwargs)
        namespace['_run_to_times'] = guarded_run
        action_ids = list(namespace['action_table']())
        means, targets = namespace['simulate_maps'](module.NBodyDarkMatterExecutor, maps,
                                                   action_ids=action_ids)
        prior = namespace['support_prior'](data['initial_support'])
        if calls != 216 or not np.isfinite(targets).all() or not all(np.isfinite(x).all() for x in means.values()):
            raise ValueError('invalid candidate simulation coverage')
        np.savez(OUT/'CANDIDATES.npz', means=np.stack([means[a] for a in action_ids]),
                 targets=targets, prior=prior, maps=maps, action_ids=np.array(action_ids))
        result['candidate_trajectories'] = calls
        result['cache_sha256'] = hashlib.sha256((OUT/'CANDIDATES.npz').read_bytes()).hexdigest()
        eig = {root: namespace['immediate_eig'](means[action], prior,
                    rng=np.random.default_rng(24510), samples_per_hypothesis=16)
               for root, action in ROOT_ACTIONS.items()}
        old = json.loads(MODEL.with_name('POLICY.json').read_text())['selection']['immediate_eig_nats']
        result['reconstructed_eig'] = eig
        result['max_eig_replay_error'] = max(abs(eig[k]-old[k]) for k in eig)
        if result['max_eig_replay_error'] > 1e-6:
            raise ValueError('old EIG replay mismatch; no risk interpretation')
        result['risk_by_order'] = {str(order): {root: myopic_prediction_risk(
            means[action], targets, prior, .075, order=order) for root, action in ROOT_ACTIONS.items()}
            for order in (16, 32, 64, 128)}
        last, previous = result['risk_by_order']['128'], result['risk_by_order']['64']
        result['risk_max_refinement_delta'] = max(abs(last[k]-previous[k]) for k in last)
        result['status'] = 'complete' if result['risk_max_refinement_delta'] <= 1e-4 else 'quadrature_unresolved'
        if result['status'] == 'complete':
            result['myopic_risk_root'] = min(last, key=lambda k: (last[k], k))
    except Exception as error:
        result.update(status='failed_closed', error=f'{type(error).__name__}: {error}')
    finally:
        signal.alarm(0)
        (OUT/'RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
