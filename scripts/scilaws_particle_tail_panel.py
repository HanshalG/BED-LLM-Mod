"""Frozen full-panel test of explicitly tail-resolved residual integration."""
import argparse
import hashlib
import json
from pathlib import Path

from environments.scilaws.tail_quantile_belief import TailQuantileGaussianModel
from scripts.scilaws_particle_correction_audit import run as run_task
from scripts.scilaws_particle_correction_panel import assess, read_bound, REFERENCES, REFERENCE_SHA
from scripts.scilaws_particle_integration_panel import SCENARIOS


def run(output):
    references = read_bound(REFERENCES, REFERENCE_SHA)['cases']
    output = Path(output)
    output.mkdir(exist_ok=False)
    cases, shards = [], []
    for t in range(8):
        for seed in (1304, 1305):
            selected = [c for c in references if c['task_index'] == t and c['seed'] == seed]
            if [c['scenario'] for c in selected] != list(SCENARIOS):
                raise ValueError('reference coverage mismatch')
            path = output / f'task{t}_seed{seed}.json'
            run_task(path, task_index=t, seed=seed, references=selected,
                     reference_sha256=REFERENCE_SHA, model_class=TailQuantileGaussianModel,
                     counts=(16, 32))
            raw = path.read_bytes()
            shards.append(dict(path=path.name, sha256=hashlib.sha256(raw).hexdigest()))
            cases.extend(dict(c, task_index=t, seed=seed) for c in json.loads(raw)['cases'])
            print('completed', t, seed, flush=True)
    result = dict(cases=cases, shards=shards, assessment=assess(cases, references, counts=(16, 32)),
                  reference_sha256=REFERENCE_SHA, source_measurements=0, model_calls=0,
                  paid_cost_usd=0, deployment_authorized=False)
    with (output / 'result.json').open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(parser.parse_args().output)
