"""Read-only model-internal diagnosis from an immutable closed response bank."""
import json
from collections import defaultdict
from pathlib import Path

from scripts.rearc_scene_run import ROOT, SceneBank, replay
from scripts.rearc_python_forecast import forecast_python_slots
from scripts.rearc_predictive_score import outcome
from scripts.rearc_luna_qualification import digest
from scripts.rearc_public_source_journal import save

OUT = Path('results/nonmyopic/REARC_SCENE_OPPORTUNITY_AUDIT_20260909.json')


def summarize(f):
    if f['conditioning']['failed']:
        return {'status': 'unsupported', 'consistent_programs': 0,
                'internal_risk': None, 'planning_ready': False}
    distributions = []
    for row in f['outputs']:
        mass = defaultdict(float)
        for value, weight in zip(row, f['weights']):
            mass[outcome(value)] += weight
        distributions.append({'distinct_positive_outputs': sum(v > 0 for v in mass.values()),
            'failure_probability': mass.get(None, 0.),
            'internal_half_brier_risk': max(0., (1-sum(v*v for v in mass.values()))/2)})
    unanimous = all(d['distinct_positive_outputs'] == 1 and d['failure_probability'] == 0
                    for d in distributions)
    return {'status': 'model_internal_only',
            'consistent_programs': f['conditioning']['consistent_programs'],
            'distributions': distributions, 'unanimous_all_public_inputs': unanimous,
            'actual_correctness': 'unknown_sealed', 'planning_ready': False}


def audit():
    verification = replay(ROOT)
    bank = SceneBank(ROOT, replay=True)
    rows = []
    for i, c in enumerate(json.loads((ROOT/'public.json').read_text())):
        for arm in ('raw', 'inventory'):
            u = json.loads((ROOT/f'update_{i}_{arm}.json').read_text())
            f = forecast_python_slots(u['slots'], {
                'inputs': c['inputs'], 'outputs': c['outputs'],
                'target_inputs': c['query_inputs']+c['target_inputs']}, bank.evaluate_python)
            rows.append({'task_index': i, 'arm': arm, **summarize(f)})
    return {'status': 'saved_model_internal_audit', 'source_result_sha256':
            digest(json.loads((ROOT/'result.json').read_text())), 'replay': verification,
            'rows': rows, 'new_model_calls': 0, 'new_program_executions': 0,
            'actual_future_outputs_opened': False, 'cost_usd': 0,
            'qualification_override': False, 'depth_authorized': False}


if __name__ == '__main__':
    if OUT.exists(): raise FileExistsError(OUT)
    value = audit()
    save(OUT, value)
    print(json.dumps(value))
