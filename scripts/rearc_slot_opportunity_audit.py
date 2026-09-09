"""Post hoc saved-record audit; no execution, model calls, or new outcomes."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from scripts.rearc_luna_qualification import digest
from scripts.rearc_predictive_score import outcome
from scripts.rearc_slot_run import artifacts

ROOT = Path('results/nonmyopic/rearc_slot_qualification_20260909')
OUT = Path('results/nonmyopic/REARC_SLOT_OPPORTUNITY_AUDIT_20260909.json')


def distribution(values, truth):
    if not values:
        return {'status':'empty_support'}
    counts = Counter(outcome(v) for v in values)
    masses = {key:count/len(values) for key,count in counts.items()}
    return {'status':'complete','distinct_outputs':len(masses),
            'entropy_nats':-sum(p*math.log(p) for p in masses.values()),
            'observed_output_probability':masses.get(outcome(truth),0.),
            'failure_probability':masses.get(None,0.)}


def audit(root):
    report = json.loads((root/'result.json').read_text())
    if report['status'] != 'complete' or report['artifact_sha256'] != artifacts(root):
        raise ValueError('terminal artifact identity')
    cases = json.loads((root/'public.json').read_text())
    forecasts = json.loads((root/'forecasts.json').read_text())
    def cached(graph, x):
        path = root/('execution_'+digest([graph,x])+'.json')
        if not path.exists():
            return False, None
        row = json.loads(path.read_text())
        return True, row['output'] if row['status']=='ok' else None
    rows = []
    for i,case in enumerate(cases):
        initial = json.loads((root/f'update_{i}_initial.json').read_text())['slots']
        unique = {digest(row['graph']):row['graph'] for row in initial if row['graph'] is not None}
        survivors = []
        for graph in unique.values():
            present,predicted = cached(graph,case['inputs'][0])
            if not present:
                raise ValueError('missing initial coverage execution')
            if outcome(predicted)==outcome(case['outputs'][0]):
                survivors.append(graph)
        queries = []
        for j in (1,2):
            values = [cached(graph,case['inputs'][j]) for graph in survivors]
            missing = sum(not present for present,_ in values)
            entry = {'demo_index':j, 'missing_cached_predictions':missing}
            if missing:
                entry['status']='unavailable_without_new_execution'
            else:
                entry.update(distribution([v for _,v in values],case['outputs'][j]))
            queries.append(entry)
        final = {}
        for arm,f in forecasts[i].items():
            distinct = [len({outcome(value) for value,w in zip(values,f['weights']) if w>0})
                        for values in f['outputs']]
            final[arm] = {'consistent_programs':f['conditioning']['consistent_programs'],
                'failed':f['conditioning']['failed'], 'distinct_positive_mass_outputs':distinct}
        updates = {}
        for arm in ('initial','aware','blind'):
            proposal = json.loads((root/f'update_{i}_{arm}.json').read_text())['proposal']
            updates[arm] = {'invalid_slots':sum(s is None for s in proposal['slots']),
                'feedback_statuses':[proposal[k]['status'] for k in ('proposal_feedback','repair_feedback')]}
        rows.append({'task_index':i,'initial_unique_programs':len(unique),
            'demo0_consistent_programs':len(survivors),
            'demo0_posterior_entropy_nats':math.log(len(survivors)) if survivors else None,
            'additional_demo_predictions':queries,'final':final,'proposal_diagnostics':updates})
    return {'status':'saved_record_audit','post_hoc':True,'new_model_calls':0,
        'new_program_executions':0,'new_outcomes':0,'cost_usd':0,
        'source_result_sha256':hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
        'rows':rows,'qualification_passed':report['qualification_passed'],'depth_authorized':False}


if __name__ == '__main__':
    if OUT.exists():
        raise FileExistsError(OUT)
    result = audit(ROOT)
    with OUT.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
