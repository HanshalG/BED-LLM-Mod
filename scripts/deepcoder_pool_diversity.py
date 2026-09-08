"""Zero-call, target-label-blind audit of banked Luna compatible pools."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from environments.program_induction import prediction as pred
from scripts import deepcoder_luna_medium_probe as probe


def summarize(forecast, evaluations):
    """Behavioral equivalence is relative to these inputs, never global."""
    if forecast is None:
        return dict(status='no_compatible_support', distinct_behaviors=0,
                    separating_queries=0, max_query_entropy_nats=None)
    pred._validate_forecast(forecast)
    n = len(forecast['candidate_keys'])
    if len(evaluations) != n or any(len(x) != len(forecast['counts']) for x in evaluations):
        raise ValueError('prediction matrix shape mismatch')
    for j, counts in enumerate(forecast['counts']):
        if dict(Counter(row[j] for row in evaluations)) != counts:
            raise ValueError('matrix/forecast disagreement')
    entropies = [-sum((v/n)*math.log(v/n) for v in c.values()) for c in forecast['counts']]
    return dict(status='supported', distinct_behaviors=len(set(map(tuple, evaluations))),
                compatible_programs=n, query_entropy_nats=entropies,
                separating_queries=sum(len(c) > 1 for c in forecast['counts']),
                max_query_entropy_nats=max(entropies),
                interpretation='uniform_syntax_pool_on_public_inputs_not_full_posterior')


def run():
    parent = probe.PARENT/'forecasts.json'
    terminal = probe.ROOT/'result.json'
    expected = 'ea89e2c19378f221e2459347ebd78ebe59c975f20490cf63c8d6a4f10bcb35c8'
    if hashlib.sha256(parent.read_bytes()).hexdigest() != probe.PARENT_SHA or hashlib.sha256(terminal.read_bytes()).hexdigest() != expected:
        raise ValueError('banked input changed')
    panel, result = json.loads(parent.read_text()), json.loads(terminal.read_text())
    dsl = probe.load_dsl()
    rows = {}
    for case, arms in result['results'].items():
        if set(arms) != set(pred.ARMS[:2]):
            continue
        rows[case] = {}
        for arm in pred.ARMS[:2]:
            path = probe.ROOT/(case+'_'+arm+'.response.json')
            raw = json.loads(path.read_text())
            candidates = probe.base.candidates(dsl, probe.validate_response(raw))
            data = panel['cases'][case]
            forecast = probe.base.shared.pool_forecast(candidates, data)
            selected = set(forecast['candidate_keys']) if forecast else set()
            matrix = [[pred.category(fn(x)) for x in data['targets']]
                      for key, fn in candidates if key in selected]
            rows[case][arm] = dict(response_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   forecast=forecast, diversity=summarize(forecast, matrix))
    report = dict(cases=rows, source_result_sha256=expected,
                  parent_sha256=probe.PARENT_SHA, model_calls=0, cost_usd=0,
                  target_labels_opened=False, depth_authorized=False)
    path = Path('results/nonmyopic/DEEPCODER_LUNA_POOL_DIVERSITY_20260909.json')
    probe.save(path, report, exclusive=True)
    print(pred.canonical({c:{a:v['diversity'] for a,v in r.items()} for c,r in rows.items()}))


if __name__ == '__main__':
    run()
