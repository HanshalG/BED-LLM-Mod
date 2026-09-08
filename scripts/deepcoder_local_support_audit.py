"""Bank a zero-call neighborhood audit without loading target answers."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from environments.program_induction import constrained, prediction as pred
from environments.program_induction.local_support import expand, evaluate
from scripts import deepcoder_luna_medium_probe as probe


def main():
    terminal = probe.ROOT/'result.json'
    if hashlib.sha256(terminal.read_bytes()).hexdigest() != 'ea89e2c19378f221e2459347ebd78ebe59c975f20490cf63c8d6a4f10bcb35c8':
        raise ValueError('terminal identity changed')
    parent = (probe.PARENT/'forecasts.json').read_bytes()
    if hashlib.sha256(parent).hexdigest() != probe.PARENT_SHA:
        raise ValueError('parent identity changed')
    cases = json.loads(parent)['cases']
    completed = json.loads(terminal.read_text())['results']
    dsl = probe.load_dsl()
    rows = {}
    for key, arms in completed.items():
        if set(arms) != set(pred.ARMS[:2]):
            continue
        rows[key] = {}
        for arm in pred.ARMS[:2]:
            raw = json.loads((probe.ROOT/(key+'_'+arm+'.response.json')).read_text())
            roots = constrained.decode(dsl, probe.validate_response(raw))
            pool, work = expand(dsl, roots, cases[key]['history'])
            behaviors = [tuple(pred.category(evaluate(p, x)) for x in cases[key]['targets']) for p in pool]
            counts = [Counter(row[i] for row in behaviors) for i in range(32)]
            entropies = [-sum(v/len(pool)*math.log(v/len(pool)) for v in c.values()) for c in counts] if pool else []
            rows[key][arm] = dict(work=work, programs=[str(p) for p in pool],
                distinct_public_behaviors=len(set(behaviors)),
                separating_queries=sum(len(c)>1 for c in counts),
                max_uniform_query_entropy=max(entropies) if entropies else None)
    report = dict(cases=rows, model_calls=0, cost_usd=0, target_answers_opened=False,
                  interpretation='one_edit_restricted_support_not_full_posterior', depth_authorized=False)
    probe.save(Path('results/nonmyopic/DEEPCODER_LOCAL_SUPPORT_AUDIT_20260909.json'), report, exclusive=True)
    print(pred.canonical({k:{a:{q:v for q,v in r.items() if q!='programs'} for a,r in arms.items()} for k,arms in rows.items()}))


if __name__ == '__main__':
    main()
