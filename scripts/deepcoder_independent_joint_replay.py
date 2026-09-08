"""Verify a terminal joint screen, never open endpoints from an active run."""
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from scripts import deepcoder_independent_joint as g


def verify(root):
    def read(name):
        return json.loads((root/name).read_text())
    result = read('result.json')
    if result['status'] != 'independent_joint_complete':
        raise ValueError('complete terminal required before endpoint replay')
    for path, sha in result['implementation_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('implementation binding changed')
    if hashlib.sha256(g.PROTOCOL.read_bytes()).hexdigest() != result['protocol_sha256']:
        raise ValueError('protocol binding changed')
    dsl = g.load_dsl()
    public = read('public.json')
    if public != g.cases(dsl):
        raise ValueError('source case identity mismatch')
    panel, total = {}, Decimal(0)
    for key, case in public.items():
        pools = {}
        obs = read(key+'.observation.json')
        if obs != g.observe(dsl,key,case['query']):
            raise ValueError('source observation mismatch')
        for arm in ('a','b','regenerated','repeat'):
            tag = key+'_'+arm
            history = case['history']+[obs] if arm=='regenerated' else case['history']
            seed = 34100000+10*int(key)+{'a':0,'b':1}.get(arm,2)
            expected = g.luna.bumped(g.cr.request(dsl,history,seed,history_blind=False))
            if read(tag+'.request.json') != expected:
                raise ValueError('request history/seed/schema mismatch')
            raw = read(tag+'.response.json')
            programs = g.constrained.decode(dsl,g.luna.validate_response(raw))
            total += g.money(raw['usage']['cost'])
            pool,work = g.expand(dsl,programs,history)
            if read(tag+'.support.json') != dict(programs=[str(p) for p in pool],work=work):
                raise ValueError('support reconstruction mismatch')
            pools[arm] = pool
        insertion,_ = g.prepare(dsl,pools['a'],case['history'])
        teachers = {name:g.joint.forecast(dsl,ps,case['history'],[case['query']],case['targets'])
                    for name,ps in [('a',pools['a']),('insertion',insertion),('ab',pools['a']+pools['b'])]}
        path = root/(key+'.preanswer.json')
        if (read(path.name) != dict(case=case,teachers=teachers) or
                hashlib.sha256(path.read_bytes()).hexdigest() != read(key+'.preanswer.seal.json')['sha256']):
            raise ValueError('pre-answer reconstruction/seal mismatch')
        branch = dict(case,history=case['history']+[obs])
        forecasts = {name:g.program_forecast(dsl,ps,branch) for name,ps in [
            ('filter',pools['a']),('insertion',insertion),
            ('regenerated',pools['a']+pools['regenerated']),('repeat',pools['a']+pools['repeat'])]}
        panel[key] = dict(case=case,observation=obs,teachers=teachers,forecasts=forecasts)
    if read('forecasts.json') != panel:
        raise ValueError('endpoint forecasts mismatch')
    if read('outcomes.json') != g.outcomes(dsl,public):
        raise ValueError('source targets mismatch')
    scored = g.scoring.score_sealed(root/'forecasts.json',result['forecast_sha256'],lambda:read('outcomes.json'))
    if any(result[k] != value for k,value in scored.items()):
        raise ValueError('terminal scores/gates mismatch')
    if total != g.money(result['accepted_cost_usd']) or result['calls'] != 32 or result['uncertain_exposure_usd'] != 0:
        raise ValueError('call/cost mismatch')
    return dict(status='replay_valid',terminal_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
                calls_verified=32,accepted_cost_usd=float(total),model_calls=0)


if __name__ == '__main__':
    print(g.pred.canonical(verify(g.ROOT)))
