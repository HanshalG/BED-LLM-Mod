"""Replay a complete feedback probe, including paired prompt information sets."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from scripts import deepcoder_feedback_probe as g


def verify(root):
    def read(name):
        return json.loads((root/name).read_text())
    r=read('result.json')
    if r['status']!='feedback_screen_complete':
        raise ValueError('complete terminal required')
    for path,sha in r['implementation_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise ValueError('implementation changed')
    if hashlib.sha256(g.PROTOCOL.read_bytes()).hexdigest()!=r['protocol_sha256']:
        raise ValueError('protocol changed')
    d=g.load_dsl()
    public=read('public.json')
    if public!=g.cases(d):
        raise ValueError('source histories/targets differ')
    panel,total={},Decimal()
    for key,case in public.items():
        forecasts,quality={},{}
        initial=None
        seed=37100000+10*int(key)
        for arm in ('initial','feedback','control'):
            tag=key+'_'+arm
            expected=g.luna.bumped(g.cr.request(d,case['history'],seed,history_blind=False) if arm=='initial'
                else g.revision_request(d,initial,case['history'],seed+1,with_feedback=arm=='feedback'))
            if read(tag+'.request.json')!=expected:
                raise ValueError('request/feedback information mismatch')
            raw=read(tag+'.response.json')
            ps=g.constrained.decode(d,g.luna.validate_response(raw))
            total+=Decimal(str(raw['usage']['cost']))
            pool,work=g.expand(d,ps,case['history'])
            quality[arm]=dict(raw_count=len(ps),raw_unique=len({str(p) for p in ps}),
                raw_compatible=sum(c['fits_observed_history'] for c in g.checks(d,ps,case['history'])),
                expanded_support=len(pool),work=work)
            if read(tag+'.support.json')!=dict(programs=[str(p) for p in pool],quality=quality[arm]):
                raise ValueError('support quality mismatch')
            if arm=='initial':
                initial,initial_pool=ps,pool
            forecasts[arm]=g.program_forecast(d,pool if arm=='initial' else initial_pool+pool,case)
        panel[key]=dict(case=case,forecasts=forecasts,quality=quality)
    if panel!=read('forecasts.json') or read('outcomes.json')!=g.outcomes(d,public):
        raise ValueError('forecasts or source endpoints changed')
    scored=g.screen.score_sealed(root/'forecasts.json',r['forecast_sha256'],lambda:read('outcomes.json'))
    if any(r[k]!=v for k,v in scored.items()):
        raise ValueError('score/gate mismatch')
    if total!=Decimal(str(r['accepted_cost_usd'])) or r['calls']!=24 or r['uncertain_exposure_usd']!=0:
        raise ValueError('cost/call mismatch')
    if len(list(root.glob('*.response.json')))!=24 or len(list(root.glob('*.request.json')))!=24:
        raise ValueError('extra/missing requests')
    return dict(status='replay_valid',terminal_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
        accepted_cost_usd=float(total),calls_verified=24,model_calls=0)


if __name__=='__main__':
    print(g.pred.canonical(verify(g.ROOT)))
