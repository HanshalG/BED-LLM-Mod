"""Replay a completed decomposed screen without paid requests or new tasks."""
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from scripts import deepcoder_decomposed_probe as g
from scripts.deepcoder_opportunity import load_dsl
from environments.program_induction.prediction import canonical


def replay(root):
    result = json.loads((root/'result.json').read_text())
    if result['status'] != 'decomposed_screen_complete':
        raise ValueError('completed screen required before replay')
    for path, sha in result['implementation_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('implementation binding mismatch')
    if result['protocol_sha256'] != g.PROTOCOL_SHA or hashlib.sha256(g.PROTOCOL.read_bytes()).hexdigest()!=g.PROTOCOL_SHA:
        raise ValueError('protocol binding mismatch')
    dsl = load_dsl()
    public = g.cases(dsl)
    if json.loads((root/'public.json').read_text()) != public:
        raise ValueError('public history mismatch')
    seen, cost = [], Decimal()
    def rpc(tag, body):
        nonlocal cost
        if json.loads((root/(tag+'.request.json')).read_text()) != body:
            raise ValueError('request mismatch')
        raw = json.loads((root/(tag+'.response.json')).read_text())
        cost += Decimal(str(raw['usage']['cost']))
        seen.append(tag)
        return raw
    def record(name, obj):
        if json.loads((root/name).read_text()) != obj:
            raise ValueError('execution construction mismatch')
    panel = {k:g.construct(dsl,c,int(k),rpc,record) for k,c in public.items()}
    saved = root/'forecasts.json'
    if json.loads(saved.read_text()) != panel:
        raise ValueError('forecast replay mismatch')
    truth = g.outcomes(dsl,public)
    if json.loads((root/'outcomes.json').read_text()) != truth:
        raise ValueError('endpoint replay mismatch')
    scored = g.screen.score_sealed(saved,result['forecast_sha256'],lambda:truth)
    if any(result[k]!=v for k,v in scored.items()):
        raise ValueError('score replay mismatch')
    if (len(seen)!=96 or result['calls']!=96 or result['uncertain_exposure_usd']!=0
            or float(cost)!=result['accepted_cost_usd'] or cost>Decimal('3.84')
            or {p.name for p in root.glob('*.request.json')}!={t+'.request.json' for t in seen}
            or {p.name for p in root.glob('*.response.json')}!={t+'.response.json' for t in seen}):
        raise ValueError('receipt coverage mismatch')
    return dict(replay_valid=True, calls=96, cost_usd=float(cost),
                result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest())


if __name__=='__main__':
    print(canonical(replay(g.ROOT)))
