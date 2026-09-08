"""Sealed prospective predictive endpoint for public execution feedback."""
import hashlib
import json

from . import prediction as pred
from .independent_joint_screen import _loss
from .rollout_risk import expected_brier

ARMS = ('initial','feedback','control')


def validate(panel):
    if set(panel) != {str(i) for i in range(8)}:
        raise ValueError('eight complete cases required')
    for row in panel.values():
        pred._targets(row['case']['targets'])
        if len(row['case']['targets']) != 32 or set(row['forecasts']) != set(ARMS):
            raise ValueError('complete matched forecasts required')
        for f in row['forecasts'].values():
            if f is not None:
                if len(f['distributions']) != 32:
                    raise ValueError('target dimensions')
                for q in f['distributions']:
                    expected_brier(q,q)


def score_sealed(path,sha,loader):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha:
        raise ValueError('forecast seal mismatch')
    panel=json.loads(data)
    validate(panel)
    outcomes=loader()
    if set(outcomes)!=set(panel):
        raise ValueError('incomplete endpoints')
    scores={}
    for key,row in panel.items():
        truth=outcomes[key]
        if truth['target_inputs']!=row['case']['targets'] or len(truth['outputs'])!=32:
            raise ValueError('target identity mismatch')
        labels=[pred.category(y) for y in truth['outputs']]
        scores[key]={a:_loss(f['distributions'] if f else None,labels)
                     for a,f in row['forecasts'].items()}
    means={a:sum(r[a]['brier'] for r in scores.values())/8 for a in ARMS}
    zeros={a:sum(r[a]['zero_mass_targets'] for r in scores.values()) for a in ARMS}
    wins=[k for k,r in scores.items() if r['control']['brier']-r['feedback']['brier']>.01]
    coverage=sum(row['forecasts']['feedback'] is not None for row in panel.values())
    passed=(coverage==8 and means['control']>0 and means['feedback']<=.9*means['control']
            and zeros['feedback']<=zeros['control'] and len(wins)>=3)
    return dict(status='feedback_screen_complete',scores=scores,mean_brier=means,
        zero_mass_targets=zeros,feedback_coverage=coverage,feedback_wins=wins,
        feedback_gate=passed,joint_validation_allowed=passed,depth_authorized=False,scientific_pass=False)
