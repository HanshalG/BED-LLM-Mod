"""Fresh-panel predictive evaluation with restricted source-prior weights."""
from fractions import Fraction
import hashlib
import json
import math

from . import prediction as pred
from .local_support import evaluate
from .prior import restricted_weights

ARMS = ('before_observation', 'filter_only', 'insertion_repair', 'regenerated', 'repeat_control')


def program_forecast(dsl, programs, case):
    programs = {str(p): p for p in programs if all(evaluate(p, h['inputs']) == h['output'] for h in case['history'])}
    if not programs:
        return None
    weights = restricted_weights(dsl, programs.values())
    rows = []
    for x in case['targets']:
        row = {}
        for key, p in programs.items():
            label = pred.category(evaluate(p, x))
            row[label] = row.get(label, Fraction()) + weights[key]
        rows.append({k: float(v) for k, v in row.items()})
    return dict(distributions=rows, support_size=len(programs), interpretation='restricted_syntax_prior_not_full_posterior')


def symbolic_forecast(old):
    if old is None:
        return None
    n = len(old['candidate_keys'])
    return dict(distributions=[{k:v/n for k,v in row.items()} for row in old['counts']],
                support_size=n, interpretation='uniform_symbolic_pool_not_full_posterior')


def validate(panel):
    if set(panel['cases']) != {'0','1','2','3'} or set(panel['forecasts']) != set(panel['cases']):
        raise ValueError('complete four-case panel required')
    for key, case in panel['cases'].items():
        pred._targets(case['targets'])
        if len(case['targets']) != 32 or set(panel['forecasts'][key]) != set(ARMS):
            raise ValueError('complete five-arm target panel required')
        for f in panel['forecasts'][key].values():
            if f is None:
                continue
            if type(f['support_size']) is not int or f['support_size'] < 1 or len(f['distributions']) != 32:
                raise ValueError('invalid forecast dimensions')
            for row in f['distributions']:
                if not row or any(type(v) not in (float, int) or not math.isfinite(v) or v<=0 for v in row.values()) or abs(sum(row.values())-1)>1e-10:
                    raise ValueError('invalid predictive mass')
                for label in row:
                    if label != 'ERROR' and pred.category(json.loads(label)) != label:
                        raise ValueError('invalid output category')


def score_sealed(path, sha, loader):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError('forecast identity mismatch')
    panel = json.loads(data)
    validate(panel)
    outcomes = loader()
    if set(outcomes) != set(panel['cases']):
        raise ValueError('incomplete outcomes')
    rows = {}
    for key, arms in panel['forecasts'].items():
        truth = outcomes[key]
        if truth['target_inputs'] != panel['cases'][key]['targets'] or len(truth['outputs']) != 32:
            raise ValueError('target mismatch')
        labels = [pred.category(y) for y in truth['outputs']]
        rows[key] = {}
        for arm, f in arms.items():
            if f is None:
                rows[key][arm] = dict(brier=1., zero_mass_targets=32, abstained=True, nll=None)
                continue
            losses, nll, zeros = [], [], 0
            for d,y in zip(f['distributions'], labels):
                p = d.get(y,0)
                losses.append(.5*(1+sum(v*v for v in d.values())-2*p))
                zeros += p == 0
                if p:
                    nll.append(-math.log(p))
            rows[key][arm] = dict(brier=sum(losses)/32, zero_mass_targets=zeros,
                abstained=False, nll=None if zeros else sum(nll)/32)
    return dict(status='transition_screen_complete', scores=rows,
                mean_brier={a:sum(r[a]['brier'] for r in rows.values())/4 for a in ARMS},
                depth_authorized=False, scientific_pass=False)
