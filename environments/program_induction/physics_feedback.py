"""Public residual/domain feedback and analytically marginalized log scale."""
import ast
import itertools
import json
import math
import numpy as np

from .physics_proposals import messages, SIGMA
from .scalar_expression import ScalarExpression

SCALE_VARIANCE=4.


def calibrate(residuals, sigma=SIGMA, prior_variance=SCALE_VARIANCE):
    """b~Normal(0,tau²), residual_i=b+Normal(0,sigma²); exact Gaussian update."""
    r=np.asarray(residuals,dtype=float)
    if r.ndim!=1 or not len(r) or not np.all(np.isfinite(r)) or not math.isfinite(sigma) or sigma<=0 or not math.isfinite(prior_variance) or prior_variance<=0:
        raise ValueError('invalid calibration inputs')
    n=len(r)
    s2=sigma*sigma
    mean=float(r.mean())
    variance=1/(1/prior_variance+n/s2)
    posterior_mean=variance*n*mean/s2
    quadratic=float(np.sum((r-mean)**2))/s2+n*mean**2/(s2+n*prior_variance)
    logdet=n*math.log(s2)+math.log1p(n*prior_variance/s2)
    return dict(log_scale_mean=posterior_mean,log_scale_variance=variance,
                log_evidence=-.5*(n*math.log(2*math.pi)+logdet+quadratic),
                standardized_residuals=((r-posterior_mean)/sigma).tolist())


def guards(names):
    if not names or len(names)>8 or len(set(names))!=len(names):
        raise ValueError('guard dimension cap')
    axes=[(3,12) if n=='N' else (.5,2.) for n in names]
    points=[dict(zip(names,values)) for values in itertools.product(*axes)]
    center={n:7 if n=='N' else 1. for n in names}
    points.append(center)
    if 'N' in names:
        points += [dict(center,N=n) for n in range(3,13)]
    return list({tuple(p[n] for n in names):p for p in points}.values())


def log_values(expr,names,points):
    f=ScalarExpression(expr,[f'x{i}' for i in range(len(names))])
    values=[]
    for point in points:
        if set(point)!=set(names):
            raise ValueError('invalid point keys')
        try:
            value=f({f'x{i}':point[n] for i,n in enumerate(names)})
            values.append(math.log(value) if value>0 else None)
        except ValueError:
            values.append(None)
    return values


def feedback(expressions,names,history):
    """No endpoint argument: guards depend only on the predeclared public box."""
    if not 1<=len(expressions)<=8 or not history:
        raise ValueError('invalid feedback dimensions')
    points=guards(names)
    result=[]
    for expr in expressions:
        h=log_values(expr,names,[p for p,y in history])
        g=log_values(expr,names,points)
        bad=[p for p,v in zip(points,g) if v is None]
        record=dict(expression=expr,invalid_guard_count=len(bad),
                    invalid_guard_examples=bad[:4],guard_count=len(points),
                    domain_guarantee=False,invalid_history_count=sum(v is None for v in h))
        if all(v is not None for v in h):
            record.update(calibrate([y-p for p,(_,y) in zip(h,history)]))
        result.append(record)
    return result


def revision_messages(names,context,descriptions,history,initial):
    request=messages(names,context,descriptions,history,True)
    payload=json.loads(request[1]['content'])
    payload.update(initial_proposal_diagnostics=feedback(initial,names,history),
                   public_domain={f'x{i}':{'integer_range':[3,12]} if n=='N' else {'positive_range':[.5,2.]}
                                  for i,n in enumerate(names)},
                   log_scale_prior={'mean':0.,'variance':SCALE_VARIANCE})
    for record in payload['initial_proposal_diagnostics']:
        record['invalid_guard_examples']=[{f'x{i}':p[n] for i,n in enumerate(names)}
                                           for p in record['invalid_guard_examples']]
    request[0]['content']+=' Numerical code integrates an independent global log scale for each proposed shape. Use the observed residuals to revise structure. Keep response positive and finite throughout the public domain. Guard checks are not a proof of global validity. Return the same expressions-only JSON schema.'
    request[1]['content']=json.dumps(payload,sort_keys=True,allow_nan=False)
    if len(json.dumps(request).encode())>30000:
        raise ValueError('feedback message cap')
    return request


def calibrated_prediction(expressions,names,history,targets):
    """Restricted candidate mixture with exact scale evidence, never fitted weights."""
    seen=set()
    components=[]
    for expr in expressions:
        key=ast.dump(ScalarExpression(expr,[f'x{i}' for i in range(len(names))]).tree)
        if key in seen:
            continue
        seen.add(key)
        h=log_values(expr,names,[p for p,y in history])
        g=log_values(expr,names,guards(names))
        t=log_values(expr,names,targets)
        if any(v is None for v in h+g+t):
            continue
        fit=calibrate([y-p for p,(_,y) in zip(h,history)])
        components.append(dict(fit,mean=[v+fit['log_scale_mean'] for v in t]))
    if not components:
        return dict(status='empty_support',mean=None)
    logw=np.array([c['log_evidence'] for c in components])
    w=np.exp(logw-max(logw))
    w/=w.sum()
    return dict(status='complete',mean=(w@np.array([c['mean'] for c in components])).tolist(),
                weights=w.tolist(),components=components)
