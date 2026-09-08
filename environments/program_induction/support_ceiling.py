"""Hindsight-only convex-mixture bounds; never a deployable weighting rule."""
import math
import numpy as np
from scipy.optimize import minimize


def oracle_mixture(rows, labels):
    if not labels or any(not isinstance(y,str) for y in labels):
        raise ValueError('categorical targets required')
    if any(len(row)!=len(labels) or any(not isinstance(y,str) for y in row) for row in rows):
        raise ValueError('matched categorical world rows required')
    if not rows:
        return dict(lower=1.,upper=1.,gap=0.,pointwise_floor=1.,
                    distinct_behaviors=0,unsupported_targets=len(labels),abstention=True)
    rows=list(dict.fromkeys(tuple(row) for row in rows))
    n,t=len(rows),len(labels)
    a=np.asarray(rows)
    b=np.mean(a==np.asarray(labels),axis=1)
    q=sum((a[:,j,None]==a[None,:,j]).astype(float) for j in range(t))/t
    def objective(w):
        return .5*(1+w@q@w-2*b@w)
    def jac(w):
        return q@w-b
    result=minimize(objective,np.full(n,1/n),jac=jac,method='SLSQP',
        bounds=[(0,1)]*n,constraints=[dict(type='eq',fun=lambda w:w.sum()-1,
        jac=lambda w:np.ones(n))],options=dict(maxiter=1000,ftol=1e-12))
    if not np.all(np.isfinite(result.x)):
        raise ValueError('nonfinite numerical solution')
    weights=np.maximum(result.x,0)
    if weights.sum()==0:
        raise ValueError('invalid simplex solution')
    weights/=weights.sum()
    # Linearization over the simplex lower-bounds the convex optimum. Report its
    # gap even when the optimizer claims success; no success-flag-only certificate.
    gradient=jac(weights)
    gap=max(0.,float(weights@gradient-gradient.min()))
    loss=float(objective(weights))
    floors=[]
    unsupported=0
    for j,y in enumerate(labels):
        categories={row[j] for row in rows}
        missing=y not in categories
        unsupported+=missing
        floors.append(.5*(1+1/len(categories)) if missing else 0.)
    pointwise=math.fsum(floors)/t
    lower=max(pointwise,0.,loss-gap-1e-10)
    upper=max(0.,loss)
    if lower>upper+1e-9:
        raise ValueError('inconsistent oracle bounds')
    return dict(lower=lower,upper=upper,gap=gap,pointwise_floor=pointwise,
        distinct_behaviors=n,unsupported_targets=unsupported,abstention=False,
        optimizer_success=bool(result.success),iterations=int(result.nit),
        numerically_resolved=upper-lower<=1e-8,
        interpretation='hindsight_weight_optimization_float_bounds_not_formal_interval_proof')
