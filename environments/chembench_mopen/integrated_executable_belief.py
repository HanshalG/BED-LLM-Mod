"""Conditional structure-mixture moments, not a generative rollout posterior."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .adaptive_parameter_integral import adaptive_parameter_integral
from .executable_belief import ExecutableBeliefPool, _evaluate, _points, _positive_integer
from .ir import INPUT_NAMES, RateLawError
from .parameter_quadrature import IntegrationUnresolved


@dataclass(frozen=True)
class IntegratedLawMoments:
    law_keys: tuple[str, ...]
    law_weights: tuple[float, ...]
    mean: tuple[float, ...]
    variance: tuple[float, ...]
    conditional_log_evidence: float
    history_sha256: str
    evaluated_scalar_nodes: int
    diagnostics: tuple[dict, ...]
    interpretation: str = 'conditional_pool_moments_not_selection_corrected_or_generative'


class IntegratedExecutableBeliefPool(ExecutableBeliefPool):
    """Reuse canonical validation/support ownership; opt-in moment-only fitting.

    The inherited finite-particle snapshot is unchanged. Call moment_snapshot
    explicitly to request numerical integration. No hypotheses are discarded.
    """

    def moment_snapshot(self, *, history_inputs, observations, targets, sigma,
                        max_integration_rows=400000):
        if not self._laws:
            raise ValueError('empty law pool')
        if any(len(item[0].parameters) > 2 for item in self._laws.values()):
            raise IntegrationUnresolved('integration supports at most two parameters per law')
        max_integration_rows = _positive_integer(max_integration_rows, 'max_integration_rows')
        if max_integration_rows > 400000:
            raise ValueError('integration cap exceeds backend maximum')
        history = _points(history_inputs, 'history', allow_empty=True)
        targets = _points(targets, 'targets')
        if len(targets) > 16:
            raise ValueError('at most sixteen predictive coordinates supported')
        observed = np.asarray(observations, float)
        if observed.shape != (len(history),) or not np.isfinite(observed).all():
            raise ValueError('one finite log1p-rate observation per history row required')
        if (isinstance(sigma, bool) or not np.isscalar(sigma)
                or not np.isfinite(sigma) or sigma <= 0):
            raise ValueError('finite positive sigma required')
        work, remaining = 0, max_integration_rows
        diagnostics, means, variances, evidence = [], [], [], []
        for key, (law, tree, node_count) in sorted(self._laws.items()):
            lower, upper = [], []
            for spec in law.parameters:
                transform = np.log if spec.transform == 'log' else float
                lower.append(transform(spec.lower))
                upper.append(transform(spec.upper))

            def evaluate(z, points):
                nonlocal work
                cells = len(z)*len(points)
                cost = cells*node_count
                if work+cost > self.max_scalar_nodes:
                    raise IntegrationUnresolved('total expression evaluation cap')
                if 8*cells*(node_count+16) > self.max_workspace_bytes:
                    raise IntegrationUnresolved('expression workspace cap')
                work += cost
                namespace = {name: points[None, :, i] for i, name in enumerate(INPUT_NAMES)}
                for i, spec in enumerate(law.parameters):
                    namespace[spec.name] = (np.exp(z[:, i]) if spec.transform == 'log' else z[:, i])[:, None]
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    rates = np.broadcast_to(_evaluate(tree, namespace), (len(z), len(points)))
                    if np.any(rates < 0):
                        raise RateLawError('negative rate in integrated support')
                    return np.log1p(rates)

            def likelihood(z):
                predictions = evaluate(z, history)
                return (-.5*((predictions-observed)/sigma)**2
                        -np.log(sigma)-.5*np.log(2*np.pi)).sum(axis=1)

            def predict(z):
                return evaluate(z, targets)

            if remaining <= 0:
                raise IntegrationUnresolved('total parameter row cap')
            result = adaptive_parameter_integral(lower, upper, likelihood, predict,
                output_size=len(targets), max_rows=remaining,
                log_likelihood_bound=-len(history)*(np.log(sigma)+.5*np.log(2*np.pi)))
            if result['status'] != 'agreement':
                raise IntegrationUnresolved(f'law {key}: {result.get("reason", "unresolved")}')
            remaining -= result['evaluated_rows']
            diagnostics.append(result)
            fitted = result['checks'][-1]
            evidence.append(fitted['log_evidence'])
            means.append(fitted['mean'])
            variances.append(fitted['variance'])
        logs = np.asarray(evidence)
        normalizer = np.logaddexp.reduce(logs)
        weights = np.exp(logs-normalizer)
        means, variances = np.asarray(means), np.asarray(variances)
        mean = weights@means
        variance = weights@(variances+(means-mean)**2)
        encoded = json.dumps({'inputs': history.tolist(), 'observations': observed.tolist()},
                             sort_keys=True, allow_nan=False).encode()
        return IntegratedLawMoments(tuple(sorted(self._laws)), tuple(weights), tuple(mean),
            tuple(variance), float(normalizer-np.log(len(logs))),
            hashlib.sha256(encoded).hexdigest(), work, tuple(diagnostics))
