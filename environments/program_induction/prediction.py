"""Seal matched finite-pool categorical forecasts before outcome access."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .proposals import _history, _unique_keys


ARMS = ('history_aware', 'history_blind', 'symbolic_search')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def category(value):
    if value is None:
        return 'ERROR'
    if not (type(value) is int and -50 <= value <= 50
            or type(value) is list and len(value) <= 5
            and all(type(x) is int and -50 <= x <= 50 for x in value)):
        raise ValueError('invalid bounded interpreter output')
    return canonical(value)


def _targets(targets):
    if not isinstance(targets, list) or not 1 <= len(targets) <= 64:
        raise ValueError('one to64 fixed targets required')
    return [_history([{'inputs': x, 'output': None}])[0]['inputs'] for x in targets]


def forecast(candidates, history, targets):
    """Candidates are (canonical identity, trusted bounded interpreter callable).

    Uniform weights over unique compatible candidates are a computational pool
    convention, NOT full-grammar posterior probabilities. Callables receive only
    inputs. The runner owns canonical identity and real-history provenance.
    """
    history, targets = _history(history), _targets(targets)
    if not candidates or len(candidates) > 8:
        raise ValueError('one to eight candidates required')
    unique = {}
    for key, evaluate in candidates:
        if not isinstance(key, str) or not key or not callable(evaluate):
            raise ValueError('invalid candidate identity/evaluator')
        unique.setdefault(key, evaluate)
    compatible = []
    evaluations = 0
    for key, evaluate in unique.items():
        fits = True
        for row in history:
            evaluations += 1
            if category(evaluate(row['inputs'])) != category(row['output']):
                fits = False
                break
        if fits:
            compatible.append((key, evaluate))
    if not compatible:
        raise ValueError('no history-compatible candidate; no forecast produced')
    counts = []
    for inp in targets:
        values = []
        for _, evaluate in compatible:
            values.append(category(evaluate(inp)))
            evaluations += 1
        counts.append(dict(sorted(Counter(values).items())))
    return dict(history_sha256=digest(history), target_inputs=targets,
                candidate_keys=[key for key, _ in compatible],
                proposed_unique_count=len(unique), counts=counts,
                interpreter_evaluations=evaluations)


def _validate_forecast(value):
    if not isinstance(value, dict) or set(value) != {
        'history_sha256', 'target_inputs', 'candidate_keys', 'proposed_unique_count',
        'counts', 'interpreter_evaluations',
    }:
        raise ValueError('forecast schema mismatch')
    h = value['history_sha256']
    if type(h) is not str or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
        raise ValueError('invalid history hash')
    targets = _targets(value['target_inputs'])
    keys = value['candidate_keys']
    if (type(keys) is not list or not 1 <= len(keys) <= 8
            or any(type(k) is not str or not k for k in keys) or len(set(keys)) != len(keys)):
        raise ValueError('invalid candidate identities')
    if (type(value['proposed_unique_count']) is not int
            or not len(keys) <= value['proposed_unique_count'] <= 8
            or type(value['interpreter_evaluations']) is not int
            or value['interpreter_evaluations'] < len(keys)*len(targets)):
        raise ValueError('invalid candidate/work counts')
    rows = value['counts']
    if type(rows) is not list or len(rows) != len(targets):
        raise ValueError('target coverage mismatch')
    for row in rows:
        if not isinstance(row, dict) or not row:
            raise ValueError('empty predictive distribution')
        for label, count in row.items():
            if label != 'ERROR':
                try:
                    if category(json.loads(label)) != label:
                        raise ValueError('noncanonical output category')
                except (TypeError, json.JSONDecodeError):
                    raise ValueError('invalid output category') from None
            if type(count) is not int or count <= 0:
                raise ValueError('invalid categorical count')
        if sum(row.values()) != len(keys):
            raise ValueError('categorical mass mismatch')


def _validate(panel):
    if not isinstance(panel, dict) or set(panel) != {'version', 'interpretation', 'cases'}:
        raise ValueError('panel schema mismatch')
    if type(panel['version']) is not int or panel['version'] != 1 or panel['interpretation'] != 'uniform_compatible_pool_not_full_posterior':
        raise ValueError('panel interpretation mismatch')
    if not isinstance(panel['cases'], dict) or not panel['cases']:
        raise ValueError('cases required')
    for key, arms in panel['cases'].items():
        if type(key) is not str or not key or not isinstance(arms, dict) or set(arms) != set(ARMS):
            raise ValueError('each case requires all three control arms')
        for value in arms.values():
            _validate_forecast(value)
        first = arms[ARMS[0]]
        if any(value['history_sha256'] != first['history_sha256']
               or value['target_inputs'] != first['target_inputs'] for value in arms.values()):
            raise ValueError('arms must share real history and ordered targets')


def seal(path, cases):
    panel = dict(version=1, interpretation='uniform_compatible_pool_not_full_posterior', cases=cases)
    _validate(panel)
    data = canonical(panel).encode()
    with Path(path).open('xb') as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def score(path, expected_sha256, load_outcomes):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('sealed prediction identity failed')
    panel = json.loads(data, object_pairs_hook=_unique_keys)
    _validate(panel)
    outcomes = load_outcomes()
    if not isinstance(outcomes, dict) or set(outcomes) != set(panel['cases']):
        raise ValueError('outcomes must cover every sealed case')
    for case, truth in outcomes.items():
        if not isinstance(truth, dict) or set(truth) != {'target_inputs', 'outputs'}:
            raise ValueError('outcome schema mismatch')
        target_inputs = panel['cases'][case][ARMS[0]]['target_inputs']
        if truth['target_inputs'] != target_inputs or type(truth['outputs']) is not list or len(truth['outputs']) != len(target_inputs):
            raise ValueError('outcome target identity mismatch')
        for value in truth['outputs']:
            category(value)
    rows = {}
    for case, arms in panel['cases'].items():
        rows[case] = {}
        for arm, f in arms.items():
            n = len(f['candidate_keys'])
            losses, nll, zeros = [], [], 0
            for counts, y in zip(f['counts'], outcomes[case]['outputs']):
                p = counts.get(category(y), 0)/n
                losses.append(.5*(1+sum((c/n)**2 for c in counts.values())-2*p))
                zeros += p == 0
                if p:
                    nll.append(-math.log(p))
            rows[case][arm] = dict(brier=sum(losses)/len(losses),
                                   nll=None if zeros else sum(nll)/len(nll),
                                   nll_is_infinite=bool(zeros), zero_mass_targets=zeros)
    return dict(cases=rows, paid_calls_authorized=False, scientific_pass=False)
