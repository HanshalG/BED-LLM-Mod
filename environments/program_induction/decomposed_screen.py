"""Fresh decomposed-proposal comparison, not a sequential policy experiment."""
import hashlib
import json

from . import prediction as pred
from .proposals import _history
from .rollout_risk import expected_brier
from .independent_joint_screen import _loss

ARMS = ('subgoal', 'execution', 'whole', 'short')


def validate(panel):
    if set(panel) != {str(i) for i in range(8)}:
        raise ValueError('complete eight-case panel required')
    for row in panel.values():
        if len(_history(row['case']['history'])) != 3:
            raise ValueError('three observations required')
        pred._targets(row['case']['targets'])
        if len(row['case']['targets']) != 32 or set(row['forecasts']) != set(ARMS):
            raise ValueError('complete arms and targets required')
        for f in row['forecasts'].values():
            if f is not None:
                if len(f['distributions']) != 32:
                    raise ValueError('forecast length mismatch')
                for q in f['distributions']:
                    expected_brier(q, q)


def score_sealed(path, sha, loader):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError('forecast seal mismatch')
    panel = json.loads(data)
    validate(panel)
    outcomes = loader()
    if set(outcomes) != set(panel):
        raise ValueError('endpoint coverage mismatch')
    cases = {}
    for key, row in panel.items():
        truth = outcomes[key]
        if truth['target_inputs'] != row['case']['targets'] or len(truth['outputs']) != 32:
            raise ValueError('endpoint identity mismatch')
        cases[key] = {a: _loss(f['distributions'] if f else None,
                             [pred.category(y) for y in truth['outputs']])
                      for a, f in row['forecasts'].items()}
    means = {a: sum(c[a]['brier'] for c in cases.values())/8 for a in ARMS}
    zeros = {a: sum(c[a]['zero_mass_targets'] for c in cases.values()) for a in ARMS}
    # Each prospective candidate is tested against all controls, not selected by its outcome.
    gates = {}
    for candidate in ('subgoal', 'execution'):
        controls = ('whole', 'short')
        gates[candidate] = (all(row['forecasts'][candidate] is not None for row in panel.values())
            and all(means[c] > 0 and means[candidate] <= .9*means[c]
                    and zeros[candidate] <= zeros[c]
                    and sum(r[c]['brier']-r[candidate]['brier'] > .01 for r in cases.values()) >= 3
                    for c in controls))
    return dict(status='decomposed_screen_complete', cases=cases, mean_brier=means,
                zero_mass_targets=zeros, candidate_gates=gates,
                fresh_joint_gate_allowed=any(gates.values()), depth_authorized=False,
                scientific_pass=False)
