"""Frozen eight-case joint simulator scoring; labels load after validation."""
import hashlib
import json
import math

from . import joint_forecast as joint
from . import prediction as pred
from .rollout_risk import expected_brier

TEACHERS = ('a', 'insertion', 'ab')
UPDATERS = ('filter', 'insertion', 'regenerated', 'repeat')


def validate(panel):
    if set(panel) != {str(i) for i in range(8)}:
        raise ValueError('eight cases required')
    for row in panel.values():
        case = row['case']
        pred._targets(case['targets'])
        if len(case['targets']) != 32 or row['observation']['inputs'] != case['query']:
            raise ValueError('query/target mismatch')
        pred.category(row['observation']['output'])
        if set(row['teachers']) != set(TEACHERS) or set(row['forecasts']) != set(UPDATERS):
            raise ValueError('complete teachers and updaters required')
        for law in row['teachers'].values():
            if law is not None and joint.validate(law) != (1, 32):
                raise ValueError('joint dimensions differ')
        for f in row['forecasts'].values():
            if f is not None:
                if len(f['distributions']) != 32:
                    raise ValueError('forecast dimensions differ')
                for q in f['distributions']:
                    expected_brier(q, q)


def _loss(distributions, labels):
    if distributions is None:
        return dict(brier=1., zero_mass_targets=len(labels), nll=None)
    losses, logs, zeros = [], [], 0
    for q, y in zip(distributions, labels):
        losses.append(expected_brier({y:1.}, q)['expected_loss'])
        mass = q.get(y, 0)
        zeros += mass == 0
        if mass:
            logs.append(-math.log(mass))
    return dict(brier=sum(losses)/len(labels), zero_mass_targets=zeros,
                nll=None if zeros else sum(logs)/len(labels))


def score_sealed(path, sha, loader):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError('endpoint seal mismatch')
    panel = json.loads(data)
    validate(panel)
    outcomes = loader()
    if set(outcomes) != set(panel):
        raise ValueError('incomplete endpoints')
    cases = {}
    for key, row in panel.items():
        truth = outcomes[key]
        if truth['target_inputs'] != row['case']['targets'] or len(truth['outputs']) != 32:
            raise ValueError('endpoint identity mismatch')
        labels = [pred.category(y) for y in truth['outputs']]
        actual = {a:_loss(f['distributions'] if f else None, labels)
                  for a, f in row['forecasts'].items()}
        teachers = {}
        for name, law in row['teachers'].items():
            branch = joint.condition(law, {0:pred.category(row['observation']['output'])}) if law else None
            reference = branch['distributions'] if branch else None
            probability = branch['evidence_probability'] if branch else 0
            risk = {}
            if reference is not None:
                for a, f in row['forecasts'].items():
                    qs = f['distributions'] if f else [None]*32
                    risk[a] = sum(expected_brier(p,q)['expected_loss']
                                  for p,q in zip(reference,qs))/32
            teachers[name] = dict(answer_probability=probability,
                answer_nll=-math.log(probability) if probability else None,
                conditional_targets=_loss(reference, labels), predicted_updater_brier=risk)
        cases[key] = dict(teachers=teachers, actual_updaters=actual)
    means = {t:sum(c['teachers'][t]['conditional_targets']['brier'] for c in cases.values())/8
             for t in TEACHERS}
    joint_pass = (all(c['teachers']['ab']['answer_probability'] > 0 and
                     c['teachers']['ab']['conditional_targets']['zero_mass_targets'] == 0
                     for c in cases.values()) and means['ab'] <= .15 and
                  means['ab'] <= means['a'] and means['ab'] <= means['insertion'])
    informative, correct = [], []
    for k, c in cases.items():
        gap = c['actual_updaters']['repeat']['brier']-c['actual_updaters']['regenerated']['brier']
        if abs(gap) > .01:
            informative.append(k)
            r = c['teachers']['ab']['predicted_updater_brier']
            if r and (r['repeat']-r['regenerated'])*(1 if gap > 0 else -1) > .01:
                correct.append(k)
    ranking_pass = len(informative) >= 3 and correct == informative
    return dict(status='independent_joint_complete', cases=cases,
        mean_conditional_brier=means, informative_pairs=informative, correct_pairs=correct,
        joint_gate=joint_pass, updater_ranking_gate=ranking_pass,
        multi_query_screen_allowed=joint_pass and ranking_pass,
        depth_authorized=False, scientific_pass=False)
