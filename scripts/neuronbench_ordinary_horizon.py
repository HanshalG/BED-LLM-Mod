"""Exact ordinary horizons on a banked deterministic table; no model discovery."""
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from time import monotonic

BANK = Path('results/nonmyopic/neuronbench_compositional_mopen_opportunity/development-20260815/RESPONSE_BANK.json')
PROTOCOL = Path('results/nonmyopic/NEURONBENCH_ORDINARY_HORIZON_PROTOCOL_20260909.md')
OUTPUT = Path('results/nonmyopic/NEURONBENCH_ORDINARY_HORIZON_20260909.json')


class Solver:
    def __init__(self, actions, targets, seconds=60, max_states=250000):
        self.actions = actions
        self.targets = targets
        self.deadline = monotonic() + seconds
        self.max_states = max_states
        self.states = 0

    def guard(self):
        self.states += 1
        if self.states > self.max_states or monotonic() > self.deadline:
            raise RuntimeError('frozen resource cap')

    @lru_cache(maxsize=None)
    def groups(self, support, action):
        groups = {}
        for model in support:
            groups.setdefault(self.actions[model][action], []).append(model)
        return tuple(tuple(group) for _, group in sorted(groups.items()))

    @lru_cache(maxsize=None)
    def forecast(self, support):
        return tuple(Fraction(sum(self.targets[i][q] for i in support), len(support))
                     for q in range(len(self.targets[0])))

    def truth_loss(self, support, truth):
        return sum((p - y)**2 for p, y in zip(self.forecast(support), self.targets[truth])) / len(self.targets[0])

    @lru_cache(maxsize=None)
    def risk(self, support):
        return sum(self.truth_loss(support, i) for i in support) / len(support)

    @lru_cache(maxsize=None)
    def value(self, support, available, horizon):
        self.guard()
        if horizon == 0 or not available:
            return self.risk(support)
        return min(self.action_value(support, available, horizon, a) for a in available)

    def action_value(self, support, available, horizon, action):
        rest = tuple(a for a in available if a != action)
        return sum(Fraction(len(group), len(support)) * self.value(group, rest, horizon-1)
                   for group in self.groups(support, action))

    def choose(self, support, available, horizon):
        return min(available, key=lambda a: (self.action_value(support, available, horizon, a), a))

    def replay(self, support, truth, horizon, budget=4):
        available = tuple(range(len(self.actions[0])))
        trace = []
        for remaining in range(budget, 0, -1):
            a = self.choose(support, available, min(horizon, remaining))
            y = self.actions[truth][a]
            support = tuple(i for i in support if self.actions[i][a] == y)
            if not support:
                raise ValueError('empty support')
            trace.append({'action': a, 'observation': y, 'support_size': len(support)})
            available = tuple(i for i in available if i != a)
        return self.truth_loss(support, truth), trace

    @lru_cache(maxsize=None)
    def random_value(self, support, available, remaining, truth):
        self.guard()
        if remaining == 0:
            return self.truth_loss(support, truth)
        return sum(self.random_value(
            tuple(i for i in support if self.actions[i][a] == self.actions[truth][a]),
            tuple(i for i in available if i != a), remaining-1, truth)
            for a in available) / len(available)


def encode(value):
    return {'fraction': str(value), 'value': float(value)}


def run():
    if OUTPUT.exists():
        raise RuntimeError('already banked')
    raw = BANK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '34ba78da58dc8016b855e9b6b3ead4ef79bce0c3732dafb37802a18472a560f4':
        raise ValueError('bank binding changed')
    bank = json.loads(raw)
    payload = dict(bank)
    expected = payload.pop('sha256')
    if hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != expected:
        raise ValueError('payload binding changed')
    actions, targets = bank['action_counts'], bank['query_counts']
    if len(actions) != 22 or any(len(r) != 9 for r in actions) or any(len(r) != 28 for r in targets):
        raise ValueError('bank dimensions')
    if any(type(v) is not int for table in (actions, targets) for row in table for v in row):
        raise ValueError('integer table required')
    truths = tuple(bank['candidate_masks'].index(m) for m in bank['truth_masks'])
    result = {'bank_sha256': hashlib.sha256(raw).hexdigest(),
              'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              'conditions': {}, 'model_calls': 0, 'cost_usd': 0,
              'old_gate_changed': False, 'paid_authorized': False}
    try:
        for name, support in (('all22', tuple(range(22))), ('oracle_pairs15', truths)):
            start = monotonic()
            solver = Solver(actions, targets)
            condition = {'rows': [], 'roots': {}, 'mean_loss': {}}
            for h in (1, 2, 3):
                condition['roots'][str(h)] = solver.choose(support, tuple(range(9)), h)
                losses = []
                for truth in truths:
                    loss, trace = solver.replay(support, truth, h)
                    condition['rows'].append({'truth_mask': bank['candidate_masks'][truth],
                                              'horizon': h, 'loss': encode(loss), 'trace': trace})
                    losses.append(loss)
                condition['mean_loss'][str(h)] = encode(sum(losses)/len(losses))
            random = sum(solver.random_value(support, tuple(range(9)), 4, t) for t in truths) / len(truths)
            condition['random_loss'] = encode(random)
            condition['states'] = solver.states
            condition['seconds'] = monotonic()-start
            result['conditions'][name] = condition
            # Method caches include self; clear them between conditions.
            for method in (solver.groups, solver.forecast, solver.risk, solver.value, solver.random_value):
                method.cache_clear()
        result['status'] = 'descriptive_complete'
    except Exception as error:
        result['status'] = 'failed_closed'
        result['error'] = str(error)
    with OUTPUT.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'conditions'}))
    for name, condition in result['conditions'].items():
        print(name, condition['mean_loss'], condition['random_loss'], condition['roots'])


if __name__ == '__main__':
    run()
