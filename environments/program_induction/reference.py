"""Finite categorical reference using the existing ordinary-horizon planner."""
from collections import Counter
from functools import lru_cache
from time import monotonic

from environments.chembench_mopen.horizon import BeliefBranch, HorizonPlanner, SearchLimits


class ProgramReference:
    def __init__(self, rows, query_count, *, seconds=120):
        self.rows = tuple(tuple(row) for row in rows)
        if (not self.rows or type(query_count) is not int or query_count < 1
                or len({len(row) for row in self.rows}) != 1
                or query_count >= len(self.rows[0])
                or any(type(value) is not str for row in self.rows for value in row)):
            raise ValueError('rectangular string outcomes and disjoint query/target columns required')
        self.num_actions = query_count
        self.initial_state = tuple(range(len(self.rows)))
        self.deadline = monotonic() + seconds
        self.planner = HorizonPlanner(self, limits=SearchLimits(
            max_nodes=100000, max_seconds=5, cache_size=4096, max_depth=4))

    def check(self):
        if monotonic() > self.deadline:
            raise TimeoutError('whole-panel reference budget exceeded')

    @lru_cache(maxsize=4096)
    def risk(self, state):
        self.check()
        if not state:
            raise ValueError('empty posterior')
        n = len(state)
        # Half multiclass Brier Bayes risk, averaged over the SAME target inputs.
        return sum((n*n-sum(c*c for c in Counter(self.rows[i][t] for i in state).values()))
                   / (2*n*n) for t in range(self.num_actions, len(self.rows[0]))) / (
                       len(self.rows[0])-self.num_actions)

    @lru_cache(maxsize=4096)
    def branches(self, state, action):
        self.check()
        if action not in range(self.num_actions) or not state:
            raise ValueError('invalid action or empty posterior')
        groups = {}
        for i in state:
            groups.setdefault(self.rows[i][action], []).append(i)
        return tuple(BeliefBranch(label, len(indices)/len(state), tuple(indices))
                     for label, indices in sorted(groups.items()))

    @lru_cache(maxsize=4096)
    def root(self, state, menu, depth, mode):
        self.check()
        plan = self.planner.plan(state, depth, available=menu, mode=mode)
        return plan.root.action, plan.root.expected_risk

    @lru_cache(maxsize=4096)
    def deployed(self, state, menu, budget, horizon, mode='adaptive'):
        self.check()
        if not budget or not menu:
            return self.risk(state)
        action, _ = self.root(state, menu, min(horizon, budget), mode)
        remaining = tuple(q for q in menu if q != action)
        return sum(b.probability*self.deployed(
            b.state, remaining, budget-1, horizon, mode) for b in self.branches(state, action))

    @lru_cache(maxsize=4096)
    def random(self, state, menu, budget):
        self.check()
        if not budget or not menu:
            return self.risk(state)
        return sum(sum(b.probability*self.random(
            b.state, tuple(q for q in menu if q != action), budget-1)
                       for b in self.branches(state, action)) for action in menu)/len(menu)

    def clear(self):
        for name in ('risk', 'branches', 'root', 'deployed', 'random'):
            getattr(self, name).cache_clear()
