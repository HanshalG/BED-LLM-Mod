"""Shared decision budget for references at multiple child beliefs."""
from time import monotonic

from environments.chembench_mopen.horizon import SearchLimitExceeded
from .particle_reference import ParticleReference


class SharedParticleReference:
    def __init__(self, *, max_seconds=5., max_evaluations=100000):
        self.start = monotonic()
        self.max_seconds, self.max_evaluations = max_seconds, max_evaluations
        self.evaluations = 0

    def check(self):
        if self.evaluations >= self.max_evaluations or monotonic()-self.start >= self.max_seconds:
            raise SearchLimitExceeded('shared decision reference budget exceeded')

    def charge(self, count):
        self.evaluations += count
        self.check()

    def action(self, model, state, action):
        self.check()
        reference = ParticleReference(model, state,
            max_seconds=self.max_seconds-(monotonic()-self.start),
            max_evaluations=self.max_evaluations-self.evaluations)
        try:
            return reference.action(action)
        finally:
            self.evaluations += reference.evaluations
