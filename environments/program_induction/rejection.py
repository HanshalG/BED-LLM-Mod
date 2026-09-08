"""Bounded prior rejection for deterministic executable observations."""
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class RejectionResult:
    particles: tuple
    draws: int
    evaluations: int
    complete: bool


def conditioned_draws(draw, evaluate, history, *, num_particles, max_draws, deadline):
    """Filter independent prior draws by ALL observations; never inspect targets.

    With iid prior draws, matching particles follow the conditional prior law.
    The caller supplies that prior and a deterministic evaluator. Multiplicity
    is retained. A draw cap may leave an incomplete particle set; it must not be
    presented as completed inference. Time exhaustion aborts rather than making
    a latency-selected set appear valid. Counts are work, not evidence estimates.
    """
    for value in (num_particles, max_draws):
        if type(value) is not int or value <= 0:
            raise ValueError('positive integer particle and draw budgets required')
    history = tuple(history)
    if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in history):
        raise ValueError('history must contain (input, outcome) pairs')
    particles, evaluations = [], 0
    for count in range(1, max_draws+1):
        if monotonic() > deadline:
            raise TimeoutError('rejection runtime cap')
        candidate = draw()
        matches = True
        for inp, observed in history:
            evaluations += 1
            if evaluate(candidate, inp) != observed:
                matches = False
                break
        if monotonic() > deadline:
            raise TimeoutError('rejection runtime cap')
        if matches:
            particles.append(candidate)
            if len(particles) == num_particles:
                return RejectionResult(tuple(particles), count, evaluations, True)
    return RejectionResult(tuple(particles), max_draws, evaluations, False)
