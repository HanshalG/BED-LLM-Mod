"""Terminal uncertainty for a fixed outer quadrature, not its integration error."""
import math


def weighted_min_interval(probabilities, intervals, *, correction=0., tolerance=5e-5):
    if not probabilities or len(probabilities) != len(intervals):
        raise ValueError('incomplete branches')
    if (not math.isfinite(correction) or not math.isfinite(tolerance) or tolerance <= 0
            or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isclose(math.fsum(probabilities), 1., rel_tol=0, abs_tol=1e-12)):
        raise ValueError('invalid mass or tolerance')
    bounds = []
    for row in intervals:
        if not row or any(not math.isfinite(lo) or not math.isfinite(hi) or lo > hi for lo, hi in row):
            raise ValueError('invalid action intervals')
        bounds.append((min(lo for lo, _ in row), min(hi for _, hi in row)))
    lower = correction + math.fsum(p*lo for p, (lo, _) in zip(probabilities, bounds))
    upper = correction + math.fsum(p*hi for p, (_, hi) in zip(probabilities, bounds))
    contributions = [p*(hi-lo) for p, (lo, hi) in zip(probabilities, bounds)]
    # Idealized lower work bound: a refinement makes one branch minimum exact.
    order = sorted(range(len(bounds)), key=lambda i: (-contributions[i], i))
    count = 0
    while math.fsum(contributions[i] for i in order[count:]) > 2*tolerance:
        count += 1
    return dict(lower=lower, upper=upper, width=upper-lower,
                midpoint_terminal_error_bound=(upper-lower)/2,
                within_terminal_budget=(upper-lower)/2 <= tolerance,
                ideal_exact_branch_refinements=count,
                refinement_order=order, weighted_widths=contributions,
                outer_error_bounded=False)
